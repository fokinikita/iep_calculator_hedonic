"""Linear hedonic model with time dummies (the classic time-dummy hedonic index).

    log(total price) = a + b·x (flat features) + c[category levels]
                       + d[region, month]          one dummy per month and region (Москва / область)
                       + e[okrug, month]           Moscow административный округ, relative to the region
                       + f[local area, month]      район (Moscow) / городской округ (oblast), relative to that

- d is free: the region index is exp(d[region, month]) (base month = 0).
- e and f are local corrections. Their change from month to month is penalised (a random-walk prior:
  sum of (e[t] - e[t-1])^2), so a small area follows its region unless its own sales show otherwise,
  and the local curve has no jumps from a few sales.
- The time effect does not depend on the flat's other features, so every flat in an area has the
  same curve: no flat-specific noise over time (the problem of the boosting curves).
- Monotone: coefficients of total / kitchen / living area (and log total area), balconies,
  bathrooms and condition level are kept >= 0, as the boosting model's monotone constraints.

Fitted by ridge-type normal equations (sparse design, dense p x p system), with an active-set step
for the sign constraints. The fitted model is a small JSON (see `to_json`) that the app evaluates.
"""

import logging
import math

import numpy as np
import polars as pl
import scipy.sparse as sp

from flats_pipeline.modeling.monotone import MODEL_TARGET, MONOTONE_INCREASING

logger = logging.getLogger(__name__)

LOG_AREA = "log_total_square"
MIN_LEVEL_COUNT = 50  # rarer category levels share one "__other__" coefficient
MIN_LOCAL_SALES = 200  # areas with fewer training sales have no own time correction
PENALTY_FEATURES = 1.0  # tiny ridge on features (identifies the full one-hot sets)
PENALTY_REGION_TIME = 1e-3  # region x month: practically free
PENALTY_LOCAL_SMOOTH = 3000.0  # okrug / local x month: penalty on month-to-month change (prior sd ~0.3%/month)
OTHER = "__other__"


def _region_key(df: pl.DataFrame) -> pl.Series:
    return df["region_type"].cast(pl.String).fill_null("unknown")


class TimeDummyModel:
    def __init__(self, continuous: list[str], categorical: list[str], months: list[str], time_col: str):
        self.numeric = [c for c in continuous] + [LOG_AREA]
        self.categorical = list(categorical)
        self.months = list(months)
        self.time_col = time_col
        self.monotone = [c for c in MONOTONE_INCREASING if c in self.numeric] + [LOG_AREA]

    # --- design -------------------------------------------------------------------------
    def _numeric_frame(self, df: pl.DataFrame) -> np.ndarray:
        x = df.with_columns(pl.col("total_square").cast(pl.Float64).log().alias(LOG_AREA)).select(self.numeric)
        return x.cast(pl.Float64).to_numpy()

    def _setup(self, df: pl.DataFrame) -> None:
        x = self._numeric_frame(df)
        self.mean = x.mean(axis=0)
        self.scale = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        self.levels = {}
        for c in self.categorical:
            vc = df[c].cast(pl.String).value_counts()
            keep = vc.filter(pl.col("count") >= MIN_LEVEL_COUNT)[c].to_list()
            self.levels[c] = sorted(keep) + [OTHER]
        self.regions = sorted(_region_key(df).unique().to_list())
        okrug = df.filter(pl.col("area_okrug").is_not_null()).group_by("area_okrug").len()
        self.okrugs = sorted(okrug.filter(pl.col("len") >= MIN_LOCAL_SALES)["area_okrug"].to_list())
        local = df.filter(pl.col("local_area") != "missing").group_by("local_area").len()
        self.locals = sorted(local.filter(pl.col("len") >= MIN_LOCAL_SALES)["local_area"].to_list())
        self._layout()

    def _layout(self) -> None:
        """Column positions: intercept, numeric, one-hot levels, then time blocks."""
        p = 1
        self.num_slice = slice(p, p + len(self.numeric))
        p += len(self.numeric)
        self.cat_offset = {}
        for c in self.categorical:
            self.cat_offset[c] = p
            p += len(self.levels[c])
        T = len(self.months) - 1  # base month has no column
        self.time_offset = {}
        for kind, keys in (("region", self.regions), ("okrug", self.okrugs), ("local", self.locals)):
            for k in keys:
                self.time_offset[(kind, k)] = p
                p += T
        self.p = p

    def design(self, df: pl.DataFrame) -> sp.csr_matrix:
        n = df.height
        blocks_rows, blocks_cols, blocks_vals = [], [], []

        def add(rows, cols, vals):
            blocks_rows.append(rows)
            blocks_cols.append(cols)
            blocks_vals.append(vals)

        rows = np.arange(n)
        add(rows, np.zeros(n, dtype=np.int64), np.ones(n))
        x = (self._numeric_frame(df) - self.mean) / self.scale
        for j in range(len(self.numeric)):
            add(rows, np.full(n, self.num_slice.start + j), x[:, j])
        for c in self.categorical:
            lv = self.levels[c]
            index = {v: i for i, v in enumerate(lv)}
            vals = df[c].cast(pl.String).to_list()
            idx = np.fromiter((index.get(v, index[OTHER]) for v in vals), dtype=np.int64, count=n)
            add(rows, self.cat_offset[c] + idx, np.ones(n))
        t = df[self.time_col].replace_strict({m: i for i, m in enumerate(self.months)}, return_dtype=pl.Int64).to_numpy()
        has_t = t > 0
        for kind, keys_series in (
            ("region", _region_key(df)),
            ("okrug", df["area_okrug"].cast(pl.String)),
            ("local", df["local_area"].cast(pl.String)),
        ):
            keys = keys_series.to_list()
            off = np.fromiter((self.time_offset.get((kind, k), -1) for k in keys), dtype=np.int64, count=n)
            ok = has_t & (off >= 0)
            add(rows[ok], off[ok] + t[ok] - 1, np.ones(int(ok.sum())))
        r = np.concatenate(blocks_rows)
        c = np.concatenate(blocks_cols)
        v = np.concatenate(blocks_vals)
        return sp.csr_matrix((v, (r, c)), shape=(n, self.p))

    def _penalty(self) -> np.ndarray:
        P = np.zeros((self.p, self.p))
        for j in range(self.num_slice.start, self.p):
            P[j, j] = PENALTY_FEATURES
        T = len(self.months) - 1
        # first-difference penalty for local time effects: (g1-0)^2 + (g2-g1)^2 + ...
        D = np.eye(T) - np.eye(T, k=-1)
        smooth = PENALTY_LOCAL_SMOOTH * D.T @ D
        for (kind, _), off in self.time_offset.items():
            block = slice(off, off + T)
            if kind == "region":
                P[block, block] = np.eye(T) * PENALTY_REGION_TIME
            else:
                P[block, block] = smooth
        return P

    # --- fit ------------------------------------------------------------------------------
    def fit(self, df: pl.DataFrame, chunk: int = 50_000) -> "TimeDummyModel":
        self._setup(df)
        H = self._penalty()
        g = np.zeros(self.p)
        y = df[MODEL_TARGET].to_numpy()
        for i in range(0, df.height, chunk):
            A = self.design(df.slice(i, chunk))
            H += (A.T @ A).toarray()
            g += A.T @ y[i:i + chunk]
        self.coef = self._solve_monotone(H, g)
        # median log residual = 0 (as the boosting model's calibration)
        resid = y - self.predict_log(df)
        self.shift = float(np.median(resid))
        self.coef[0] += self.shift
        logger.info("Time-dummy model: %d columns, calibration shift %.4f", self.p, self.shift)
        return self

    def _solve_monotone(self, H: np.ndarray, g: np.ndarray) -> np.ndarray:
        """min x'Hx - 2g'x  s.t. monotone coefficients >= 0 (active set over the few constrained ones)."""
        constrained = [self.num_slice.start + self.numeric.index(c) for c in self.monotone]
        fixed: set[int] = set()
        for _ in range(20):
            free = np.array([j for j in range(self.p) if j not in fixed])
            x = np.zeros(self.p)
            x[free] = np.linalg.solve(H[np.ix_(free, free)], g[free])
            neg = [j for j in constrained if j not in fixed and x[j] < 0]
            # a fixed coefficient is released if the objective would fall by raising it
            grad = g - H @ x
            release = [j for j in fixed if grad[j] > 1e-9]
            if not neg and not release:
                break
            fixed |= set(neg)
            fixed -= set(release)
        if fixed:
            logger.info("Time-dummy model: kept at 0 by the sign constraints: %s",
                        [self.numeric[j - self.num_slice.start] for j in sorted(fixed)])
        return x

    def predict_log(self, df: pl.DataFrame, chunk: int = 100_000) -> np.ndarray:
        out = []
        for i in range(0, df.height, chunk):
            out.append(self.design(df.slice(i, chunk)) @ self.coef)
        return np.concatenate(out)

    def predict_price_sq(self, df: pl.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df)) / df["total_square"].to_numpy()

    # --- indices ----------------------------------------------------------------------------
    def time_effects(self, kind: str, key: str) -> list[float]:
        off = self.time_offset.get((kind, key))
        if off is None:
            return [0.0] * len(self.months)
        return [0.0] + [float(v) for v in self.coef[off:off + len(self.months) - 1]]

    # --- export -----------------------------------------------------------------------------
    @classmethod
    def from_json(cls, spec: dict, time_col: str) -> "TimeDummyModel":
        names = [f["name"] for f in spec["numeric"]]
        m = cls(names[:-1], list(spec["categorical"]), spec["months"], time_col)
        m.mean = np.array([f["mean"] for f in spec["numeric"]])
        m.scale = np.array([f["scale"] for f in spec["numeric"]])
        m.levels = {c: list(levels) for c, levels in spec["categorical"].items()}
        m.regions, m.okrugs, m.locals = (list(spec["time"][k]) for k in ("region", "okrug", "local"))
        m._layout()
        m.shift = spec["calibration_shift"]
        coef = np.zeros(m.p)
        coef[0] = spec["intercept"]
        coef[m.num_slice] = [f["coef"] for f in spec["numeric"]]
        for c, levels in spec["categorical"].items():
            coef[m.cat_offset[c]:m.cat_offset[c] + len(levels)] = list(levels.values())
        for kind in ("region", "okrug", "local"):
            for k, effects in spec["time"][kind].items():
                off = m.time_offset[(kind, k)]
                coef[off:off + len(m.months) - 1] = effects[1:]
        m.coef = coef
        return m

    def to_json(self) -> dict:
        """Everything the app needs to evaluate the model (no numpy / scipy objects)."""
        ns = self.num_slice.start
        return {
            "kind": "time_dummy_linear",
            "target": MODEL_TARGET,
            "months": self.months,
            "intercept": float(self.coef[0]),
            "numeric": [
                {"name": c, "mean": float(self.mean[j]), "scale": float(self.scale[j]), "coef": float(self.coef[ns + j])}
                for j, c in enumerate(self.numeric)
            ],
            "log_area_feature": LOG_AREA,
            "categorical": {
                c: {lv: float(self.coef[self.cat_offset[c] + i]) for i, lv in enumerate(self.levels[c])}
                for c in self.categorical
            },
            "other_level": OTHER,
            "time": {
                kind: {k: self.time_effects(kind, k) for (kd, k) in self.time_offset if kd == kind}
                for kind in ("region", "okrug", "local")
            },
            "time_keys": {"region": "region_type", "okrug": "area_okrug", "local": "local_area"},
            "monotone_nonnegative": self.monotone,
            "calibration_shift": self.shift,
            "penalties": {
                "features": PENALTY_FEATURES, "region_time": PENALTY_REGION_TIME, "local_smooth": PENALTY_LOCAL_SMOOTH,
            },
        }


def predict_from_json(spec: dict, row: dict, month: str) -> float:
    """Log total price for one flat (dict of feature values) in `month`, from the exported JSON.

    The same arithmetic as the app; used to check the export.
    """
    y = spec["intercept"]
    for f in spec["numeric"]:
        v = math.log(row["total_square"]) if f["name"] == spec["log_area_feature"] else row[f["name"]]
        y += f["coef"] * (float(v) - f["mean"]) / f["scale"]
    for c, levels in spec["categorical"].items():
        v = row.get(c)
        v = "missing" if v is None else str(v)
        y += levels.get(v, levels[spec["other_level"]])
    t = spec["months"].index(month)
    for kind, col in spec["time_keys"].items():
        key = row.get(col)
        if kind == "region":
            key = key or "unknown"
        effects = spec["time"][kind].get(key)
        if effects:
            y += effects[t]
    return y
