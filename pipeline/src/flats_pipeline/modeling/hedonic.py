"""Hedonic price indices: gradient boosting (CatBoost) and a linear time-dummy model.

Both models are fitted on 80% of the sales of every month; the other 20% is a fixed basket.
Target: log(total price); price per m2 = total / area. Time enters both as simple month dummies
(t_<month> = 1 in that month; boosting has one for every month, the regression drops the first).

- Boosting (build_index): strongly regularised CatBoost (shallow trees, l2, 80% subsampling of rows
  per tree), so that the curve of a single flat (only the month dummies change) does not jump
  from month to month. Monotone in the areas, balconies, bathrooms and condition.
- Linear (linear.py): log price = features + region x month dummies + smoothed local corrections.
  The time effect is shared by all flats of an area, so a single flat's curve has no own noise.

Index: the basket priced in every month and averaged, relative to the first month (= 100). It is
the price change of flats of the same quality, not the change in what was sold.

train_served_model (train_hedonic_model.py): the earlier single model for an API, kept as a port.

Output in DATA_DIR/modeling/hedonic/.
"""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from flats_pipeline.config import settings
from flats_pipeline.modeling import config_model
from flats_pipeline.modeling.build_features import load_features
from flats_pipeline.modeling.local import REGION_NAMES, add_areas  # noqa: F401  (re-exported for the app)
from flats_pipeline.modeling.monotone import MODEL_TARGET, MONOTONE_INCREASING

logger = logging.getLogger(__name__)

TARGET = config_model.TARGET_NAME
TIME = "year_month"
# All empty without a neighbor reference set (as in top_flip IEP), so it is not used.
DROP_FEATURES = {"neighbors_median_total_rooms"}
MIN_MONTH_OBS = 1000  # months with fewer sales (the edges of the data) are left out
BASKET_SHARE = 0.2
SPLIT_SEED = 42
# Regularised so a single flat's curve over months is stable (see README, "Single-flat curves").
INDEX_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.08,
    "depth": 6,
    "l2_leaf_reg": 50,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,  # every tree sees a random 80% of the training sales
    "loss_function": "MAE",
    "random_seed": 228,
    "verbose": 200,
    # Categorical features with up to 255 values (house series, район, metro, ...) as one-hot columns:
    # with monotone constraints CatBoost did not use them in the default encoding (importance 0).
    "one_hot_max_size": 255,
}
SERVED_PARAMS = {
    "iterations": 1450,
    "learning_rate": 0.142,
    "depth": 8,
    "l2_leaf_reg": 0.005,
    "random_seed": 228,
    "verbose": 100,
    "loss_function": "MAE",
}


def hedonic_dir() -> Path:
    d = settings.modeling_dir / "hedonic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare(
    data: pl.DataFrame, continuous: list[str], categorical: list[str], start_month: str | None = None
) -> tuple[pl.DataFrame, list[str], list[str], dict[str, float]]:
    """Add year_month, keep months with enough sales, fill gaps as the notebook does.

    Continuous: inf/NaN -> null -> column mean (else 0). Categorical: string, null -> "missing".
    Also returns the fill value per continuous feature (to fill a single flat the same way).
    """
    continuous = [c for c in continuous if c not in DROP_FEATURES and c in data.columns]
    categorical = [c for c in categorical if c not in DROP_FEATURES and c in data.columns]
    data = data.with_columns(pl.col("max_start_status3").dt.strftime("%Y_%m").alias(TIME))
    counts = data.group_by(TIME).len()
    months_ok = counts.filter(pl.col("len") >= MIN_MONTH_OBS)[TIME]
    data = data.filter(pl.col(TIME).is_in(months_ok.implode()))
    if start_month:
        data = data.filter(pl.col(TIME) >= start_month)
    data = data.with_columns(
        [pl.col(c).cast(pl.Float64) for c in continuous] + [pl.col("price").log().alias(MODEL_TARGET)]
    ).with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in continuous]
    )
    means = data.select([pl.col(c).mean() for c in continuous]).row(0, named=True)
    fill = {c: float(v) if v is not None else 0.0 for c, v in means.items()}
    data = data.with_columns(
        [pl.col(c).fill_null(fill[c]) for c in continuous]
        + [pl.col(c).cast(pl.String).fill_null("missing") for c in categorical]
    )
    return data, continuous, categorical, fill


def _pool(df: pl.DataFrame, features: list[str], cat: list[str], with_label: bool = True):
    from catboost import Pool

    return Pool(
        df.select(features).to_pandas(),
        label=df[MODEL_TARGET].to_numpy() if with_label else None,
        cat_features=cat,
    )


def new_model(params: dict, features: list[str]):
    """CatBoost on log(total price); larger areas and a better condition can only raise the price."""
    from catboost import CatBoostRegressor

    mono = {f: 1 for f in MONOTONE_INCREASING if f in features}
    return CatBoostRegressor(**params, thread_count=settings.CATBOOST_THREADS, monotone_constraints=mono)


def calibrate(model, train: pl.DataFrame, features: list[str], cat: list[str], sample: int = 150_000) -> float:
    """Shift the model so its median log error on the training sales is 0.

    With monotone constraints on log(total price) the boosting stops ~1.8% low for every segment
    alike (a constant offset). Returns the applied shift (log units).
    """
    s = train.sample(min(sample, train.height), seed=SPLIT_SEED)
    resid = s[MODEL_TARGET].to_numpy() - model.predict(_pool(s, features, cat, with_label=False))
    shift = float(np.median(resid))
    scale, bias = model.get_scale_and_bias()
    model.set_scale_and_bias(scale, bias + shift)
    logger.info("Calibrated: shift %.4f (x%.4f)", shift, np.exp(shift))
    return shift


def predict_price_sq(model, df: pl.DataFrame, features: list[str], cat: list[str]) -> np.ndarray:
    """The model predicts log(total price); price per m2 = total / area."""
    return np.exp(model.predict(_pool(df, features, cat, with_label=False))) / df["total_square"].to_numpy()


def _metrics(df: pl.DataFrame, pred: np.ndarray) -> dict:
    """pred: predicted price per m2."""
    price_sq = np.exp(df[TARGET].to_numpy())
    return {
        "n": int(df.height),
        "wape": float(np.abs(price_sq - pred).sum() / price_sq.sum()),
        "mape": float(np.mean(np.abs(price_sq - pred) / price_sq)),
    }


def time_dummies(months: list[str]) -> list[str]:
    """One 0/1 column per month, the first month included.

    Trees need every month as its own column: with the first month as "all dummies 0" (as in a
    regression) the boosting pools it with other months and the whole index shifts by ~3%.
    """
    return [f"t_{m}" for m in months]


# Month number 0..n-1 next to the dummies. Trees then split "before / after a month" and pool
# neighbouring months instead of judging each month on its own few sales: in the center (ЦАО) the
# zig-zag of a single flat's curve fell 1.69% -> 0.64% a month, flats with a >5% monthly jump 8% -> 0%,
# elsewhere ~1.0% -> ~0.5%; MAPE 11.69% -> 11.62%.
TIME_INDEX = "t_index"


def with_time(df: pl.DataFrame, months: list[str], month: str | None = None) -> pl.DataFrame:
    """Set the month dummies and the month number: from the sale month (`month` None) or as if sold
    in `month`."""
    number = {m: float(i) for i, m in enumerate(months)}
    if month is None:
        return df.with_columns(
            [(pl.col(TIME) == d[2:]).cast(pl.Float64).alias(d) for d in time_dummies(months)]
            + [pl.col(TIME).replace_strict(number, default=None, return_dtype=pl.Float64).alias(TIME_INDEX)]
        )
    return df.with_columns(
        [pl.lit(1.0 if d == f"t_{month}" else 0.0).alias(d) for d in time_dummies(months)]
        + [pl.lit(number[month]).alias(TIME_INDEX), pl.lit(month).alias(TIME)]
    )


def boost_features(continuous: list[str], categorical: list[str], months: list[str]) -> list[str]:
    """Column order of the boosting model."""
    return continuous + time_dummies(months) + [TIME_INDEX] + categorical


def curve_stability(curves_log: np.ndarray) -> dict:
    """How much single flats move over time. curves_log: (flats, months) log price per m2.

    jump: typical month-to-month move of a flat; own_noise: the part not shared with the market
    (the basket average); big_jumps: share of flats with any own move above 5% in a month.
    """
    d = np.diff(curves_log, axis=1)
    market = np.log(np.exp(curves_log).mean(axis=0))
    own = d - np.diff(market)
    return {
        "jump_pct": round(100 * float(np.median(np.abs(d).mean(axis=1))), 3),
        "own_noise_pct": round(100 * float(np.median(own.std(axis=1))), 3),
        "big_jumps_share": round(float((np.abs(own) > 0.05).any(axis=1).mean()), 4),
    }


STABILITY_SAMPLE = 20_000


def split_basket(data: pl.DataFrame, months: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """80/20 inside every month (fixed seed): the 20% is the fixed basket."""
    train_parts, basket_parts = [], []
    for m in months:
        g = data.filter(pl.col(TIME) == m).sample(fraction=1.0, shuffle=True, seed=SPLIT_SEED)
        n_basket = max(1, int(round(BASKET_SHARE * g.height)))
        basket_parts.append(g.head(n_basket))
        train_parts.append(g.tail(g.height - n_basket))
    return pl.concat(train_parts), pl.concat(basket_parts)


def build_index(start_month: str | None = None, variant: str = "iep") -> pl.DataFrame:
    from flats_pipeline.modeling.linear import TimeDummyModel

    raw, continuous, categorical = load_features(variant)
    data, continuous, categorical, fill = prepare(raw, continuous, categorical, start_month)
    del raw
    months = sorted(data[TIME].unique().to_list())
    logger.info("Hedonic index: %d sales, %d months %s..%s", data.height, len(months), months[0], months[-1])
    train, basket = split_basket(data, months)
    train = with_time(train, months)
    basket = with_time(basket, months)

    feats = boost_features(continuous, categorical, months)
    model = new_model(INDEX_PARAMS, feats)
    model.fit(_pool(train, feats, categorical))
    shift = calibrate(model, train, feats, categorical)
    quality = _metrics(basket, predict_price_sq(model, basket, feats, categorical))
    logger.info("Boosting, basket in its own month: WAPE=%.4f MAPE=%.4f", quality["wape"], quality["mape"])

    linear = TimeDummyModel(continuous, categorical, months, TIME).fit(train)
    quality_lin = _metrics(basket, linear.predict_price_sq(basket))
    logger.info("Linear time-dummy, basket in its own month: WAPE=%.4f MAPE=%.4f", quality_lin["wape"], quality_lin["mape"])
    del train

    # Fixed basket priced in every month (only the month changes), both models.
    level, level_lin = {}, {}
    sample = basket.sample(min(STABILITY_SAMPLE, basket.height), seed=1)
    curves, curves_lin = [], []
    for m in months:
        b = with_time(basket, months, m)
        # geometric mean of the basket (see build_area_indices)
        level[m] = float(np.exp(np.log(predict_price_sq(model, b, feats, categorical)).mean()))
        level_lin[m] = float(np.exp(np.log(linear.predict_price_sq(b)).mean()))
        sm = with_time(sample, months, m)
        curves.append(np.log(predict_price_sq(model, sm, feats, categorical)))
        curves_lin.append(np.log(linear.predict_price_sq(sm)))
    stability = {"boosting": curve_stability(np.column_stack(curves)), "linear": curve_stability(np.column_stack(curves_lin))}
    logger.info("Single-flat curves: %s", stability)

    raw_median = data.group_by(TIME).agg(
        pl.len().alias("n_sales"), (pl.col(TARGET).exp()).median().alias("median_price_sq")
    )
    index = (
        pl.DataFrame({
            TIME: months,
            "basket_price_sq": [level[m] for m in months],
            "basket_price_sq_linear": [level_lin[m] for m in months],
        })
        .with_columns(
            pl.col(TIME).str.replace("_", "-").alias("period"),
            (100 * pl.col("basket_price_sq") / level[months[0]]).alias("index"),
            (100 * pl.col("basket_price_sq_linear") / level_lin[months[0]]).alias("index_linear"),
        )
        .with_columns((pl.col("index") / pl.col("index").shift(1) * 100 - 100).alias("mom_pct"))
        .join(raw_median, on=TIME)
        .sort(TIME)
        .with_columns(
            (100 * pl.col("median_price_sq") / pl.col("median_price_sq").first()).alias("median_index")
        )
        .select("period", "index", "index_linear", "mom_pct", "basket_price_sq", "basket_price_sq_linear",
                "n_sales", "median_price_sq", "median_index")
        .sort("period")
    )

    out = hedonic_dir()
    as_of = data["max_start_status3"].max()
    model.save_model(str(out / "model_index.cbm"))
    (out / "linear_model.json").write_text(json.dumps(linear.to_json(), ensure_ascii=False), encoding="utf-8")
    index.write_parquet(out / "hedonic_index.parquet")
    basket.select("guid").write_parquet(out / "basket_guids.parquet")
    _write_excel(index, out / "hedonic_index.xlsx")
    _write_chart(index, out / "chart_index.png")
    meta = {
        "built": date.today().isoformat(),
        "last_sale_date": str(as_of),
        "months": months,
        "n_sales": int(data.height),
        "n_train": int(data.height - basket.height),
        "n_basket": int(basket.height),
        "basket_quality": quality,
        "basket_quality_linear": quality_lin,
        "single_flat_curves": stability,
        "continuous_features": continuous,
        "categorical_features": categorical,
        "time_feature": TIME,
        "time_dummies": time_dummies(months),
        "time_index_feature": TIME_INDEX,
        "feature_order": feats,
        "params": INDEX_PARAMS,
        "continuous_fill": fill,
        "target": MODEL_TARGET,
        "calibration_shift": shift,
        "monotone_increasing": [f for f in MONOTONE_INCREASING if f in feats],
    }
    (out / "hedonic_index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Hedonic index written: %s", out / "hedonic_index.xlsx")
    return index


AREA_LEVELS = {
    "region": "Регион",
    "okrug": "Административный округ Москвы",
    "raion": "Район Москвы",
    "go": "Городской округ Московской области",
}
MIN_AREA_BASKET = 100  # fewer basket flats in an area: no area hedonic index
# Median price per m2 of an area: sales of the last MEDIAN_WINDOW_MONTHS months (one month alone jumps:
# in the centre ±5% a month for ЦАО, up to ±23% for a район, from which flats happened to sell).
MEDIAN_WINDOW_MONTHS = 3
MIN_AREA_MONTH_SALES = 60  # fewer sales in the window: no median for that month


ROOM_BUCKETS = ["студия", "1", "2", "3", "4+"]
MIN_SIMILAR_SALES = 30


def room_bucket(rooms: pl.Expr) -> pl.Expr:
    r = rooms.cast(pl.Int64)
    return (
        pl.when(r <= 0).then(pl.lit("студия"))
        .when(r >= 4).then(pl.lit("4+"))
        .otherwise(r.cast(pl.String))
    )


def build_similar_medians(data: pl.DataFrame, months: list[str]) -> pl.DataFrame:
    """Median price per m2 of *similar* flats: same area and number of rooms, 3-month rolling window.

    The all-flats median of an area mixes studios (high price per m2) with big flats (low), so a
    given flat should be compared with flats of its own size. A 3-month window keeps enough sales.
    """
    frames = []
    for level in AREA_LEVELS:
        col = f"area_{level}"
        g = data.filter(pl.col(col).is_not_null()).select(
            pl.col(col).alias("area"), room_bucket(pl.col("total_room_count")).alias("rooms"), TIME,
            pl.col(TARGET).exp().alias("psq"),
        )
        for i, m in enumerate(months):
            window = months[max(0, i - 2): i + 1]
            frames.append(
                g.filter(pl.col(TIME).is_in(window))
                .group_by("area", "rooms")
                .agg(pl.col("psq").median().alias("similar_median_price_sq"), pl.len().alias("n_similar"))
                .with_columns(pl.lit(level).alias("level"), pl.lit(m.replace("_", "-")).alias("period"))
            )
    out = pl.concat(frames).with_columns(
        pl.when(pl.col("n_similar") >= MIN_SIMILAR_SALES).then(pl.col("similar_median_price_sq")).alias("similar_median_price_sq")
    )
    out.write_parquet(hedonic_dir() / "similar_medians.parquet")
    logger.info("Similar-flat medians written: %s", hedonic_dir() / "similar_medians.parquet")
    return out


def window_medians(data: pl.DataFrame, by: list[str], months: list[str]) -> pl.DataFrame:
    """Per group and month: median price per m2 of the sales in that month and the
    MEDIAN_WINDOW_MONTHS - 1 before it (null below MIN_AREA_MONTH_SALES), and the month's own sales."""
    own = data.group_by(*by, TIME).agg(pl.len().alias("n_sales"))
    frames = []
    for i, m in enumerate(months):
        window = months[max(0, i - MEDIAN_WINDOW_MONTHS + 1): i + 1]
        g = data.filter(pl.col(TIME).is_in(window))
        agg = [pl.col(TARGET).exp().median().alias("median_price_sq"), pl.len().alias("n_window")]
        frames.append((g.group_by(*by).agg(agg) if by else g.select(agg)).with_columns(pl.lit(m).alias(TIME)))
    med = pl.concat(frames).with_columns(
        pl.when(pl.col("n_window") >= MIN_AREA_MONTH_SALES).then(pl.col("median_price_sq")).alias("median_price_sq")
    ).drop("n_window")
    return med.join(own, on=[*by, TIME], how="left").with_columns(pl.col("n_sales").fill_null(0))


def build_area_indices(variant: str = "iep") -> pl.DataFrame:
    """Hedonic (both models) and median price per m2 by month for every region / округ / район /
    городской округ.

    Hedonic: the fixed basket, restricted to the area, priced in every month.
    Median: median price per m2 of all sales in the area in that month and the 2 before it
    (not quality-adjusted; one month alone is too noisy in small or mixed areas).
    """
    from catboost import CatBoostRegressor

    from flats_pipeline.modeling.linear import TimeDummyModel

    out = hedonic_dir()
    meta = json.loads((out / "hedonic_index_meta.json").read_text(encoding="utf-8"))
    model = CatBoostRegressor()
    model.load_model(str(out / "model_index.cbm"))
    linear = TimeDummyModel.from_json(json.loads((out / "linear_model.json").read_text(encoding="utf-8")), TIME)
    categorical, feats, months = meta["categorical_features"], meta["feature_order"], meta["months"]

    raw, cont_all, cat_all_meta = load_features(variant)
    data, _, _, _ = prepare(raw, cont_all, cat_all_meta)
    del raw
    data = data.filter(pl.col(TIME).is_in(months))
    if "area_region" not in data.columns:
        data = add_areas(data)
    build_similar_medians(data, months)
    basket = data.join(pl.read_parquet(out / "basket_guids.parquet"), on="guid", how="semi")

    # Predict the whole basket in every month once (both models), then average per area.
    preds = []
    for m in months:
        b = with_time(basket, months, m)
        preds.append(basket.select(
            "area_region", "area_okrug", "area_raion", "area_go",
            room_bucket(pl.col("total_room_count").cast(pl.Int64, strict=False)).alias("rooms"),
        ).with_columns(
            pl.lit(m).alias(TIME),
            pl.Series("pred", predict_price_sq(model, b, feats, categorical)),
            pl.Series("pred_lin", linear.predict_price_sq(b)),
        ))
    preds = pl.concat(preds)
    # Geometric mean (= mean of log prices): every basket flat weighs the same, as the model is in logs.
    # The arithmetic mean was price-weighted: expensive flats (slower growth) set the index, so a typical
    # flat grew faster than its район / округ / city (Moscow 2025-09..2026-09: 13.8% vs 15-16%).
    hedonic_aggs = [
        pl.col("pred").log().mean().exp().alias("hedonic_price_sq"),
        pl.col("pred_lin").log().mean().exp().alias("hedonic_price_sq_linear"),
        pl.len().alias("n_basket"),
    ]

    frames = []
    for level in AREA_LEVELS:
        col = f"area_{level}"
        hed = (
            preds.filter(pl.col(col).is_not_null())
            .group_by(col, TIME)
            .agg(hedonic_aggs)
            .filter(pl.col("n_basket") >= MIN_AREA_BASKET)
        )
        med = window_medians(data.filter(pl.col(col).is_not_null()), [col], months)
        frames.append(
            hed.join(med, on=[col, TIME], how="full", coalesce=True)
            .rename({col: "area"})
            .with_columns(pl.lit(level).alias("level"))
        )
    # Whole sample (Москва + область + the rest) as a fallback.
    frames.append(
        preds.group_by(TIME).agg(hedonic_aggs)
        .join(window_medians(data, [], months), on=TIME)
        .with_columns(pl.lit("Москва и область").alias("area"), pl.lit("all").alias("level"))
    )
    first = months[0]

    def rebased(col: str) -> pl.Expr:
        return (100 * pl.col(col) / pl.col(col).filter(pl.col(TIME) == first).first()).over("level", "area")

    areas = (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(pl.col(TIME).str.replace("_", "-").alias("period"))
        .sort("level", "area", TIME)
        .with_columns(
            rebased("hedonic_price_sq").alias("hedonic_index"),
            rebased("hedonic_price_sq_linear").alias("hedonic_index_linear"),
            rebased("median_price_sq").alias("median_index"),
        )
        .select("level", "area", "period", "hedonic_price_sq", "hedonic_index", "hedonic_price_sq_linear",
                "hedonic_index_linear", "n_basket", "median_price_sq", "median_index", "n_sales")
    )
    areas.write_parquet(out / "area_indices.parquet")
    build_room_indices(preds, first)
    n = areas.select("level", "area").unique().group_by("level").len()
    logger.info("Area indices written: %s %s", out / "area_indices.parquet", dict(n.iter_rows()))
    return areas


def build_room_indices(preds: pl.DataFrame, first: str) -> pl.DataFrame:
    """Hedonic price per m2 of the basket flats with the same number of rooms (bucket), per area and
    month: the same basket predictions as the area index, split by rooms."""
    frames = []
    for level in AREA_LEVELS:
        col = f"area_{level}"
        frames.append(
            preds.filter(pl.col(col).is_not_null() & pl.col("rooms").is_not_null())
            .group_by(col, "rooms", TIME)
            .agg(pl.col("pred").log().mean().exp().alias("hedonic_price_sq"), pl.len().alias("n_basket"))
            .filter(pl.col("n_basket") >= MIN_AREA_BASKET)
            .rename({col: "area"})
            .with_columns(pl.lit(level).alias("level"))
        )
    frames.append(
        preds.filter(pl.col("rooms").is_not_null()).group_by("rooms", TIME)
        .agg(pl.col("pred").log().mean().exp().alias("hedonic_price_sq"), pl.len().alias("n_basket"))
        .with_columns(pl.lit("Москва и область").alias("area"), pl.lit("all").alias("level"))
    )
    rooms = (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(pl.col(TIME).str.replace("_", "-").alias("period"))
        .select("level", "area", "rooms", "period", "hedonic_price_sq", "n_basket")
        .sort("level", "area", "rooms", "period")
    )
    path = hedonic_dir() / "room_indices.parquet"
    rooms.write_parquet(path)
    logger.info("Room hedonic indices written: %s (%d rows)", path, rooms.height)
    return rooms


def train_served_model(start_month: str | None = None, variant: str = "iep") -> dict:
    """One model for the API: train+valid, checked on test (train_hedonic_model.py)."""
    from catboost import CatBoostRegressor

    raw, continuous, categorical = load_features(variant)
    data, continuous, categorical, fill = prepare(raw, continuous, categorical, start_month)
    del raw
    cat_all = categorical + [TIME]
    feats = continuous + cat_all
    train = data.filter(pl.col("split") != "test")
    test = data.filter(pl.col("split") == "test")
    model = new_model(SERVED_PARAMS, feats)
    model.fit(_pool(train, feats, cat_all), eval_set=_pool(test, feats, cat_all))
    calibrate(model, train, feats, cat_all)
    quality = _metrics(test, predict_price_sq(model, test, feats, cat_all))
    logger.info("Served model test: WAPE=%.4f MAPE=%.4f n=%d", quality["wape"], quality["mape"], quality["n"])

    out = hedonic_dir()
    model.save_model(str(out / "model_hedonic.cbm"))
    months = sorted(train[TIME].unique().to_list())
    meta = {  # same keys as iep_calculator data/features_iep/features_meta_hedonic.json
        "continuous_features": continuous,
        "categorical_features": categorical,
        "time_feature": TIME,
        "latest_month": months[-1],
        "hedonic_months": months,
        "test_quality": quality,
        "continuous_fill": fill,
    }
    (out / "features_meta_hedonic.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Served model written: %s", out / "model_hedonic.cbm")
    return quality


def _write_excel(index: pl.DataFrame, path: Path) -> None:
    """Excel with a native (editable) line chart, as in hedonic_indices.ipynb."""
    import xlsxwriter

    wb = xlsxwriter.Workbook(str(path))
    bold = wb.add_format({"bold": True})
    ws = wb.add_worksheet("Индекс")
    ws.write_row(0, 0, ["Период", "Индекс (бустинг)", "Индекс (лин. регрессия)", "Δ м/м бустинг, %", "Продаж",
                        "Медиана ₽/м²", "Индекс медианы"], bold)
    for i, r in enumerate(index.iter_rows(named=True), start=1):
        ws.write(i, 0, r["period"])
        ws.write(i, 1, round(r["index"], 2))
        ws.write(i, 2, round(r["index_linear"], 2))
        ws.write(i, 3, "" if r["mom_pct"] is None else round(r["mom_pct"], 2))
        ws.write(i, 4, r["n_sales"])
        ws.write(i, 5, round(r["median_price_sq"]))
        ws.write(i, 6, round(r["median_index"], 2))
    ws.set_column(0, 6, 18)
    n = index.height
    chart = wb.add_chart({"type": "line"})
    for col, color, dash in ((1, "#059669", None), (2, "#2563eb", None), (6, "#94a3b8", "dash")):
        series = {
            "name": ["Индекс", 0, col], "categories": ["Индекс", 1, 0, n, 0], "values": ["Индекс", 1, col, n, col],
            "line": {"width": 2.5 if dash is None else 1.5, "color": color},
        }
        if dash:
            series["line"]["dash_type"] = dash
        else:
            series["marker"] = {"type": "circle", "size": 5}
        chart.add_series(series)
    chart.set_title({"name": "Гедонические индексы цен (база = 100)"})
    chart.set_x_axis({"name": "Месяц"})
    chart.set_y_axis({"name": "Индекс"})
    chart.set_size({"width": 860, "height": 440})
    ws.insert_chart("I2", chart)
    wb.close()


def _write_chart(index: pl.DataFrame, path: Path) -> None:
    """Two panels: boosting and the linear time-dummy model, each with the raw median."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = index["period"].to_list()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, col, color, title in (
        (axes[0], "index", "#059669", "Градиентный бустинг (временные дамми)"),
        (axes[1], "index_linear", "#2563eb", "Линейная регрессия с временными дамми"),
    ):
        ax.plot(periods, index[col].to_list(), marker="o", color=color, lw=2.6, label="Гедонический индекс")
        ax.plot(periods, index["median_index"].to_list(), color="#94a3b8", lw=1.5, ls="--",
                label="Медиана ₽/м² (без поправки на качество)")
        ax.axhline(100, color="#64748b", ls=":", lw=1)
        ax.set_title(f"{title}\nбаза: {periods[0]} = 100", fontsize=10)
        ax.grid(True, color="#e2e8f0")
        ax.legend(fontsize=8, loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    axes[0].set_ylabel("Индекс")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_forward(test_months: int = 3, variant: str = "iep") -> dict:
    """Out-of-time test: train on all sales before the last `test_months` months, estimate those
    months as the app would (as if sold in the last training month, as for a new flat today).

    Shows how well estimates hold for sales the models have not seen, in months they have not seen.
    """
    from flats_pipeline.modeling.linear import TimeDummyModel

    raw, continuous, categorical = load_features(variant)
    data, continuous, categorical, _ = prepare(raw, continuous, categorical)
    del raw
    months = sorted(data[TIME].unique().to_list())
    cutoff = months[-test_months]
    train_months = [m for m in months if m < cutoff]
    anchor = train_months[-1]
    train = with_time(data.filter(pl.col(TIME) < cutoff), train_months)
    test = data.filter(pl.col(TIME) >= cutoff)
    feats = boost_features(continuous, categorical, train_months)
    model = new_model({**INDEX_PARAMS, "verbose": 0}, feats)
    model.fit(_pool(train, feats, categorical))
    calibrate(model, train, feats, categorical)
    linear = TimeDummyModel(continuous, categorical, train_months, TIME).fit(train)
    del train
    at_anchor = with_time(test, train_months, anchor)
    actual = np.exp(test[TARGET].to_numpy())
    res = {"train_months": f"{train_months[0]}..{anchor}", "test_months": f"{cutoff}..{months[-1]}"}
    for name, pred in (
        ("boosting", predict_price_sq(model, at_anchor, feats, categorical)),
        ("linear", linear.predict_price_sq(at_anchor)),
    ):
        by_month = {}
        for m in months[-test_months:]:
            mask = (test[TIME] == m).to_numpy()
            by_month[m] = _metrics(test.filter(pl.col(TIME) == m), pred[mask])
        res[name] = {**_metrics(test, pred), "median_actual/pred": float(np.median(actual / pred)), "by_month": by_month}
        logger.info("Forward test %s, %s: MAPE=%.4f WAPE=%.4f median actual/pred=%.3f", res["test_months"], name,
                    res[name]["mape"], res[name]["wape"], res[name]["median_actual/pred"])
    (hedonic_dir() / "forward_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res
