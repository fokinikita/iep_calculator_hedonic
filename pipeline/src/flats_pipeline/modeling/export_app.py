"""Export everything the web app needs into a small, self-contained bundle (the app repo
iep_calculator_hedonic, folder data/).

The app then starts without the training data and parquet files: it only runs the model for the
entered flat. Everything else it shows is precomputed here.

    data/
      manifest.json                  what was exported, when, from which data; model quality
      model/model.cbm                gradient boosting model (calibration shift already inside)
      model/model_meta.json          feature order (with month dummies), fill values, months, conditions
      model/linear_model.json        linear time-dummy model: coefficients and time effects
      indices/areas.json             per area: hedonic (both models) / median price per m2 and indices by
                                     month, medians of similar flats (by number of rooms)
      checks/reference_estimates.json  curves of a few flats computed here; the app is tested against them
      local/local_price_reference.npz  coordinates + log price per m2 of all sales (30-nearest feature)
      geo/*.geojson, *.json          polygons and metro used by the feature code

Run: python -m flats_pipeline export-app [--out PATH]   (default: settings.APP_BUNDLE_DIR)
"""

import hashlib
import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from flats_pipeline.config import settings
from flats_pipeline.modeling import paths
from flats_pipeline.modeling.build_features import LOCAL_PRICE, features_dir
from flats_pipeline.modeling.hedonic import ROOM_BUCKETS, TIME, hedonic_dir, predict_price_sq, with_time
from flats_pipeline.modeling.linear import TimeDummyModel
from flats_pipeline.modeling.local import K_NEIGHBORS, LocalPrice
from flats_pipeline.modeling.single_flat import REFERENCE_FLATS, flat_features, request_of
from flats_pipeline.modeling.monotone import CONDITION_DEFAULT, CONDITION_FEATURE, CONDITION_LEVELS

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2  # 2: month dummies, linear time-dummy model, reference checks
DEFAULT_OUT = settings.APP_BUNDLE_DIR
GEO_FILES = [
    paths.MOSCOW_POLYGONS_ABSOLUTE_PATH,
    paths.MOSCOW_AREAS_ABSOLUTE_PATH,
    paths.OBLAST_POLYGONS_ABSOLUTE_PATH,
    paths.OBLAST_CITIES_ABSOLUTE_PATH,
    paths.METRO_ABSOLUTE_PATH,
]


def _num(v, digits: int):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), digits)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _areas(hed: Path, periods: list[str]) -> dict:
    """One record per (level, area): series aligned to `periods` (null where not enough sales)."""
    areas = pl.read_parquet(hed / "area_indices.parquet")
    similar = pl.read_parquet(hed / "similar_medians.parquet")
    room_path = hed / "room_indices.parquet"
    room_hed = pl.read_parquet(room_path) if room_path.exists() else None
    grid = pl.DataFrame({"period": periods})
    out = []
    for (level, area), a in areas.group_by("level", "area", maintain_order=True):
        a = grid.join(a, on="period", how="left")
        sim = similar.filter((pl.col("level") == level) & (pl.col("area") == area))
        similar_by_rooms = {}
        for rooms in ROOM_BUCKETS:
            s = grid.join(sim.filter(pl.col("rooms") == rooms), on="period", how="left")
            values = [_num(v, 0) for v in s["similar_median_price_sq"].to_list()]
            if any(v is not None for v in values):
                similar_by_rooms[rooms] = values
        hedonic_by_rooms = {}
        if room_hed is not None:
            rh = room_hed.filter((pl.col("level") == level) & (pl.col("area") == area))
            for rooms in ROOM_BUCKETS:
                s = grid.join(rh.filter(pl.col("rooms") == rooms), on="period", how="left")
                values = [_num(v, 0) for v in s["hedonic_price_sq"].to_list()]
                if all(v is not None for v in values):
                    hedonic_by_rooms[rooms] = values
        out.append({
            "level": level,
            "area": area,
            "hedonic_price_sq": [_num(v, 0) for v in a["hedonic_price_sq"].to_list()],
            "hedonic_index": [_num(v, 2) for v in a["hedonic_index"].to_list()],
            "hedonic_price_sq_linear": [_num(v, 0) for v in a["hedonic_price_sq_linear"].to_list()],
            "hedonic_index_linear": [_num(v, 2) for v in a["hedonic_index_linear"].to_list()],
            "median_price_sq": [_num(v, 0) for v in a["median_price_sq"].to_list()],
            "median_index": [_num(v, 2) for v in a["median_index"].to_list()],
            "n_sales": int(a["n_sales"].fill_null(0).sum()),
            "similar_median_price_sq": similar_by_rooms,
            "hedonic_price_sq_by_rooms": hedonic_by_rooms,
        })
    return {"periods": periods, "room_buckets": ROOM_BUCKETS, "areas": out}


def _reference(meta: dict, local: LocalPrice) -> dict:
    """Price per m2 by month of the reference flats, both models, computed with the pipeline code."""
    from catboost import CatBoostRegressor

    hed = hedonic_dir()
    model = CatBoostRegressor()
    model.load_model(str(hed / "model_index.cbm"))
    linear = TimeDummyModel.from_json(json.loads((hed / "linear_model.json").read_text(encoding="utf-8")), TIME)
    months, cont, cat = meta["months"], meta["continuous_features"], meta["categorical_features"]
    fill = meta["continuous_fill"]
    cases = []
    for flat in REFERENCE_FLATS:
        f = flat_features(flat, local)
        x = f.with_columns(
            [pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c).cast(pl.Float64))
             .otherwise(pl.lit(fill.get(c, 0.0))).alias(c) for c in cont]
            + [pl.col(c).cast(pl.String).fill_null("missing") for c in cat]
        )
        boost, lin = [], []
        for m in months:
            xm = with_time(x, months, m)
            boost.append(float(predict_price_sq(model, xm, meta["feature_order"], cat)[0]))
            lin.append(float(linear.predict_price_sq(xm)[0]))
        cases.append({"request": request_of(flat), "boosting_price_sq": boost, "linear_price_sq": lin})
    return {"months": months, "cases": cases}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_app(out: Path = DEFAULT_OUT, variant: str = "iep") -> Path:
    hed = hedonic_dir()
    meta = json.loads((hed / "hedonic_index_meta.json").read_text(encoding="utf-8"))
    months = meta["months"]
    periods = [m.replace("_", "-") for m in months]

    # Write into a temp folder, then swap: a running app never sees a half-written bundle.
    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "model").mkdir(parents=True)

    shutil.copy2(hed / "model_index.cbm", tmp / "model" / "model.cbm")
    shutil.copy2(hed / "linear_model.json", tmp / "model" / "linear_model.json")
    _write_json(tmp / "model" / "model_meta.json", {
        "continuous_features": meta["continuous_features"],
        "categorical_features": meta["categorical_features"],
        "feature_order": meta["feature_order"],
        "time_dummies": meta["time_dummies"],
        "time_index_feature": meta.get("time_index_feature"),
        "months": months,
        "continuous_fill": meta.get("continuous_fill", {}),
        "target": meta.get("target"),
        "monotone_increasing": meta.get("monotone_increasing", []),
        "condition_feature": CONDITION_FEATURE,
        "condition_levels": CONDITION_LEVELS,
        "condition_default": CONDITION_DEFAULT,
        "local_price_feature": LOCAL_PRICE,
        "local_k_neighbors": K_NEIGHBORS,
    })
    _write_json(tmp / "indices" / "areas.json", _areas(hed, periods))

    ref = pl.read_parquet(features_dir(variant) / "local_price_reference.parquet")
    _write_json(tmp / "checks" / "reference_estimates.json", _reference(meta, LocalPrice.from_frame(ref)))
    (tmp / "local").mkdir()
    np.savez_compressed(
        tmp / "local" / "local_price_reference.npz",
        lat=ref["location.lat"].cast(pl.Float64).to_numpy(),
        lon=ref["location.lon"].cast(pl.Float64).to_numpy(),
        log_price_sq=ref["log_price_sq"].cast(pl.Float64).to_numpy(),
    )

    (tmp / "geo").mkdir()
    for f in GEO_FILES:
        shutil.copy2(f, tmp / "geo" / Path(f).name)

    forward = hed / "forward_test.json"
    files = {p.relative_to(tmp).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
             for p in sorted(tmp.rglob("*")) if p.is_file()}
    _write_json(tmp / "manifest.json", {
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model_built": meta.get("built"),
        "last_sale_date": meta.get("last_sale_date"),
        "periods": periods,
        "n_sales": meta.get("n_sales"),
        "quality": {
            "basket": meta.get("basket_quality"),
            "basket_linear": meta.get("basket_quality_linear"),
            "single_flat_curves": meta.get("single_flat_curves"),
            "forward_test": json.loads(forward.read_text(encoding="utf-8")) if forward.exists() else None,
        },
        "files": files,
    })

    old = out.parent / (out.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    if old.exists():
        shutil.rmtree(old)
    total = sum(f["bytes"] for f in files.values())
    logger.info("App bundle exported: %s (%d files, %.1f MB)", out, len(files), total / 1e6)
    return out
