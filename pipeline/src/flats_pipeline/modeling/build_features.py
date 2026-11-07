"""Features from the cleaned dataset (port of top_flip entrypoints/run_calculate_features[_iep].py).

variant "iep" (default): no description-text (NLP) features, as in top_flip `features_iep`.
variant "full": with NLP features, as in top_flip `features`.

Output in DATA_DIR/modeling/features_<variant>/: train/valid/test_features.parquet (split by the
first listing date `min_start`, dates from config_model) and features_meta.json.
The feature code has no cross-row features here (no neighbor reference set, as in top_flip), so
it is computed once on all rows and then split; the result is the same as per split.
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from flats_pipeline.config import settings
from flats_pipeline.modeling import config_model
from flats_pipeline.modeling.dataset import latest_dataset_path
from flats_pipeline.modeling.features.features import FeaturesService
from flats_pipeline.modeling.local import LocalPrice, add_areas
from flats_pipeline.modeling.monotone import add_condition_level, model_feature_lists

TARGET = config_model.TARGET_NAME
LOCAL_PRICE = "local_log_price_sq"

logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


def features_dir(variant: str = "iep") -> Path:
    return settings.modeling_dir / ("features_iep" if variant == "iep" else "features")


def build_features(dataset: Path | None = None, variant: str = "iep") -> Path:
    if variant not in ("iep", "full"):
        raise ValueError("variant must be 'iep' or 'full'")
    include_nlp = variant == "full"
    dataset = dataset or latest_dataset_path()
    lf = pl.scan_parquet(dataset)
    if not include_nlp:
        lf = lf.drop("note")  # texts are ~2 GB and not needed without NLP features
    data = lf.collect()
    logger.info("Calculating %s features for %d flats from %s", variant, data.height, dataset)

    service = FeaturesService(data, include_nlp=include_nlp)
    features = service.calculate_features()
    continuous, categorical = service.get_feature_types()

    # Local location features (see local.py): район / городской округ, and the price level of the
    # nearest other sales. The reference sales are saved for the app.
    features = add_areas(features)
    valid = features["location.lat"].is_not_null() & features["location.lon"].is_not_null() & features[TARGET].is_finite()
    ref = features.filter(valid).select("location.lat", "location.lon", TARGET)
    local = LocalPrice.from_frame(ref, TARGET).query_reference_itself()
    values = np.full(features.height, np.nan)
    values[valid.to_numpy()] = local
    features = features.with_columns(pl.Series(LOCAL_PRICE, values).fill_nan(None))
    continuous = continuous + [LOCAL_PRICE]
    categorical = categorical + ["local_area"]

    # Monotone model (monotone.py): merged ordered condition, raw areas instead of area ratios.
    features = add_condition_level(features)
    continuous, categorical = model_feature_lists(continuous, categorical)
    categorical_present = [c for c in categorical if c in features.columns]
    features = features.with_columns(pl.col(categorical_present).fill_null("missing"))

    start = pl.col("min_start").cast(pl.Date)
    split = {
        "train": start < config_model.VALID_START_DATE,
        "valid": (start >= config_model.VALID_START_DATE) & (start < config_model.TEST_START_DATE),
        "test": start >= config_model.TEST_START_DATE,
    }
    out_dir = features_dir(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = features.filter(split[name])
        part.write_parquet(out_dir / f"{name}_features.parquet", compression="zstd")
        logger.info("  %s: %d rows", name, part.height)
    meta = {
        "continuous_features": continuous,
        "categorical_features": categorical,
        "dataset": str(dataset),
        "valid_start_date": config_model.VALID_START_DATE.isoformat(),
        "test_start_date": config_model.TEST_START_DATE.isoformat(),
    }
    (out_dir / "features_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ref.write_parquet(out_dir / "local_price_reference.parquet")
    logger.info("Features written: %s (%d continuous, %d categorical)", out_dir, len(continuous), len(categorical))
    return out_dir


def load_features(variant: str = "iep") -> tuple[pl.DataFrame, list[str], list[str]]:
    """All splits concatenated, with a `split` column, plus continuous / categorical feature names."""
    d = features_dir(variant)
    meta = json.loads((d / "features_meta.json").read_text(encoding="utf-8"))
    frames = [pl.read_parquet(d / f"{s}_features.parquet").with_columns(pl.lit(s).alias("split")) for s in SPLITS]
    return pl.concat(frames, how="diagonal_relaxed"), meta["continuous_features"], meta["categorical_features"]
