"""Features of one flat entered by hand, with the training feature code (IEP variant).

Used by export_app to record reference answers: the app (iep_calculator_hedonic) has its own copy
of the feature code and is tested to give the same estimates for these flats.
"""

import polars as pl

from flats_pipeline.modeling.build_features import LOCAL_PRICE
from flats_pipeline.modeling.features.features import FeaturesService
from flats_pipeline.modeling.local import LocalPrice, add_areas
from flats_pipeline.modeling.monotone import CONDITION_FEATURE, CONDITION_LEVELS

# Flats the app is checked on: center, Марьино (II-18), Мытищи, Балашиха new build, a big
# Хамовники flat, a studio, the far oblast, the Новая Москва border.
REFERENCE_FLATS = [
    {"lat": 55.7558, "lon": 37.6173},
    {"lat": 55.6505, "lon": 37.7440, "building_batch_name": "II-18", "built_year": 1975},
    {"lat": 55.9116, "lon": 37.7307},
    {"lat": 55.7963, "lon": 37.9382, "total_room_count": 1, "total_square": 38, "kitchen_square": 10,
     "life_square": 18, "built_year": 2018, "storeys_count": 25, "storey": 20,
     "apartment_condition": "хорошее состояние"},
    {"lat": 55.7300, "lon": 37.5900, "total_room_count": 4, "total_square": 160, "kitchen_square": 25,
     "life_square": 90, "bathroom_count": 2, "balcony_count": 2, "built_year": 2008,
     "apartment_condition": "дизайнерский", "building_batch_name": "индивидуальный проект"},
    {"lat": 55.5700, "lon": 37.5800, "total_room_count": 0, "total_square": 24, "kitchen_square": 4,
     "life_square": 14, "storey": 1, "storeys_count": 17, "apartment_condition": "требуется ремонт"},
    {"lat": 54.9500, "lon": 38.5000, "built_year": 1965, "storeys_count": 5, "storey": 5, "building_batch_name": "П-44"},
    {"lat": 55.4000, "lon": 36.8000, "elevator_count": 0, "storeys_count": 3, "storey": 2},
]
BASE_FLAT = {
    "total_room_count": 2, "total_square": 54.0, "kitchen_square": 9.0, "life_square": 32.0,
    "balcony_count": 1, "storey": 5, "storeys_count": 12, "bathroom_count": 1, "elevator_count": 1,
    "built_year": 1985, "apartment_condition": "среднее состояние", "building_batch_name": None,
}


def flat_features(flat: dict, local: LocalPrice) -> pl.DataFrame:
    """One-row frame: the flat (form fields + lat/lon) through the training feature code.

    Always a regular flat (not апартаменты) in a direct sale; condition = merged level.
    """
    d = {**BASE_FLAT, **flat, "is_apartment": 0, "sale_type_name": "прямая продажа"}
    df = pl.DataFrame([{
        **{k: v for k, v in d.items() if k not in ("lat", "lon")},
        "location.lat": float(d["lat"]), "location.lon": float(d["lon"]),
    }]).with_columns(
        pl.col("total_square", "kitchen_square", "life_square").cast(pl.Float64),
        pl.col("building_batch_name").cast(pl.String),
        pl.lit("flat").alias("guid"),
        (pl.col("storey") == 1).cast(pl.Int8).alias("storey_first"),
        (pl.col("storey") == pl.col("storeys_count")).cast(pl.Int8).alias("storey_last"),
        (pl.col("storey") / pl.col("storeys_count")).alias("storey_relative"),
        pl.lit(1.0).alias("price"),
    )
    feats = add_areas(FeaturesService(df, include_nlp=False).calculate_features())
    feats = feats.with_columns(pl.lit(float(CONDITION_LEVELS.index(d["apartment_condition"]))).alias(CONDITION_FEATURE))
    lp = local.query(feats["location.lat"].to_numpy(), feats["location.lon"].to_numpy())
    return feats.with_columns(pl.Series(LOCAL_PRICE, lp))


def request_of(flat: dict) -> dict:
    """The /estimate request body for a reference flat (as the page sends it)."""
    d = {**BASE_FLAT, **flat}
    return {**{k: (str(v) if isinstance(v, (int, float)) and k not in ("lat", "lon") else v) for k, v in d.items()},
            "is_apartment": 0, "sale_type_name": "прямая продажа"}
