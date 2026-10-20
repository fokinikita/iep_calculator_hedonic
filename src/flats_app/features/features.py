"""Feature code for one flat: the IEP variant of flats_pipeline FeaturesService (no text features).

Copied from flats_pipeline/modeling/features. Keep in sync with it: the model was trained on
these features (tests/test_app.py checks the estimates against reference values).
"""

import polars as pl

from flats_app.features.additional_base.additional_base_features import AdditionalBaseFeaturesService
from flats_app.features.geo.geo_features import GeoFeaturesService


def _add_neighbors_stats(df: pl.DataFrame) -> pl.DataFrame:
    """As in training. The app has no reference sales here, so the stats are empty (null)."""
    lookup = df.select("guid", "price", "total_square", "total_room_count")
    exploded = df.with_row_index("row_id").explode("neighbors_guid").rename({"neighbors_guid": "neighbor_guid"})
    joined = exploded.join(lookup, left_on="neighbor_guid", right_on="guid", how="left").rename({
        "price": "neighbor_price",
        "total_square": "neighbor_total_square",
        "total_room_count": "neighbor_total_rooms",
    })
    agg = joined.group_by("row_id").agg(
        pl.col("neighbor_price").mean().alias("neighbors_mean_price"),
        pl.col("neighbor_total_square").mean().alias("neighbors_mean_total_square"),
        pl.col("neighbor_total_rooms").median().cast(pl.Int64).alias("neighbors_median_total_rooms"),
    )
    return df.with_row_index("row_id").join(agg, on="row_id", how="left").drop("row_id")


def calculate_features(df: pl.DataFrame) -> pl.DataFrame:
    df = AdditionalBaseFeaturesService(df).calculate_features()
    df = df.drop("note", strict=False)
    df = GeoFeaturesService(df).calculate_features()
    df = _add_neighbors_stats(df).drop("neighbors_distance_m", "neighbors_guid", strict=False)
    return df.with_columns((pl.col("price") / pl.col("total_square")).log().alias("log_price_sq"))
