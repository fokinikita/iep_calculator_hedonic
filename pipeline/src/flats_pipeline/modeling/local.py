"""Local location features.

- Areas: region (Москва / Московская область), Moscow административный округ and район, oblast
  городской округ. `local_area` = район in Moscow, городской округ in the oblast: a categorical
  model feature, finer than the округ the top_flip geo features give.
- Local price level `local_log_price_sq`: median log(price/m2) of the K nearest other sales. Without it
  the model mixed cheap and expensive districts of one округ (biases up to 15-19% by район).
  In training each flat's own sale is excluded; the app queries all sales.
"""

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

REGION_NAMES = {"moscow": "Москва", "oblast": "Московская область"}
K_NEIGHBORS = 30
_KM_PER_DEG = 111.32
_COS_LAT = float(np.cos(np.radians(55.75)))


def add_areas(df: pl.DataFrame) -> pl.DataFrame:
    """area_region / area_okrug / area_raion / area_go and local_area; needs region_type,
    district_name, location.lat/lon (from the geo features)."""
    from flats_pipeline.modeling.features.geo.geo_features import GeoFeaturesService

    svc = GeoFeaturesService(df.select("location.lat", "location.lon"))
    svc._ensure_data_loaded()
    lat, lon = svc._coords(svc.features)
    in_moscow = (df["region_type"] == "moscow").to_numpy()
    raion = svc._first_containing(svc.moscow_polygons, lon, lat, in_moscow)
    return df.with_columns(
        pl.col("region_type").replace_strict(REGION_NAMES, default=None).alias("area_region"),
        pl.when(pl.col("region_type") == "moscow").then(pl.col("district_name")).alias("area_okrug"),
        pl.Series("area_raion", raion.tolist(), dtype=pl.Utf8),
        pl.when(pl.col("region_type") == "oblast").then(pl.col("district_name")).alias("area_go"),
    ).with_columns(
        pl.coalesce("area_raion", "area_go", "district_name").fill_null("missing").alias("local_area")
    )


def _xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Plane coordinates in km (good enough within the Moscow region)."""
    return np.column_stack([lon * _KM_PER_DEG * _COS_LAT, lat * _KM_PER_DEG])


class LocalPrice:
    """Median target of the K nearest reference sales."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, log_price_sq: np.ndarray):
        ok = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(log_price_sq))
        self.y = log_price_sq[ok]
        self.tree = cKDTree(_xy(lat[ok], lon[ok]))

    @classmethod
    def from_frame(cls, df: pl.DataFrame, target: str = "log_price_sq") -> "LocalPrice":
        return cls(
            df["location.lat"].cast(pl.Float64).to_numpy(),
            df["location.lon"].cast(pl.Float64).to_numpy(),
            df[target].cast(pl.Float64).to_numpy(),
        )

    def query(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """For new flats (the app): K nearest reference sales."""
        _, idx = self.tree.query(_xy(lat, lon), k=K_NEIGHBORS, workers=2)
        return np.median(self.y[idx], axis=1)

    def query_reference_itself(self) -> np.ndarray:
        """For the reference sales themselves (training): K nearest *other* sales.

        Each row's own index is removed explicitly (flats in one building share coordinates, so
        the nearest point is not always the flat itself). Needs a reference without missing values.
        """
        n = self.y.shape[0]
        _, idx = self.tree.query(self.tree.data, k=K_NEIGHBORS + 1, workers=2)
        own = idx == np.arange(n)[:, None]
        # drop the own index where found, else the farthest neighbor
        drop = np.where(own.any(axis=1), own.argmax(axis=1), K_NEIGHBORS)
        keep = np.ones_like(idx, dtype=bool)
        keep[np.arange(n), drop] = False
        return np.median(self.y[idx[keep].reshape(n, K_NEIGHBORS)], axis=1)
