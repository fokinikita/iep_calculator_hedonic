"""Local location features (copied from flats_pipeline/modeling/local.py, query part only).

- Areas: region (Москва / Московская область), Moscow округ and район, oblast городской округ;
  `local_area` = район in Moscow, городской округ in the oblast.
- Local price level: median log(price/m2) of the K nearest sales (the bundle's reference sales).
"""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from flats_app.features.geo.geo_features import GeoFeaturesService

REGION_NAMES = {"moscow": "Москва", "oblast": "Московская область"}
_KM_PER_DEG = 111.32
_COS_LAT = float(np.cos(np.radians(55.75)))


def add_areas(df: pl.DataFrame) -> pl.DataFrame:
    """area_region / area_okrug / area_raion / area_go and local_area; needs region_type,
    district_name, location.lat/lon (from the geo features)."""
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
    """Median log price per m2 of the K nearest reference sales."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, log_price_sq: np.ndarray, k: int, workers: int = 2):
        ok = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(log_price_sq))
        self.y = log_price_sq[ok]
        self.tree = cKDTree(_xy(lat[ok], lon[ok]))
        self.k = k
        self.workers = workers

    @classmethod
    def from_npz(cls, path: Path, k: int, workers: int = 2) -> "LocalPrice":
        with np.load(path) as z:
            return cls(z["lat"], z["lon"], z["log_price_sq"], k, workers)

    def query(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(_xy(lat, lon), k=self.k, workers=self.workers)
        return np.median(self.y[idx], axis=1)
