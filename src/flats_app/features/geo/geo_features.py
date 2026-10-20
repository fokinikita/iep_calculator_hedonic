import json
import math
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, TypedDict

import attr
import numpy as np
import polars as pl
import shapely
from shapely.geometry import Point, shape
from sklearn.neighbors import BallTree

from flats_app import paths


class Geometry(TypedDict):
    type: str
    coordinates: Any


class FeatureProperties(TypedDict, total=False):
    name: str
    name_ru: str
    name_en: str


class Feature(TypedDict):
    type: str
    geometry: Geometry
    properties: Dict[str, Any]


class FeatureCollection(TypedDict):
    type: str
    features: List[Feature]


CENTER_LAT: Final[float] = 55.751028
CENTER_LON: Final[float] = 37.617611
EARTH_RADIUS_KM: Final[float] = 6371.0


@attr.attrs(slots=True, auto_attribs=True)
class GeoFeaturesService:
    features: pl.DataFrame
    reference_df: Optional[pl.DataFrame] = None

    moscow_polygons: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )
    moscow_areas: Optional[List[Dict[str, Any]]] = attr.attrib(init=False, default=None)
    oblast_polygons: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )
    oblast_cities: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )
    metro_stations: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )

    geo_features_names: List[str] = attr.attrib(init=False, factory=list)

    def _get_cols_features(self) -> List[str]:
        return self.features.columns

    @staticmethod
    def load_polygons(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data: FeatureCollection = json.load(f)

        polygons: List[Dict[str, Any]] = []

        for feat in data["features"]:
            props = feat["properties"]

            name: str = (
                props.get("name")
                or props.get("name:ru")
                or props.get("name:en")
                or "unknown"
            )

            polygons.append(
                {
                    "name": name,
                    "geometry": shape(feat["geometry"]),
                }
            )

        return polygons

    @staticmethod
    def load_points(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points: List[Dict[str, Any]] = []
        features = data["features"] if isinstance(data, dict) else data

        for feat in features:
            props = feat.get("properties", {})

            name: str = (
                props.get("name") or feat.get("name") or feat.get("city") or "unknown"
            )

            if "geometry" in feat:
                geom = shape(feat["geometry"])
                if isinstance(geom, Point):
                    points.append(
                        {
                            "name": name,
                            "geometry": geom,
                        }
                    )
            else:
                lat = feat.get("lat") or feat.get("latitude")
                lon = feat.get("lon") or feat.get("lng") or feat.get("longitude")

                if lat is not None and lon is not None:
                    points.append(
                        {
                            "name": name,
                            "geometry": Point(lon, lat),
                        }
                    )

        return points

    @staticmethod
    def load_metro(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        res: List[Dict[str, Any]] = []

        for feat in data["features"]:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]

            name = props.get("name")
            if not name:
                continue

            res.append(
                {
                    "name": name,
                    "lat": coords[1],
                    "lon": coords[0],
                }
            )

        return res

    def _ensure_data_loaded(self) -> None:
        if self.moscow_polygons is None:
            self.moscow_polygons = self.load_polygons(
                paths.MOSCOW_POLYGONS_ABSOLUTE_PATH
            )

        if self.moscow_areas is None:
            self.moscow_areas = self.load_polygons(paths.MOSCOW_AREAS_ABSOLUTE_PATH)

        if self.oblast_polygons is None:
            self.oblast_polygons = self.load_polygons(
                paths.OBLAST_POLYGONS_ABSOLUTE_PATH
            )

        if self.oblast_cities is None:
            self.oblast_cities = self.load_points(paths.OBLAST_CITIES_ABSOLUTE_PATH)

        if self.metro_stations is None:
            self.metro_stations = self.load_metro(paths.METRO_ABSOLUTE_PATH)

    @staticmethod
    def classify(
        row: Dict[str, float],
        moscow_polygons: List[Dict[str, Any]],
        moscow_areas: List[Dict[str, Any]],
        oblast_polygons: List[Dict[str, Any]],
        oblast_cities: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        lat = row.get("location.lat")
        lon = row.get("location.lon")

        if lat is None or lon is None:
            return ("unknown", None, None)

        point = Point(lon, lat)

        for poly in moscow_polygons:
            if poly["geometry"].contains(point):
                for area in moscow_areas:
                    if area["geometry"].contains(point):
                        return ("moscow", area["name"], "Москва")
                return ("moscow", None, "Москва")

        for poly in oblast_polygons:
            if poly["geometry"].contains(point):
                closest_city = None
                min_distance = float("inf")

                for city in oblast_cities:
                    dist = point.distance(city["geometry"])
                    if dist < min_distance:
                        min_distance = dist
                        closest_city = city["name"]

                return ("oblast", poly["name"], closest_city)

        return ("unknown", None, None)

    @staticmethod
    def _coords(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lat = df["location.lat"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        lon = df["location.lon"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        return lat, lon

    @staticmethod
    def _first_containing(polygons: List[Dict[str, Any]], lon: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Name of the first polygon (in file order) containing each point where mask is True."""
        names = np.full(lat.shape[0], None, dtype=object)
        todo = mask.copy()
        for poly in polygons:
            idx = np.flatnonzero(todo)
            if idx.size == 0:
                break
            geom = poly["geometry"]
            shapely.prepare(geom)
            hit = shapely.contains_xy(geom, lon[idx], lat[idx])
            names[idx[hit]] = poly["name"]
            todo[idx[hit]] = False
        return names

    def add_region_feature(self) -> None:
        """Vectorized `classify`: moscow (+ area) / oblast (+ municipality, nearest city) / unknown."""
        self._ensure_data_loaded()
        lat, lon = self._coords(self.features)
        valid = ~(np.isnan(lat) | np.isnan(lon))
        n = lat.shape[0]

        in_moscow = self._first_containing(self.moscow_polygons, lon, lat, valid) != None  # noqa: E711
        moscow_area = self._first_containing(self.moscow_areas, lon, lat, in_moscow)
        oblast_poly = self._first_containing(self.oblast_polygons, lon, lat, valid & ~in_moscow)
        in_oblast = oblast_poly != None  # noqa: E711

        # Nearest oblast city by planar distance in degrees (as shapely point.distance in `classify`).
        city = np.full(n, None, dtype=object)
        idx = np.flatnonzero(in_oblast)
        if idx.size and self.oblast_cities:
            cx = np.array([c["geometry"].x for c in self.oblast_cities])
            cy = np.array([c["geometry"].y for c in self.oblast_cities])
            cnames = np.array([c["name"] for c in self.oblast_cities], dtype=object)
            for start in range(0, idx.size, 50_000):
                part = idx[start : start + 50_000]
                d2 = (lon[part, None] - cx) ** 2 + (lat[part, None] - cy) ** 2
                city[part] = cnames[np.argmin(d2, axis=1)]

        region_type = np.where(in_moscow, "moscow", np.where(in_oblast, "oblast", "unknown"))
        district = np.where(in_moscow, moscow_area, np.where(in_oblast, oblast_poly, None))
        city_name = np.where(in_moscow, "Москва", city)

        self.features = self.features.with_columns(
            pl.Series("region_type", region_type.tolist(), dtype=pl.Utf8),
            pl.Series("district_name", district.tolist(), dtype=pl.Utf8),
            pl.Series("city_name", city_name.tolist(), dtype=pl.Utf8),
        ).with_columns(
            pl.when(pl.col("region_type") == "moscow")
            .then(pl.col("district_name"))
            .when(pl.col("region_type") == "oblast")
            .then(pl.col("city_name"))
            .otherwise(None)
            .alias("region_name")
        )

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )

        return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _haversine_np(lat1, lon1, lat2, lon2):
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def add_center_distance_feature(self) -> None:
        """Distance to the Moscow center. Note: in km, despite the name (kept from top_flip)."""
        lat, lon = self._coords(self.features)
        dist = self._haversine_np(lat, lon, CENTER_LAT, CENTER_LON)
        self.features = self.features.with_columns(
            pl.Series("distance_m", dist).fill_nan(None)
        ).with_columns(
            [
                pl.col("distance_m").sqrt().alias("distance_sqrt"),
                (pl.col("distance_m") ** 2).alias("distance_sq"),
            ]
        )

    @staticmethod
    def nearest_metro(
        row: Dict[str, float],
        metro: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lat = row.get("location.lat")
        lon = row.get("location.lon")

        if lat is None or lon is None:
            return {
                "metro_name": None,
                "metro_distance_m": None,
            }

        min_dist = float("inf")
        closest = None

        for st in metro:
            d = GeoFeaturesService.haversine(lat, lon, st["lat"], st["lon"])
            if d < min_dist:
                min_dist = d
                closest = st["name"]

        return {
            "metro_name": closest,
            "metro_distance_m": min_dist,
        }

    def add_metro_feature(self) -> None:
        """Nearest metro station (name, distance in km) via a haversine BallTree."""
        self._ensure_data_loaded()
        lat, lon = self._coords(self.features)
        valid = ~(np.isnan(lat) | np.isnan(lon))
        names = np.full(lat.shape[0], None, dtype=object)
        dist = np.full(lat.shape[0], np.nan)
        if valid.any() and self.metro_stations:
            st = np.radians([[m["lat"], m["lon"]] for m in self.metro_stations])
            st_names = np.array([m["name"] for m in self.metro_stations], dtype=object)
            tree = BallTree(st, metric="haversine")
            d, i = tree.query(np.radians(np.column_stack([lat[valid], lon[valid]])), k=1)
            names[valid] = st_names[i[:, 0]]
            dist[valid] = d[:, 0] * EARTH_RADIUS_KM
        self.features = self.features.with_columns(
            pl.Series("metro_name", names.tolist(), dtype=pl.Utf8),
            pl.Series("metro_distance_m", dist).fill_nan(None),
        ).with_columns(
            [
                pl.col("metro_distance_m").sqrt().alias("metro_distance_sqrt"),
                (pl.col("metro_distance_m") ** 2).alias("metro_distance_sq"),
            ]
        )

    def _add_empty_neighbors(self) -> None:
        rows = self.features.height
        self.features = self.features.with_columns(
            [
                pl.Series(
                    "neighbors_guid", [[] for _ in range(rows)], dtype=pl.List(pl.Utf8)
                ),
                pl.Series(
                    "neighbors_distance_m",
                    [[] for _ in range(rows)],
                    dtype=pl.List(pl.Float64),
                ),
            ]
        )

    def add_nearest_neighbors(self, k: int = 5) -> None:
        if self.features.height == 0:
            self._add_empty_neighbors()
            return

        # Если reference_df не передан — не падаем
        if self.reference_df is None or self.reference_df.height == 0:
            self._add_empty_neighbors()
            return

        required_cols = {"guid", "location.lat", "location.lon"}
        if not required_cols.issubset(set(self.reference_df.columns)):
            self._add_empty_neighbors()
            return

        query_coords = self.features.select(["location.lat", "location.lon"]).to_numpy()
        ref_coords = self.reference_df.select(
            ["location.lat", "location.lon"]
        ).to_numpy()
        ref_guids = self.reference_df["guid"].to_list()

        if len(ref_coords) == 0:
            self._add_empty_neighbors()
            return

        query_radians = np.radians(query_coords)
        ref_radians = np.radians(ref_coords)

        k_eff = min(k, len(ref_coords))
        if k_eff <= 0:
            self._add_empty_neighbors()
            return

        tree = BallTree(ref_radians, metric="haversine")
        distances, indices = tree.query(query_radians, k=k_eff)

        distances_km = distances * EARTH_RADIUS_KM
        neighbors_guid = [[ref_guids[i] for i in row] for row in indices]

        self.features = self.features.with_columns(
            [
                pl.Series("neighbors_guid", neighbors_guid),
                pl.Series("neighbors_distance_m", distances_km.tolist()),
            ]
        )

    def calculate_features(self) -> pl.DataFrame:
        self._ensure_data_loaded()

        feature_pipeline: List[Callable[[], None]] = [
            self.add_region_feature,
            self.add_center_distance_feature,
            self.add_metro_feature,
            self.add_nearest_neighbors,
        ]

        features_before = self._get_cols_features()

        for feature in feature_pipeline:
            feature()

        features_after = self._get_cols_features()
        self.geo_features_names = list(set(features_after) - set(features_before))

        return self.features
