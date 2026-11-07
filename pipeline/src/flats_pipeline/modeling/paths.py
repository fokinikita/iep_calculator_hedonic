"""Geo reference files, copied from top_flip/data/geo into flats_pipeline/data/geo."""

from pathlib import Path

GEO_DIR = Path(__file__).resolve().parents[3] / "data" / "geo"

MOSCOW_POLYGONS_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow.geojson")
MOSCOW_AREAS_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow_areas.geojson")
OBLAST_POLYGONS_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow_oblast.geojson")
OBLAST_CITIES_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow_oblast_cities.json")
METRO_ABSOLUTE_PATH: str = str(GEO_DIR / "metro.geojson")
