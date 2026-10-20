"""Geo reference files inside the bundle (APP_DATA_DIR/geo)."""

from flats_app.settings import settings

GEO_DIR = settings.APP_DATA_DIR / "geo"

MOSCOW_POLYGONS_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow.geojson")
MOSCOW_AREAS_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow_areas.geojson")
OBLAST_POLYGONS_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow_oblast.geojson")
OBLAST_CITIES_ABSOLUTE_PATH: str = str(GEO_DIR / "moscow_oblast_cities.json")
METRO_ABSOLUTE_PATH: str = str(GEO_DIR / "metro.geojson")
