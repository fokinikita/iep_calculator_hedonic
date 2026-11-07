import polars as pl

from flats_pipeline.modeling.features.geo.geo_features import GeoFeaturesService
from flats_pipeline.modeling.preprocessors.cleaner import DatasetCleaner, fix_mojibake


def test_fix_mojibake():
    assert fix_mojibake("РїСЂСЏРјР°СЏ РїСЂРѕРґР°Р¶Р°") == "прямая продажа"
    assert fix_mojibake("прямая продажа") == "прямая продажа"
    assert fix_mojibake("Рязанский проспект") == "Рязанский проспект"  # a normal word with Р stays
    assert fix_mojibake(None) is None


def test_cleaner_drops_outliers():
    df = pl.DataFrame(
        {
            "price": [10e6, 10e6, 300e6, 1e5, 10e6],
            "total_square": [50.0, 1000.0, 50.0, 40.0, 50.0],  # 1000 m2; 6M/m2; price 100k
            "storeys_count": [9, 9, 9, 9, 150],
            "built_year": [1700, 2000, 2000, 2000, 2000],
            "sale_type_name": ["РїСЂСЏРјР°СЏ РїСЂРѕРґР°Р¶Р°"] * 5,
        }
    )
    out = DatasetCleaner(df).clean()
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["built_year"] is None  # before 1800 -> missing
    assert row["sale_type_name"] == "прямая продажа"


def test_vectorized_geo_matches_row_by_row():
    points = pl.DataFrame(
        {
            "location.lat": [55.751, 55.70, 55.43297, 55.92406, 56.5, None, 55.311077],
            "location.lon": [37.617, 37.55, 37.73257, 37.43735, 39.9, None, 38.654255],
        }
    )
    svc = GeoFeaturesService(points)
    svc._ensure_data_loaded()
    svc.add_region_feature()
    svc.add_center_distance_feature()
    svc.add_metro_feature()
    got = svc.features
    for i, row in enumerate(points.iter_rows(named=True)):
        region_type, district, city = GeoFeaturesService.classify(
            row, svc.moscow_polygons, svc.moscow_areas, svc.oblast_polygons, svc.oblast_cities
        )
        assert got["region_type"][i] == region_type
        assert got["district_name"][i] == district
        assert got["city_name"][i] == city
        metro = GeoFeaturesService.nearest_metro(row, svc.metro_stations)
        assert got["metro_name"][i] == metro["metro_name"]
        if metro["metro_distance_m"] is not None:
            assert abs(got["metro_distance_m"][i] - metro["metro_distance_m"]) < 1e-6
            expected = GeoFeaturesService.haversine(row["location.lat"], row["location.lon"], 55.751028, 37.617611)
            assert abs(got["distance_m"][i] - expected) < 1e-9


def test_condition_merge():
    from flats_pipeline.modeling.monotone import add_condition_level

    df = pl.DataFrame({"apartment_condition": ["без отделки", "требуется капитальный ремонт", "хорошее состояние",
                                               "отличное состояние", "евроремонт", "дизайнерский", None]})
    out = add_condition_level(df)
    assert out["apartment_condition"].to_list() == [
        "требуется ремонт", "требуется ремонт", "среднее состояние", "хорошее состояние", "дизайнерский", "дизайнерский",
        "среднее состояние",
    ]
    assert out["condition_level"].to_list() == [0, 0, 1, 2, 3, 3, 1]
