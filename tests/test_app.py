import json

import pytest

from flats_app.settings import settings

if not (settings.APP_DATA_DIR / "manifest.json").exists():
    pytest.skip("no app bundle (run: python -m flats_pipeline export-app)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from flats_app.api import CONDITION_LEVELS, app  # noqa: E402

client = TestClient(app)
REFERENCE = settings.APP_DATA_DIR / "checks" / "reference_estimates.json"
FLAT = {
    "total_room_count": "2", "total_square": "54", "kitchen_square": "9", "life_square": "32",
    "balcony_count": "1", "is_apartment": 0, "storey": "5", "storeys_count": "12", "bathroom_count": "1",
    "elevator_count": "1", "built_year": "1985", "apartment_condition": "среднее состояние",
    "building_batch_name": None, "sale_type_name": "прямая продажа",
}


def test_moscow_flat():
    r = client.post("/estimate", json={**FLAT, "lat": 55.7558, "lon": 37.6173})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["price_per_m2"] > 100_000
    assert body["price_total"] == pytest.approx(body["price_per_m2"] * 54, rel=1e-6)
    assert body["flat"]["index"][0] == 100
    assert body["flat"]["price_sq"][-1] == body["price_per_m2"]
    levels = [a["level"] for a in body["areas"]]
    assert levels == ["region", "okrug", "raion"]
    assert body["areas"][0]["name"] == "Москва"
    assert all(len(a["hedonic_index"]) == len(body["periods"]) for a in body["areas"])


def test_oblast_flat():
    r = client.post("/estimate", json={**FLAT, "lat": 55.9116, "lon": 37.7307})  # Мытищи
    assert r.status_code == 200, r.text
    body = r.json()
    assert [a["level"] for a in body["areas"]] == ["region", "go"]
    assert body["areas"][0]["name"] == "Московская область"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"storey": "13"}, "не может быть больше этажности"),
        ({"storeys_count": "101", "storey": "5"}, "от 1 до 100"),
        ({"kitchen_square": "20", "life_square": "34"}, "должны быть меньше общей"),
        ({"balcony_count": "-1"}, "не может быть отрицательным"),
        ({"total_square": "-54"}, "не может быть отрицательным"),
        ({"total_square": "54 м2"}, "Уберите текст"),
        ({"total_square": "301", "life_square": "200", "kitchen_square": "30"}, "не может быть больше 300"),
        ({"lat": 59.93, "lon": 30.31}, "Москве или Московской области"),
    ],
)
def test_rules(change, message):
    r = client.post("/estimate", json={**FLAT, "lat": 55.7558, "lon": 37.6173, **change})
    assert r.status_code == 422
    assert any(message in m for m in r.json()["detail"]), r.json()


def test_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "{{" not in r.text
    assert "Тип продажи" not in r.text and "Апартаменты" not in r.text


def _price(**change):
    r = client.post("/estimate", json={**FLAT, "lat": 55.6505, "lon": 37.7440, **change})
    assert r.status_code == 200, r.text
    return r.json()["price_total"]


def test_monotone_total_square():
    prices = [_price(total_square=str(t)) for t in [45, 50, 54, 60, 70, 85]]
    assert prices == sorted(prices), prices


def test_monotone_kitchen_and_life():
    kitchen = [_price(kitchen_square=str(k)) for k in [5, 7, 9, 12, 15]]
    life = [_price(life_square=str(v)) for v in [20, 25, 32, 38, 42]]
    assert kitchen == sorted(kitchen), kitchen
    assert life == sorted(life), life


def test_monotone_balcony_and_bathroom():
    balcony = [_price(balcony_count=str(b)) for b in [0, 1, 2, 3]]
    bathroom = [_price(bathroom_count=str(b)) for b in [1, 2, 3]]
    assert balcony == sorted(balcony), balcony
    assert bathroom == sorted(bathroom), bathroom


def test_monotone_condition():
    prices = [_price(apartment_condition=c) for c in CONDITION_LEVELS]  # worse -> better
    assert prices == sorted(prices), prices
    assert len(set(prices)) == len(prices), prices  # every level gives its own price


def test_sale_type_and_apartment_are_fixed():
    # whatever is sent, the flat is estimated as a direct sale of a regular flat
    assert _price() == _price(sale_type_name="альтернатива", is_apartment=1)


def test_region_median_line():
    m = client.post("/estimate", json={**FLAT, "lat": 55.7558, "lon": 37.6173}).json()["region_median"]
    o = client.post("/estimate", json={**FLAT, "lat": 55.9116, "lon": 37.7307}).json()  # oblast flat
    assert m["name"] == "Москва" and o["region_median"]["name"] == "Московская область"
    assert len(o["region_median"]["index"]) == len(o["periods"]) and o["region_median"]["index"][0] == 100
    assert o["region_median"]["index"] != m["index"]


def test_house_series_matters():
    prices = {s: _price(building_batch_name=s) for s in [None, "П-44", "II-18", "индивидуальный проект"]}
    assert len(set(prices.values())) > 1, prices


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_same_answers_as_pipeline():
    """The app's own copy of the feature code and of the linear model gives what the pipeline gives.

    The bundle holds curves of a few flats computed by flats_pipeline (export-app) with its code.
    """
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for case in ref["cases"]:
        r = client.post("/estimate", json=case["request"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["flat"]["price_sq"] == pytest.approx(case["boosting_price_sq"], abs=1), case["request"]
        assert body["linear"]["price_sq"] == pytest.approx(case["linear_price_sq"], abs=1), case["request"]


def test_linear_curve_follows_its_area():
    """Time-dummy model: the flat's index is the index of its area (same time effects for all flats)."""
    body = client.post("/estimate", json={**FLAT, "lat": 55.6505, "lon": 37.7440}).json()
    local = [a for a in body["areas"] if a["level"] == "raion"][0]
    assert body["linear"]["index"] == pytest.approx(local["hedonic_index_linear"], abs=0.05)


def test_linear_monotone_and_condition():
    prices = [_price_linear(total_square=str(t)) for t in [45, 54, 70]]
    assert prices == sorted(prices), prices
    cond = [_price_linear(apartment_condition=c) for c in CONDITION_LEVELS]
    assert cond == sorted(cond), cond


def _price_linear(**change):
    r = client.post("/estimate", json={**FLAT, "lat": 55.6505, "lon": 37.7440, **change})
    assert r.status_code == 200, r.text
    return r.json()["linear"]["price_total"]
