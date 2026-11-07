import numpy as np
import polars as pl
import pytest

from flats_pipeline.modeling.linear import TimeDummyModel, predict_from_json
from flats_pipeline.modeling.monotone import MODEL_TARGET

MONTHS = ["2025_01", "2025_02", "2025_03", "2025_04"]
MOSCOW_TREND = [0.0, 0.02, 0.05, 0.08]
OBLAST_TREND = [0.0, 0.01, 0.01, 0.03]


def _synthetic(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    region = rng.choice(["moscow", "oblast"], n)
    month = rng.integers(0, len(MONTHS), n)
    area = rng.uniform(25, 120, n)
    kitchen = rng.uniform(5, 15, n)
    cond = rng.integers(0, 4, n).astype(float)
    trend = np.where(region == "moscow", np.take(MOSCOW_TREND, month), np.take(OBLAST_TREND, month))
    # kitchen has a (wrong-signed) negative effect: the sign constraint must hold it at >= 0
    y = 12 + 0.9 * np.log(area) - 0.01 * kitchen + 0.05 * cond + 0.4 * (region == "moscow") + trend
    y += rng.normal(0, 0.05, n)
    return pl.DataFrame({
        "total_square": area, "kitchen_square": kitchen, "condition_level": cond,
        "region_type": region, "total_room_count": rng.integers(1, 4, n).astype(str),
        "area_okrug": pl.Series([("ЦАО" if r == "moscow" else None) for r in region], dtype=pl.String), "local_area": np.where(region == "moscow", "Арбат", "Химки"),
        "year_month": np.take(MONTHS, month), MODEL_TARGET: y,
    })


@pytest.fixture(scope="module")
def fitted():
    df = _synthetic()
    m = TimeDummyModel(["total_square", "kitchen_square", "condition_level"],
                       ["total_room_count", "region_type", "local_area"], MONTHS, "year_month").fit(df)
    return df, m


def test_recovers_region_time_effects(fitted):
    _, m = fitted
    # region + its local corrections = the true trend (one local area per region here)
    for region, okrug, local, trend in (("moscow", "ЦАО", "Арбат", MOSCOW_TREND), ("oblast", None, "Химки", OBLAST_TREND)):
        total = np.add(m.time_effects("region", region), m.time_effects("local", local))
        if okrug:
            total = np.add(total, m.time_effects("okrug", okrug))
        assert np.allclose(total, trend, atol=0.004), (region, total)


def test_sign_constraints(fitted):
    _, m = fitted
    spec = m.to_json()
    coefs = {f["name"]: f["coef"] for f in spec["numeric"]}
    assert coefs["kitchen_square"] >= 0
    assert coefs["condition_level"] > 0


def test_json_matches_model(fitted):
    df, m = fitted
    spec = m.to_json()
    rows = df.head(20)
    expected = m.predict_log(rows)
    got = [predict_from_json(spec, r, r["year_month"]) for r in rows.iter_rows(named=True)]
    assert np.allclose(got, expected, atol=1e-9)
    # and the model rebuilt from the JSON predicts the same
    back = TimeDummyModel.from_json(spec, "year_month")
    assert np.allclose(back.predict_log(rows), expected, atol=1e-9)
