"""Flat price + hedonic indices web app.

Runs from the precomputed bundle (APP_DATA_DIR, see bundle.py). The only thing computed per request
is the model estimate of the entered flat (gradient boosting with month dummies): its features
(same code as training), its price in the latest month and its curve over all months (only the
month dummies change).

Run:  uvicorn flats_app.api:app --host 127.0.0.1 --port 8010
"""

import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Optional

from flats_app.settings import settings

os.environ.setdefault("POLARS_MAX_THREADS", str(settings.APP_THREADS))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from catboost import Pool  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, field_validator  # noqa: E402

from flats_app import form  # noqa: E402
from flats_app.bundle import load_bundle  # noqa: E402
from flats_app.features.features import calculate_features  # noqa: E402
from flats_app.local import add_areas  # noqa: E402

logger = logging.getLogger(__name__)

MAX_STOREYS = 100
MAX_TOTAL_SQUARE = 300
MAX_ROOMS = 20
MAX_BALCONIES = 5
MAX_BATHROOMS = 5
MIN_BUILT_YEAR = 1800
# Moscow + Moscow Oblast, with a margin
LAT_RANGE, LON_RANGE = (54.2, 56.99), (35.1, 40.25)
AREA_LEVELS = {
    "region": "Регион",
    "okrug": "Административный округ Москвы",
    "raion": "Район Москвы",
    "go": "Городской округ Московской области",
}
ALL_AREA = ("all", "Москва и область")

FIELD_LABELS = {
    "total_room_count": "Количество комнат",
    "total_square": "Общая площадь",
    "kitchen_square": "Площадь кухни",
    "life_square": "Жилая площадь",
    "balcony_count": "Количество балконов",
    "storey": "Этаж квартиры",
    "storeys_count": "Количество этажей в доме",
    "bathroom_count": "Количество санузлов",
    "elevator_count": "Количество лифтов",
    "built_year": "Год постройки",
    "lat": "Широта",
    "lon": "Долгота",
}

# --- precomputed bundle, loaded once -----------------------------------------------------
B = load_bundle(settings.APP_DATA_DIR, threads=settings.APP_THREADS)
META = B.meta
CONTINUOUS = META["continuous_features"]
CATEGORICAL = META["categorical_features"]
FEATURE_ORDER = META["feature_order"]  # continuous + month dummies + month number + categorical
TIME_DUMMIES = META["time_dummies"]
TIME_INDEX = META.get("time_index_feature")  # month number 0..n-1 (None in older bundles)
FILL = META.get("continuous_fill", {})
MONTHS = B.months
PERIODS = B.periods
CONDITION_LEVELS = META["condition_levels"]
CONDITION_DEFAULT = META["condition_default"]
CONDITION_FEATURE = META["condition_feature"]
LOCAL_PRICE = META["local_price_feature"]

app = FastAPI(title="Индекс цен квартиры")


def room_bucket(rooms: int) -> str:
    """Same buckets as the pipeline (hedonic.room_bucket)."""
    if rooms <= 0:
        return "студия"
    return "4+" if rooms >= 4 else str(rooms)


def _bad_chars(raw: str) -> str:
    found = re.findall(r"[^\d.,\s\-]", raw)
    return "".join(dict.fromkeys(found)) or raw


class FlatRequest(BaseModel):
    total_room_count: int
    total_square: float
    kitchen_square: float
    life_square: float
    balcony_count: int
    is_apartment: int = 0
    storey: int
    storeys_count: int
    bathroom_count: int
    elevator_count: int
    lat: float
    lon: float
    apartment_condition: str
    built_year: int
    building_batch_name: Optional[str] = None
    sale_type_name: str = "прямая продажа"

    @field_validator("total_square", "kitchen_square", "life_square", "lat", "lon", mode="before")
    @classmethod
    def coerce_decimal(cls, v, info):
        raw = str(v).strip()
        label = FIELD_LABELS.get(info.field_name, info.field_name)
        if not raw:
            raise ValueError(f"Заполните поле «{label}»")
        try:
            return float(raw.replace(",", ".").replace(" ", ""))
        except ValueError:
            raise ValueError(f"Уберите текст «{_bad_chars(raw)}» из поля «{label}»")

    @field_validator(
        "total_room_count", "balcony_count", "storey", "storeys_count", "bathroom_count",
        "elevator_count", "built_year", mode="before",
    )
    @classmethod
    def coerce_int(cls, v, info):
        raw = str(v).strip()
        label = FIELD_LABELS.get(info.field_name, info.field_name)
        if not raw:
            raise ValueError(f"Заполните поле «{label}»")
        try:
            f = float(raw.replace(",", ".").replace(" ", ""))
        except ValueError:
            raise ValueError(f"Уберите текст «{_bad_chars(raw)}» из поля «{label}»")
        if f != int(f):
            raise ValueError(f"«{label}» должно быть целым числом")
        return int(f)


def check_flat(d: FlatRequest) -> list[str]:
    """Business rules. Returns the list of problems (empty = ok)."""
    errors = []
    for field in ["total_room_count", "total_square", "kitchen_square", "life_square", "balcony_count",
                  "storey", "storeys_count", "bathroom_count", "elevator_count", "built_year"]:
        if getattr(d, field) < 0:
            errors.append(f"«{FIELD_LABELS[field]}» не может быть отрицательным")
    if d.total_square <= 0:
        errors.append("Общая площадь должна быть больше 0")
    elif d.total_square > MAX_TOTAL_SQUARE:
        errors.append(f"Общая площадь не может быть больше {MAX_TOTAL_SQUARE} м²")
    if d.kitchen_square >= 0 and d.life_square >= 0 and d.kitchen_square + d.life_square >= d.total_square:
        errors.append(
            f"Кухня + жилая площадь ({d.kitchen_square:g} + {d.life_square:g} = "
            f"{d.kitchen_square + d.life_square:g} м²) должны быть меньше общей ({d.total_square:g} м²)"
        )
    if not 1 <= d.storeys_count <= MAX_STOREYS:
        errors.append(f"Этажей в доме: от 1 до {MAX_STOREYS}")
    if d.storey < 1:
        errors.append("Этаж квартиры должен быть не меньше 1")
    elif d.storey > d.storeys_count:
        errors.append(f"Этаж квартиры ({d.storey}) не может быть больше этажности дома ({d.storeys_count})")
    if d.balcony_count > MAX_BALCONIES:
        errors.append(f"Балконов не может быть больше {MAX_BALCONIES}")
    if d.bathroom_count > MAX_BATHROOMS:
        errors.append(f"Санузлов не может быть больше {MAX_BATHROOMS}")
    if d.total_room_count > MAX_ROOMS:
        errors.append(f"Комнат не может быть больше {MAX_ROOMS}")
    if not MIN_BUILT_YEAR <= d.built_year <= date.today().year + 5:
        errors.append(f"Год постройки: от {MIN_BUILT_YEAR} до {date.today().year + 5}")
    if not (LAT_RANGE[0] <= d.lat <= LAT_RANGE[1] and LON_RANGE[0] <= d.lon <= LON_RANGE[1]):
        errors.append("Точка на карте должна быть в Москве или Московской области")
    if d.apartment_condition not in CONDITION_LEVELS:
        errors.append("Выберите состояние квартиры из списка")
    return errors


def flat_features(d: FlatRequest) -> pl.DataFrame:
    """One-row frame through the same feature code as training (IEP variant)."""
    # Always estimated as a flat (not апартаменты) in a direct sale (прямая продажа).
    d = d.model_copy(update={"is_apartment": 0, "sale_type_name": "прямая продажа"})
    df = pl.DataFrame([d.model_dump()]).rename({"lat": "location.lat", "lon": "location.lon"}).with_columns(
        pl.lit("flat").alias("guid"),
        (pl.col("storey") == 1).cast(pl.Int8).alias("storey_first"),
        (pl.col("storey") == pl.col("storeys_count")).cast(pl.Int8).alias("storey_last"),
        (pl.col("storey") / pl.col("storeys_count")).alias("storey_relative"),
        pl.lit(1.0).alias("price"),  # not used by the model; the feature code expects it
    )
    feats = add_areas(calculate_features(df))
    # The form offers the merged condition levels: the level is its position in the list.
    feats = feats.with_columns(pl.lit(float(CONDITION_LEVELS.index(d.apartment_condition))).alias(CONDITION_FEATURE))
    local = B.local.query(feats["location.lat"].to_numpy(), feats["location.lon"].to_numpy())
    return feats.with_columns(pl.Series(LOCAL_PRICE, local))


def model_input(features: pl.DataFrame) -> pl.DataFrame:
    """Fill gaps as in training: continuous inf/NaN -> training mean; categorical -> "missing"."""
    return features.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c).cast(pl.Float64))
            .otherwise(pl.lit(FILL.get(c, 0.0))).alias(c)
            for c in CONTINUOUS
        ]
        + [pl.col(c).cast(pl.String).fill_null("missing") for c in CATEGORICAL]
    ).select(CONTINUOUS + CATEGORICAL)


def area_series(level: str, name: str | None, rooms: str | None = None) -> dict | None:
    a = B.area(level, name)
    if not a or all(v is None for v in a["hedonic_index"]):
        return None
    return {
        "level": level,
        "label": AREA_LEVELS.get(level, "Москва и область"),
        "name": name,
        "hedonic_price_sq": a["hedonic_price_sq"],
        "hedonic_index": a["hedonic_index"],
        "median_price_sq": a["median_price_sq"],
        "median_index": a["median_index"],
        "n_sales": a["n_sales"],
        "similar_median_price_sq": (a["similar_median_price_sq"].get(rooms) or [None] * len(PERIODS)) if rooms else None,
    }


@app.post("/estimate")
def estimate(d: FlatRequest):
    errors = check_flat(d)
    if errors:
        return JSONResponse(status_code=422, content={"detail": errors})

    feats = flat_features(d)
    x = model_input(feats)
    # One row per month: the flat as if sold in that month (month dummy = 1, the others 0; month number).
    by_month = x.join(pl.DataFrame({"month_no": range(len(MONTHS))}), how="cross").with_columns(
        [(pl.col("month_no") == i).cast(pl.Float64).alias(dummy) for i, dummy in enumerate(TIME_DUMMIES)]
        + ([pl.col("month_no").cast(pl.Float64).alias(TIME_INDEX)] if TIME_INDEX else [])
    ).select(FEATURE_ORDER)
    # The model predicts log(total price) (monotone in the areas and condition); per m2 = total / area.
    # Plain rows (no pandas / pyarrow needed); columns in training order.
    pool = Pool(by_month.rows(), cat_features=CATEGORICAL, feature_names=FEATURE_ORDER)
    price_sq = np.exp(B.model.predict(pool)) / d.total_square
    latest = float(price_sq[-1])

    place = feats.row(0, named=True)
    rooms = room_bucket(d.total_room_count)
    areas = []
    if place["area_region"]:
        areas.append(area_series("region", place["area_region"], rooms))
        if place["region_type"] == "moscow":
            areas += [area_series("okrug", place["area_okrug"], rooms), area_series("raion", place["area_raion"], rooms)]
        else:
            areas.append(area_series("go", place["area_go"], rooms))
    else:
        areas.append(area_series(*ALL_AREA))
    areas = [a for a in areas if a]
    # Compare with similar flats: the smallest area with a recent median, else the region.
    compare = None
    for a in reversed(areas):
        sim = a.get("similar_median_price_sq") or []
        if sim and sim[-1]:
            compare = {"area": a["name"], "rooms": rooms, "median_price_sq": sim[-1],
                       "diff_pct": round(100 * (latest / sim[-1] - 1), 1)}
            break

    region = ("region", place["area_region"]) if place["area_region"] else ALL_AREA
    region_area = B.area(*region)
    return {
        "price_per_m2": round(latest, 0),
        "price_total": round(latest * d.total_square, 0),
        "latest_period": PERIODS[-1],
        "periods": PERIODS,
        "flat": {
            "price_sq": [round(float(p), 0) for p in price_sq],
            "index": [round(100 * float(p) / float(price_sq[0]), 2) for p in price_sq],
        },
        "place": {
            "region": place["area_region"] or "за пределами Москвы и области",
            "okrug": place["area_okrug"],
            "raion": place["area_raion"],
            "go": place["area_go"],
            "metro": place["metro_name"],
            "metro_km": round(place["metro_distance_m"], 2) if place["metro_distance_m"] is not None else None,
        },
        "areas": areas,
        "region_median": {
            "name": region[1],
            "index": region_area["median_index"] if region_area else [None] * len(PERIODS),
        },
        "rooms": rooms,
        "compare": compare,
        "local_price_sq": round(float(np.exp(place[LOCAL_PRICE])), 0),
    }


@app.get("/health")
def health():
    """For docker / load balancers: the bundle is loaded and which data it holds."""
    m = B.manifest
    return {"status": "ok", "exported_at": m.get("exported_at"), "last_sale_date": m.get("last_sale_date"),
            "periods": [PERIODS[0], PERIODS[-1]]}


@app.exception_handler(Exception)
async def unhandled(request, exc):
    logger.exception("estimate failed")
    return JSONResponse(status_code=500, content={"detail": [f"Внутренняя ошибка: {exc}"]})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        msg = err.get("msg", "")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        elif err.get("type") == "missing":
            field = (err.get("loc") or ["поле"])[-1]
            msg = f"Поле «{FIELD_LABELS.get(str(field), field)}» обязательно"
        messages.append(msg)
    return JSONResponse(status_code=422, content={"detail": messages})


PAGE = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def page():
    building = '<option value="">— не знаю —</option>' + "".join(
        f'<option value="{b}">{b}</option>' for b in form.BUILDING_BATCHES
    )
    condition = "".join(
        f'<option value="{c}"{" selected" if c == CONDITION_DEFAULT else ""}>{c}</option>'
        for c in CONDITION_LEVELS
    )
    return (
        PAGE.replace("{{YANDEX_KEY}}", settings.YANDEX_MAPS_API_JS_KEY.get_secret_value())
        .replace("{{BUILDING_OPTIONS}}", building)
        .replace("{{CONDITION_OPTIONS}}", condition)
        .replace("{{BASE_PERIOD}}", PERIODS[0])
        .replace("{{LATEST_PERIOD}}", PERIODS[-1])
        .replace("{{MAX_STOREYS}}", str(MAX_STOREYS))
        .replace("{{MAX_TOTAL_SQUARE}}", str(MAX_TOTAL_SQUARE))
        .replace("{{MAX_BALCONIES}}", str(MAX_BALCONIES))
        .replace("{{MAX_BATHROOMS}}", str(MAX_BATHROOMS))
    )
