"""Flat price + hedonic indices web app.

Runs from precomputed bundles (see bundle.py): the baseline one in APP_DATA_DIR (in git) and the
dated ones the weekly pipeline writes to APP_BUNDLES_DIR/<version>/. Each bundle is one model
version; the page picks a version and shows the other versions' estimate of the same flat beside it.
The only thing computed per request is the model estimate of the entered flat (gradient boosting
with month dummies): its features (same code as training), its price in the latest month and its
curve over all months (only the month dummies change).

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

# Index base: 12 months before the latest month (= 100). Earlier months are "history" (below / above 100).
BASE_MONTHS_BACK = 12
MAX_DATED_BUNDLES = 4  # newest dated bundles kept in memory (plus the baseline)


class Model:
    """One bundle = one model version, with everything a request needs."""

    def __init__(self, path: Path, version: str | None = None):
        b = load_bundle(path, threads=settings.APP_THREADS)
        self.B, self.path = b, path
        meta = b.meta
        self.version = version or b.manifest.get("model_built") or b.manifest.get("exported_at", "")[:10]
        self.continuous = meta["continuous_features"]
        self.categorical = meta["categorical_features"]
        self.feature_order = meta["feature_order"]  # continuous + month dummies + month number + categorical
        self.time_dummies = meta["time_dummies"]
        self.time_index = meta.get("time_index_feature")  # month number 0..n-1 (None in older bundles)
        self.fill = meta.get("continuous_fill", {})
        self.months = b.months
        self.periods = b.periods
        self.condition_levels = meta["condition_levels"]
        self.condition_default = meta["condition_default"]
        self.condition_feature = meta["condition_feature"]
        self.local_price = meta["local_price_feature"]
        self.base_idx = max(0, len(self.periods) - 1 - BASE_MONTHS_BACK)

    def info(self) -> dict:
        m = self.B.manifest
        fwd = ((m.get("quality") or {}).get("forward_test") or {}).get("boosting") or {}
        return {"version": self.version, "model_built": m.get("model_built"), "exported_at": m.get("exported_at"),
                "last_sale_date": m.get("last_sale_date"), "periods": [self.periods[0], self.periods[-1]],
                "n_sales": m.get("n_sales"), "forward_mape": fwd.get("mape"), "forward_wape": fwd.get("wape")}


def load_models() -> dict[str, Model]:
    """Baseline bundle + the newest dated bundles; a dated bundle wins over a baseline of the same version."""
    models = {}
    base = Model(settings.APP_DATA_DIR)
    models[base.version] = base
    root = settings.APP_BUNDLES_DIR
    dated = sorted(p for p in root.iterdir() if (p / "manifest.json").exists()) if root and root.is_dir() else []
    for p in dated[-MAX_DATED_BUNDLES:]:
        try:
            models[p.name] = Model(p, version=p.name)
        except Exception:  # noqa: BLE001 - one broken bundle must not take the app down
            logger.exception("Skip bundle %s", p)
    return dict(sorted(models.items()))


MODELS = load_models()
DEFAULT_MODEL = list(MODELS)[-1]  # newest
M0 = MODELS[DEFAULT_MODEL]
# Form options and limits come from the newest model (condition levels are the same in all).
CONDITION_LEVELS = M0.condition_levels
CONDITION_DEFAULT = M0.condition_default

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
    model: Optional[str] = None  # model version (bundle); default: the newest

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


def flat_features(d: FlatRequest, m: Model) -> pl.DataFrame:
    """One-row frame through the same feature code as training (IEP variant)."""
    # Always estimated as a flat (not апартаменты) in a direct sale (прямая продажа).
    d = d.model_copy(update={"is_apartment": 0, "sale_type_name": "прямая продажа"})
    df = pl.DataFrame([d.model_dump(exclude={"model"})]).rename({"lat": "location.lat", "lon": "location.lon"}).with_columns(
        pl.lit("flat").alias("guid"),
        (pl.col("storey") == 1).cast(pl.Int8).alias("storey_first"),
        (pl.col("storey") == pl.col("storeys_count")).cast(pl.Int8).alias("storey_last"),
        (pl.col("storey") / pl.col("storeys_count")).alias("storey_relative"),
        pl.lit(1.0).alias("price"),  # not used by the model; the feature code expects it
    )
    feats = add_areas(calculate_features(df))
    # The form offers the merged condition levels: the level is its position in the list.
    feats = feats.with_columns(
        pl.lit(float(m.condition_levels.index(d.apartment_condition))).alias(m.condition_feature)
    )
    local = m.B.local.query(feats["location.lat"].to_numpy(), feats["location.lon"].to_numpy())
    return feats.with_columns(pl.Series(m.local_price, local))


def model_input(features: pl.DataFrame, m: Model) -> pl.DataFrame:
    """Fill gaps as in training: continuous inf/NaN -> training mean; categorical -> "missing"."""
    return features.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c).cast(pl.Float64))
            .otherwise(pl.lit(m.fill.get(c, 0.0))).alias(c)
            for c in m.continuous
        ]
        + [pl.col(c).cast(pl.String).fill_null("missing") for c in m.categorical]
    ).select(m.continuous + m.categorical)


def flat_curve(d: FlatRequest, m: Model) -> tuple[np.ndarray, pl.DataFrame]:
    """Price per m2 of the flat in every month of the model, and its features."""
    feats = flat_features(d, m)
    x = model_input(feats, m)
    # One row per month: the flat as if sold in that month (month dummy = 1, the others 0; month number).
    by_month = x.join(pl.DataFrame({"month_no": range(len(m.months))}), how="cross").with_columns(
        [(pl.col("month_no") == i).cast(pl.Float64).alias(dummy) for i, dummy in enumerate(m.time_dummies)]
        + ([pl.col("month_no").cast(pl.Float64).alias(m.time_index)] if m.time_index else [])
    ).select(m.feature_order)
    # The model predicts log(total price) (monotone in the areas and condition); per m2 = total / area.
    # Plain rows (no pandas / pyarrow needed); columns in training order.
    pool = Pool(by_month.rows(), cat_features=m.categorical, feature_names=m.feature_order)
    return np.exp(m.B.model.predict(pool)) / d.total_square, feats


def area_series(m: Model, level: str, name: str | None, rooms: str | None = None) -> dict | None:
    a = m.B.area(level, name)
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
        "similar_median_price_sq": (a["similar_median_price_sq"].get(rooms) or [None] * len(m.periods)) if rooms else None,
        # hedonic price per m2 of the basket flats with the same number of rooms (None: too few)
        "hedonic_rooms_price_sq": a.get("hedonic_price_sq_by_rooms", {}).get(rooms) if rooms else None,
    }


@app.get("/models")
def models():
    """Model versions the dashboard serves (one per bundle), oldest first."""
    return {"default": DEFAULT_MODEL, "models": [m.info() for m in MODELS.values()]}


@app.post("/estimate")
def estimate(d: FlatRequest):
    errors = check_flat(d)
    if d.model and d.model not in MODELS:
        errors.append(f"Нет модели «{d.model}», есть: {', '.join(MODELS)}")
    if errors:
        return JSONResponse(status_code=422, content={"detail": errors})

    m = MODELS[d.model or DEFAULT_MODEL]
    price_sq, feats = flat_curve(d, m)
    latest = float(price_sq[-1])
    periods = m.periods

    place = feats.row(0, named=True)
    rooms = room_bucket(d.total_room_count)
    areas = []
    if place["area_region"]:
        areas.append(area_series(m, "region", place["area_region"], rooms))
        if place["region_type"] == "moscow":
            areas += [area_series(m, "okrug", place["area_okrug"], rooms),
                      area_series(m, "raion", place["area_raion"], rooms)]
        else:
            areas.append(area_series(m, "go", place["area_go"], rooms))
    else:
        areas.append(area_series(m, *ALL_AREA))
    areas = [a for a in areas if a]
    # Compare with similar flats: the smallest area with a recent median, else the region.
    compare = None
    for a in reversed(areas):
        sim = a.get("similar_median_price_sq") or []
        if sim and sim[-1]:
            compare = {"area": a["name"], "rooms": rooms, "median_price_sq": sim[-1],
                       "diff_pct": round(100 * (latest / sim[-1] - 1), 1)}
            break

    # The same flat by the other model versions (their own months).
    others = []
    for v, om in MODELS.items():
        if v == m.version:
            continue
        p, _ = flat_curve(d, om)
        others.append({"version": v, "periods": om.periods, "price_sq": [round(float(x), 0) for x in p],
                       "price_per_m2": round(float(p[-1]), 0), "latest_period": om.periods[-1],
                       "base_index": om.base_idx})

    region = ("region", place["area_region"]) if place["area_region"] else ALL_AREA
    region_area = m.B.area(*region)
    return {
        "model": m.version,
        "model_info": m.info(),
        "price_per_m2": round(latest, 0),
        "price_total": round(latest * d.total_square, 0),
        "latest_period": periods[-1],
        "periods": periods,
        "base_index": m.base_idx,
        "base_period": periods[m.base_idx],
        "flat": {
            "price_sq": [round(float(p), 0) for p in price_sq],
            "index": [round(100 * float(p) / float(price_sq[0]), 2) for p in price_sq],
        },
        "other_models": others,
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
            "index": region_area["median_index"] if region_area else [None] * len(periods),
        },
        "rooms": rooms,
        "compare": compare,
        "local_price_sq": round(float(np.exp(place[m.local_price])), 0),
    }


@app.get("/health")
def health():
    """For docker / load balancers: the bundles are loaded and which data the newest holds."""
    man = M0.B.manifest
    return {"status": "ok", "exported_at": man.get("exported_at"), "last_sale_date": man.get("last_sale_date"),
            "periods": [M0.periods[0], M0.periods[-1]], "models": list(MODELS), "default_model": DEFAULT_MODEL}


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
    model_opts = "".join(
        f'<option value="{v}"{" selected" if v == DEFAULT_MODEL else ""}>'
        f'{v} · данные по {mm.B.manifest.get("last_sale_date") or mm.periods[-1]}'
        f'{" (новая)" if v == DEFAULT_MODEL and len(MODELS) > 1 else ""}</option>'
        for v, mm in reversed(MODELS.items())
    )
    condition = "".join(
        f'<option value="{c}"{" selected" if c == CONDITION_DEFAULT else ""}>{c}</option>'
        for c in CONDITION_LEVELS
    )
    return (
        PAGE.replace("{{YANDEX_KEY}}", settings.YANDEX_MAPS_API_JS_KEY.get_secret_value())
        .replace("{{BUILDING_OPTIONS}}", building)
        .replace("{{CONDITION_OPTIONS}}", condition)
        .replace("{{MODEL_OPTIONS}}", model_opts)
        .replace("{{BASE_PERIOD}}", M0.periods[M0.base_idx])
        .replace("{{FIRST_PERIOD}}", M0.periods[0])
        .replace("{{LATEST_PERIOD}}", M0.periods[-1])
        .replace("{{MAX_STOREYS}}", str(MAX_STOREYS))
        .replace("{{MAX_TOTAL_SQUARE}}", str(MAX_TOTAL_SQUARE))
        .replace("{{MAX_BALCONIES}}", str(MAX_BALCONIES))
        .replace("{{MAX_BATHROOMS}}", str(MAX_BATHROOMS))
    )
