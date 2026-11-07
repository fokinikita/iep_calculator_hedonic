# flats_pipeline

Weekly data collection for Moscow / Moscow Oblast secondary-market flats.
Raw exports → price history → sold flats dataset (input for `top_flip` model training).

## Inputs

Two exports, downloaded every Wednesday after 14:00 MSK:

| File | Content |
|---|---|
| `for-analytics.msk_habit.offer_pub_list(N).csv.gz` | **prices**: full publication history per `guid` (`start_pub_datetime, end_pub_datetime, price, deal_status_id`) |
| `for-analytics.msk_habit.offer(N).csv.gz` | **offers**: one row per `guid` with flat attributes (square, rooms, address, note, …) |

The file type is detected from the CSV header, not the name (e.g. archive `offer(65)` is really a prices file).

## Rules

- `deal_status_id = 1`: listed (price can change). `3`: removed from the site (can come back).
- Each prices export holds the **full** history of every guid in it, and weekly exports overlap.
  For each guid the history is taken from the **newest** export that contains it.
- The exports are **not** snapshots of all listings: they hold lots **closed in the last ~90 days**
  (a guid enters the first export after it closes and stays ~13 weeks). So the first export of a
  guid's latest run of exports tells the week it closed (`silver/export_presence.parquet`).
- **Closed** = last event is a removal (status 3), or the lot entered the exports while its last row
  is still "listed" (the source missed the status-3 row; ~138k lots in Jan–May 2026).
  A lot whose last event is a listing *after* it entered the exports came back and is active.
- **Sold** = closed more than `SOLD_AFTER_DAYS` (14) days before the newest snapshot. If a lot is
  removed again later and stays removed, the final removal counts as the sale.
- **Sale date** `sold_date` (bounds `sold_date_min`..`sold_date_max`, source in `sold_date_source`):
  - `status_3`: removal date recorded by the source;
  - `export_window`: no status-3 row; midpoint between the last export without the lot (or its last
    listing event) and the first export with it (~7 days precision);
  - `catch_up`: removal stamped in a catch-up batch right after an outage of the source's removal
    detection (weeks with almost no removals: Dec 15 2025, Jun 15–28 2026); midpoint of the outage
    start (or last listing event) and the batch date. Outages are detected automatically each run.
- **One flat, several listings.** ~20% of closed listings are not sales: the same flat is re-posted
  under a new guid (new object code, a deal with a deposit fell through, developer and agent both list it,
  copies on several sites). Listings are grouped into flats and sale episodes (`episodes.py`):
  - same flat = same region, street, building (missing = equal), area, floor and number of storeys,
    within 200 m of each other; **or** the same description text (copy-pasted between sites; object
    codes, deposit notes and dates are ignored) with area within 0.5 m² and the same floor. Short texts
    and texts shared by more than 10 listings (developer templates) are not used;
  - listings of a flat belong to one episode while each next one starts ≤ 30 days after the earlier
    ones closed; the episode is **sold** when all are closed: sale = last closure, price = its last
    price, `sell_days_count` = from the episode's first listing.
  - Checked by hand on random samples (`review-relists`): almost all matches are the same flat.
  - The exports contain only closed listings, so a relist still on sale is invisible until it closes:
    a recent sale can turn into a relist in a later weekly run.
- Known source quirk, not corrected: Nov 2025 has a burst of listings that closed within 14 days.
- `sell_days_count` = sale date − first publication date (the whole exposure, gaps included).
  `active_days` = days actually listed (status 1).
- Flat attributes also come from the newest offers export that contains the guid.
- The snapshot date of a prices export is the newest date inside it. Offers are ordered by download time.

## Layout (`DATA_DIR`, default `D:/data_winner/flats_pipeline`)

```
manifest.json                      every raw file seen: sha256, kind, snapshot, S3 key, errors
bronze/prices/prices_<snapshot>_<sha>.parquet   one parquet per raw export (also uploaded to S3)
bronze/offers/offers_<date>_<sha>.parquet
silver/price_history.parquet       deduplicated history, one row per event
silver/export_presence.parquet     per guid: first export of its latest run of exports (= closure week)
silver/lots.parquet                one row per guid (listing): lot_status, sale date + bounds, price stats, price_history
silver/flat_episodes.parquet       per guid: flat_id, episode_id, episode status, first listing date, listing guids
silver/offers_latest/*.parquet     newest attributes per guid (folder of parts: one huge wide file needs too much memory to read)
gold/sold_flats_<as_of>/*.parquet  one row per sold flat episode (its last listing) + attributes (folder of parts, read with scan_parquet(folder/'*.parquet'))
reports/sales_<as_of>.txt|csv      monthly sales by sale-date source, outages, statuses (after every build)
reports/relist_review*.csv         sampled flats with several listings, for checking the matching by hand
logs/pipeline_<date>.log
```

`lot_status` values: `sold`, `removed_recently` (≤ 14 days, may still come back), `active`,
`active_not_in_latest_export` (last seen listed but missing from the newest export), `other`.

S3 (bucket from `HEDONIC_S3_BUCKET` in `.env.pipeline`): `s3://<bucket>/<S3_PREFIX>/bronze/...`, plus `silver/lots.parquet` and `gold/sold_flats_<as_of>/`.

## Modelling: dataset -> features -> hedonic index

Code copied from top_flip and iep_calculator into `src/flats_pipeline/modeling/` (geo files in `data/geo/`):

| Step | Command | Code | Output in `DATA_DIR/modeling/` |
|---|---|---|---|
| prepare + clean | `dataset` | `preprocessors/stats_preprocessor.py` (top_flip), `preprocessors/cleaner.py` | `dataset_<as_of>.parquet` |
| features | `features [--variant iep\|full]` | `features/` (top_flip `services/features`), `build_features.py` (= `run_calculate_features_iep.py`) | `features_iep/{train,valid,test}_features.parquet`, `features_meta.json` |
| hedonic indices | `hedonic-index [--start-month 2025_06]` | `hedonic.py` (boosting), `linear.py` (time-dummy regression) | `hedonic/hedonic_index.{parquet,xlsx}`, `chart_index.png` (both models), `model_index.cbm`, `linear_model.json`, `area_indices.parquet` |
| out-of-time test | `evaluate` | `hedonic.py` | `hedonic/forward_test.json` (both models) |
| model for the API | `hedonic-model` | `hedonic.py` (= `train_hedonic_model.py`) | `hedonic/model_hedonic.cbm`, `features_meta_hedonic.json` |
| web app bundle | `export-app` (also at the end of `model-all`) | `export_app.py`, `single_flat.py` | `APP_BUNDLE_DIR` = `Desktop/iep_calculator_hedonic/data` (both models, JSON indices, local prices, geo, reference answers) |

`python -m flats_pipeline model-all` runs dataset -> features -> hedonic-index. Notebooks (flats_pipeline venv):
`notebooks/hedonic_index.ipynb` (all steps) and `notebooks/get_dataset.ipynb` (dataset, plus a copy to `top_flip/data`).

- **Cleaning** (on top of `StatsPreprocessor`): price per m² outside 30k–3M ₽, price < 500k ₽, area > 500 m²,
  buildings over 100 storeys are dropped; built year < 1800 -> missing; broken encodings fixed (~0.5% of rows).
- **Features** are the same as top_flip's (checked column by column on a sample); the geo part is vectorized
  (nearest metro via BallTree, polygons via shapely), ~10x faster.
- **Hedonic indices**: target log(total price), IEP features + month dummies (`t_<month>` = 1 in the sale
  month). Each month: 80% of sales train the models, 20% form a fixed basket that is priced in every month;
  index = basket average, first month = 100. Months with fewer than 1000 sales are left out. Two models:
  - **gradient boosting** (CatBoost, depth 6, l2 50, every tree on a random 80% of the training sales,
    monotone in areas / balconies / bathrooms / condition), one dummy for every month;
  - **linear time-dummy regression** (`linear.py`): flat features + region × month dummies (first month = base)
    + local corrections by округ and район / городской округ whose month-to-month change is penalised, so
    an area follows its region unless its own sales show otherwise. All flats of an area share one curve.
  The Excel and `chart_index.png` show both, with the median ₽/m² index (no quality adjustment).

### Single-flat curves (checked 2026-10-20)
The curve of one flat over months (only the month changes) used to jump: the depth-8 model with `year_month`
let the month interact with everything. Measured on 20,000 held-out flats:

| | MAPE (held-out, own month) | MAPE (next 3 months) | zig-zag of a flat's curve | flats with a >5% monthly jump against their region |
|---|---|---|---|---|
| old boosting (depth 8, `year_month` categorical) | 11.3% | 11.8% | 2.0% | 24% |
| **boosting** (depth 6, l2 50, 80% rows, month dummies) | 11.7% | 12.0% | 1.1% | 3.3% |
| **linear time dummies** | 14.1% | 14.4% | 0.55% | 0% |

- With the first month as "all dummies 0" (as in a regression) the boosting pooled it with other months and
  the whole index shifted by ~3%: boosting has a dummy for every month, the regression drops the first.
- Month as one ordered number (0..16) instead of dummies was smoother still for boosting (zig-zag ~0.6%,
  MAPE 11.5%): trees then split "before / after a month" instead of singling out months.
- Both indices agree: +18.8% (boosting) and +18.0% (linear) from 2025-05 to 2026-09.
- Modelling commands run with a higher memory cap, `MODEL_MAX_MEMORY_GB` (default 6); CatBoost uses `CATBOOST_THREADS` (4).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[modeling]   # [modeling]: catboost, shapely, scikit-learn, ...
copy .env.example .env   # fill S3 creds and, optionally, download links
```

(AVG's HTTPS scanning breaks Poetry's SSL check, so use pip. The pipeline itself trusts the Windows certificate store via `truststore`, so S3 and downloads work with AVG on.)

## Run

```powershell
.venv\Scripts\python -m flats_pipeline run      # download -> ingest -> S3 -> rebuild -> S3
.venv\Scripts\python -m flats_pipeline ingest   # only new raw files -> bronze
.venv\Scripts\python -m flats_pipeline build    # only rebuild silver/gold
.venv\Scripts\python -m flats_pipeline upload   # only push missing bronze + latest lots/sold files to S3
.venv\Scripts\python -m flats_pipeline run --no-s3
.venv\Scripts\python -m flats_pipeline report                    # monthly sales report -> DATA_DIR/reports
.venv\Scripts\python -m flats_pipeline review-relists --n 50     # sample flats with several listings -> CSV
.venv\Scripts\python -m flats_pipeline review-relists --by-note  # only flats linked by description text
```

Ingest is idempotent: files already seen (same path/size/mtime or same sha256) are skipped, and
empty or broken downloads are retried on the next run. Every run fully recalculates silver/gold from bronze.

## Memory

This PC has little free memory, so the pipeline is built to stay small (peak ~1.6 GB on the full history):
- every step works one export file at a time, newest first, and keeps only guids not yet seen
- the process has a hard cap (`MAX_MEMORY_GB`, default 2.5). If it ever needs more, the pipeline fails alone and other apps are not affected
- polars uses `POLARS_MAX_THREADS` (default 2) threads, and parquet files are written with 10k-row groups
- a run that crashes releases its lock automatically

## Schedule

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
```

This registers the Windows task `flats_pipeline_weekly`: Wednesday 14:30 plus a Thursday 10:00 retry,
and it runs late if the PC was off. Remove it with `Unregister-ScheduledTask flats_pipeline_weekly`.

If `PRICES_URL` / `OFFERS_URL` are empty, the job only picks up files you have put into `INBOX_DIRS`.

## Web app (price + hedonic indices)

The app is the separate git repo `Desktop/iep_calculator_hedonic` (Docker / docker compose). It needs only the
bundle written by `python -m flats_pipeline export-app` (~22 MB), not the training data; see its README.
The bundle includes reference answers for a few flats computed here (`single_flat.py`): the app's own copy of
the feature code and of the linear model is tested against them.

### Why the estimate is not the area median (checked 2026-10-20)
- The median of *all* flats of an area mixes studios (m² up to +25% vs the area median) with big flats (−10–20%),
  new builds with old houses, and cheap and expensive parts of the area. Compare with similar flats and with the
  nearest sales instead, both shown in the app.
- The first API model (trained like iep_calculator `train_hedonic_model.py`, only flats first listed before
  2025-12) underestimated: actual prices 1.8% above the prediction on average, 3–3.5% for older houses, and up to
  ±15–19% by район (e.g. Хамовники), because 2026 months had only flats that sat on the market 100–320 days and
  the model knew only the округ. Fixed: one model trained on every month, plus the район / городской округ and
  the local price level (30 nearest other sales) as features (`modeling/local.py`). On held-out sales: median
  actual/prediction 0.995–1.005 for every region, size, age, condition, floor and month; all Moscow округа within
  ±0.2%, районы within −3…+4%; MAPE 13.2% -> 11.2%.
