# iep_calculator_hedonic

Web app: estimate of a flat's price per m² and price indices (Москва / Московская область, округ,
район, городской округ). It runs from a small **precomputed bundle** in `data/`. No training data or
parquet files are needed. Per request it computes only the estimate of the entered flat by the
gradient boosting model (CatBoost) with simple month dummies plus the month number (0, 1, 2, ...):
the flat's curve on the chart is the same flat with only the month changed. The model is regularised
(depth 6, l2 50, every tree on a random 80% of the sales), and the month number lets trees pool
neighbouring months, so a single flat's curve does not jump from month to month, also in the center
(ЦАО: zig-zag 0.6% a month, no flat with a >5% monthly jump).

The linear time-dummy variant (a second model and chart) is on the branch `linear-time-dummies`.

## Layout

```
iep_calculator_hedonic/
  data/                         the bundle (made by flats_pipeline, ~24 MB)
    manifest.json               export time, data period, model quality, file list with sha256
    model/model.cbm             CatBoost model (calibration already inside)
    model/model_meta.json       feature order (with month dummies), fill values, months, conditions
    model/linear_model.json     linear time-dummy model (used on the branch linear-time-dummies)
    indices/areas.json          per area and month: hedonic and median price per m²,
                                indices, medians of similar flats by number of rooms
    checks/reference_estimates.json  curves of 8 flats computed by the pipeline (tests compare)
    local/local_price_reference.npz   coordinates + log price per m² of all sales (30 nearest)
    geo/                        polygons (Москва, районы, области) and metro
  src/flats_app/
    api.py                      FastAPI: /, /estimate, /health
    bundle.py                   loads and checks the bundle
    features/                   feature code for one flat (copy of the pipeline's, IEP variant)
    local.py                    районы / городские округа, price of the 30 nearest sales
    form.py, page.html          form options and the page
    settings.py                 settings from env / .env
  tests/                        rules, monotonicity, match with the pipeline's reference answers
  Dockerfile, docker-compose.yml, requirements.txt, .env.example
```

## Update the data

In `flats_pipeline` after retraining (`model-all` also does it at the end):

```
python -m flats_pipeline export-app            # writes this repo's data/ (APP_BUNDLE_DIR in the pipeline .env)
python -m flats_pipeline export-app --out D:\some\folder
```

The export is written to `data.tmp` and then swapped in, so the folder is never half-written.
Restart the app (or the container) to load the new data. No image rebuild is needed, because
`data/` is mounted into the container.

Every export also writes `data/checks/reference_estimates.json` (answers of the pipeline for 8 flats);
`pytest` checks the app gives the same, so a retrained bundle is checked too.

## Run locally

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
copy .env.example .env          # fill in YANDEX_MAPS_API_JS_KEY
scripts\run_local.ps1           # http://127.0.0.1:8010
.venv\Scripts\python -m pytest
```

## Run in Docker

```
docker compose up -d --build    # http://127.0.0.1:8010 (APP_PORT in .env)
```

- The image holds only the code and the runtime dependencies. The bundle is mounted read-only from `./data`.
- `/health` is used as the container health check.
- Memory: about 210 MB at rest. The compose file caps the container at 1 GB.

## Settings (env / .env)

| Variable | Default | Meaning |
|---|---|---|
| `YANDEX_MAPS_API_JS_KEY` | empty | map + address search on the page |
| `APP_DATA_DIR` | `./data` (`/app/data` in docker) | bundle folder |
| `APP_THREADS` | 2 | threads for the model and the nearest-sales search |
| `APP_PORT` | 8010 | host port for docker compose |
