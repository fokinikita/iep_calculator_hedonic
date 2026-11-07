# iep_calculator_hedonic

Flat price estimate and hedonic price indices (Москва / Московская область).

## 1. Settings

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Set `YANDEX_MAPS_API_JS_KEY=` in `.env` (without it the map and address search do not load).

## 2. Run with Docker

```bash
docker compose up -d --build
```

Open http://127.0.0.1:8010 (port: `APP_PORT` in `.env`).

| What | Command |
|---|---|
| Status (wait for `healthy`) | `docker compose ps` |
| Logs | `docker compose logs -f` |
| Restart | `docker compose restart` |
| Stop | `docker compose down` |
| Update after `git push` | `git pull && docker compose up -d --build` |

The app listens on `127.0.0.1` only. To open it from another computer, either use an SSH tunnel
(`ssh -L 8020:127.0.0.1:8010 user@server`, then http://127.0.0.1:8020), or change the ports line in
`docker-compose.yml` to `- "8010:8010"` and allow port 8010 in the firewall.

## 3. Run without Docker (Python 3.12)

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
```

Linux:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/uvicorn flats_app.api:app --host 127.0.0.1 --port 8010
```

Open http://127.0.0.1:8010.

## 4. Check

```bash
curl http://127.0.0.1:8010/health          # {"status":"ok",...}
```

Tests:

```bash
.venv/bin/pip install -r requirements-dev.txt      # Windows: .venv\Scripts\python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest                         # Windows: .venv\Scripts\python -m pytest
```


## 5. Weekly pipeline (data → model → dashboard)

`pipeline/` is the data and model pipeline (flats_pipeline). On the server, the Airflow DAG
`hedonic_pipeline` (`dags/`) runs it every **Wednesday 15:00 Moscow**, one Docker step at a time:

```
winner_fetch → sync_raw → ingest_build → dataset → features → hedonic_index → evaluate → export_bundle → publish → deploy
```

Each run is a version named by its date (`RUN_DATE`, Moscow). Everything goes to the pipeline bucket
(`.env.pipeline`, see `.env.pipeline.example`):

| S3 key | What |
|---|---|
| `cian_winner/{old,new}/<export>_<date>.csv.gz` | raw baza-winner exports (input; `new/` gets two files each Wednesday) |
| `flats_pipeline/{bronze,silver,gold}/…` | ingested exports, lots, sold flats |
| `flats_pipeline/runs/<date>/modeling/…` | dataset, features, hedonic outputs of the run |
| `models/<date>/model.cbm` (+ `model_meta.json`, `linear_model.json`, `forward_test.json`) | dated model |
| `bundles/<date>/…` | dashboard bundle; `models/latest.json`, `bundles/latest.json` point to the newest |
| `logs/<date>/pipeline_<day>.log` | pipeline log (also in Airflow task logs and `pipeline_data/logs/`) |

Dashboards: `app_v3` (http://127.0.0.1:8022) always uses the latest model, the newest dated bundle in
`bundles/`, and is redeployed by the DAG. For comparison: `app` (8010) serves the git baseline `data/`,
`app_v2` (8011) is frozen on bundle 2026-11-04 (`APP_BUNDLE_VERSION`).

Server setup (once):

```bash
cd /root/projects/iep_calculator_hedonic
cp .env.pipeline.example .env.pipeline        # fill in the bucket keys
docker compose --profile pipeline build pipeline
ln -s $PWD/dags/hedonic_pipeline.py $PWD/dags/hedonic_common.py /root/airflow/dags/
docker compose up -d app_v3
```

| What | Command |
|---|---|
| One step by hand | `docker compose --profile pipeline run --rm -e RUN_DATE=2026-11-11 pipeline <step>` |
| Steps | `winner-fetch`, `sync-raw`, `run`, `dataset`, `features`, `hedonic-index`, `evaluate`, `export-bundle`, `publish` |
| Whole run now | `airflow dags trigger hedonic_pipeline` (`--conf '{"run_date": "YYYY-MM-DD"}'` to redo a version) |
| Model version on the main dashboard | `curl http://127.0.0.1:8022/health` |
