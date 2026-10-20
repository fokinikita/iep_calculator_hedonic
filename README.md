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

