# Start the web app locally on http://127.0.0.1:8010 (needs .venv, see README)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m uvicorn flats_app.api:app --host 127.0.0.1 --port 8010
