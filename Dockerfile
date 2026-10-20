FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app/src \
    APP_DATA_DIR=/app/data

# catboost needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# The bundle (model + indices + geo) is mounted at /app/data, so a new export needs no rebuild.
RUN useradd --create-home app
USER app
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')" || exit 1
CMD ["uvicorn", "flats_app.api:app", "--host", "0.0.0.0", "--port", "8010"]
