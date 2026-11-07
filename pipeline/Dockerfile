# Weekly data + model pipeline (flats_pipeline). Code is mounted from the checkout at run time
# (docker-compose.yml), so `git pull` is enough; rebuild only when requirements.txt changes.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONIOENCODING=utf-8 \
    TZ=Europe/Moscow MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src
COPY data/geo ./data/geo
ENV PYTHONPATH=/app/src DATA_DIR=/data BUNDLES_DIR=/bundles
ENTRYPOINT ["python", "-m", "flats_pipeline"]
