"""Steps of the automatic weekly run (Airflow DAG `hedonic_pipeline`, every Wednesday 15:00 Moscow).

    winner-fetch   baza-winner exports -> s3://<bucket>/cian_winner/new/<name>_<date>.csv.gz
    sync-raw       s3 cian_winner/{old,new}/ -> DATA_DIR/inbox (only missing files)
    run            ingest -> bronze/silver/gold (+ S3)            (existing command)
    dataset, features, hedonic-index, evaluate                    (existing commands)
    export-bundle  web app bundle -> BUNDLES_DIR/<RUN_DATE>/
    publish        modeling outputs, dated model and bundle -> S3

Everything this run produces is stored under its version RUN_DATE (default: today, Moscow):

    s3://<bucket>/cian_winner/{old,new}/...                       raw exports (input)
    s3://<bucket>/flats_pipeline/{bronze,silver,gold}/...         ingest + build outputs
    s3://<bucket>/flats_pipeline/runs/<RUN_DATE>/modeling/...     dataset, features, hedonic outputs
    s3://<bucket>/models/<RUN_DATE>/model.cbm (+ meta, linear model, quality)
    s3://<bucket>/bundles/<RUN_DATE>/...                          web app bundle
    s3://<bucket>/{models,bundles}/latest.json                    pointer to the newest version
    s3://<bucket>/logs/<RUN_DATE>/pipeline_<day>.log              pipeline log
"""

import datetime as dt
import gzip
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

from flats_pipeline.config import settings

logger = logging.getLogger(__name__)

_EXT = ".csv.gz"
_DATE = re.compile(r"_(\d{4}-\d{2}-\d{2})(?:_(\d+))?\.csv\.gz$")


def client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL or None,
        region_name=settings.S3_REGION,
        aws_access_key_id=settings.S3_ACCESS_KEY.get_secret_value(),
        aws_secret_access_key=settings.S3_SECRET_KEY.get_secret_value(),
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "standard"}),
    )


def _listing(c, prefix: str) -> dict[str, int]:
    out = {}
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = o["Size"]
    return out


def dated_name(name: str, day: dt.date) -> str:
    """``x.csv.gz`` -> ``x_2026-10-28.csv.gz`` (same naming as cian_winner_export)."""
    stem = name[: -len(_EXT)] if name.endswith(_EXT) else os.path.splitext(name)[0]
    return f"{stem}_{day.isoformat()}{_EXT}"


def file_time(name: str) -> float | None:
    """Download time of a raw export from its name: the date (noon Moscow) + 1 hour per ``_<n>``
    copy. The ingest orders offers exports by this time (file mtime)."""
    m = _DATE.search(name)
    if not m:
        return None
    t = dt.datetime.fromisoformat(m.group(1)).replace(hour=12, tzinfo=dt.timezone(dt.timedelta(hours=3)))
    return (t + dt.timedelta(hours=int(m.group(2) or 1) - 1)).timestamp()


def _check_gzip(path: Path) -> int:
    """Size of a complete gzip file; raise on an error page or a cut-off download."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size == 0 or f.read(2) != b"\x1f\x8b":
            raise ValueError(f"{path.name}: not a gzip file ({size} bytes)")
    with gzip.open(path, "rb") as g:
        while g.read(1 << 24):
            pass
    return size


def winner_fetch(day: dt.date | None = None) -> dict:
    """Download both baza-winner exports and store dated copies in cian_winner/new/.

    Idempotent: a file already stored for this day is not downloaded again.
    """
    day = day or dt.date.fromisoformat(settings.run_date)
    c = client()
    prefix = settings.WINNER_PREFIX + "new/"
    have = _listing(c, prefix)
    uploaded = {}
    with tempfile.TemporaryDirectory() as tmp:
        for url in settings.WINNER_URLS:
            name = url.rsplit("/", 1)[1]
            key = prefix + dated_name(name, day)
            if key in have:
                logger.info("Already in S3: %s", key)
                continue
            path = Path(tmp) / name
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as resp, path.open("wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            size = _check_gzip(path)
            c.upload_file(str(path), settings.S3_BUCKET_NAME, key)
            logger.info("%s (%.1f MB) -> s3://%s/%s", name, size / 1e6, settings.S3_BUCKET_NAME, key)
            uploaded[key] = size
    return {"day": day.isoformat(), "uploaded": uploaded}


def sync_raw() -> list[Path]:
    """Copy raw exports missing locally from cian_winner/{old,new}/ into DATA_DIR/inbox.

    The file mtime is set from the date in its name, so the ingest sees the true download order.
    """
    c = client()
    inbox = settings.download_dir
    inbox.mkdir(parents=True, exist_ok=True)
    got = []
    for key, size in sorted(_listing(c, settings.WINNER_PREFIX).items()):
        name = key.rsplit("/", 1)[1]
        if not name.endswith(_EXT):
            continue
        path = inbox / name
        if path.exists() and path.stat().st_size == size:
            continue
        part = path.with_suffix(".part")
        c.download_file(settings.S3_BUCKET_NAME, key, str(part))
        part.replace(path)
        if (t := file_time(name)) is not None:
            os.utime(path, (t, t))
        logger.info("Synced %s (%.1f MB)", key, size / 1e6)
        got.append(path)
    logger.info("Raw sync: %d new files, inbox %s", len(got), inbox)
    return got


def bundle_dir() -> Path:
    return settings.BUNDLES_DIR / settings.run_date


def _upload_tree(c, root: Path, prefix: str, only: list[Path] | None = None) -> int:
    files = only if only is not None else sorted(p for p in root.rglob("*") if p.is_file())
    for f in files:
        c.upload_file(str(f), settings.S3_BUCKET_NAME, prefix + f.relative_to(root).as_posix())
    logger.info("Uploaded %d files -> s3://%s/%s", len(files), settings.S3_BUCKET_NAME, prefix)
    return len(files)


def publish() -> dict:
    """Store this run's modeling outputs, the dated model and the bundle in S3."""
    from flats_pipeline.modeling.build_features import features_dir
    from flats_pipeline.modeling.hedonic import hedonic_dir

    run, c = settings.run_date, client()
    modeling = settings.modeling_dir
    datasets = sorted(modeling.glob("dataset_*.parquet"))
    run_prefix = f"{settings.S3_PREFIX.strip('/')}/runs/{run}/modeling/"
    _upload_tree(c, modeling, run_prefix, datasets[-1:])
    _upload_tree(c, modeling, run_prefix, sorted(p for p in features_dir().rglob("*") if p.is_file()))
    _upload_tree(c, modeling, run_prefix, sorted(p for p in hedonic_dir().rglob("*") if p.is_file()))

    hed = hedonic_dir()
    model_files = {
        "model.cbm": hed / "model_index.cbm",
        "linear_model.json": hed / "linear_model.json",
        "model_meta.json": hed / "hedonic_index_meta.json",
        "forward_test.json": hed / "forward_test.json",
    }
    for name, path in model_files.items():
        if path.exists():
            c.upload_file(str(path), settings.S3_BUCKET_NAME, f"models/{run}/{name}")
    bundle = bundle_dir()
    if not (bundle / "manifest.json").exists():
        raise FileNotFoundError(f"No bundle in {bundle}: run export-bundle first")
    _upload_tree(c, bundle, f"bundles/{run}/")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    pointer = json.dumps({"version": run, "exported_at": manifest.get("exported_at"),
                          "last_sale_date": manifest.get("last_sale_date"), "periods": manifest.get("periods"),
                          "quality": manifest.get("quality")}, ensure_ascii=False, indent=1).encode()
    for kind in ("models", "bundles"):
        c.put_object(Bucket=settings.S3_BUCKET_NAME, Key=f"{kind}/latest.json", Body=pointer,
                     ContentType="application/json")
    logger.info("Published version %s (model + bundle + modeling outputs)", run)
    return {"version": run, "bundle": str(bundle)}


def upload_log(log_file: Path) -> None:
    """Pipeline log of the day -> s3 logs/<RUN_DATE>/ (never fails the run)."""
    try:
        if log_file.exists() and settings.S3_BUCKET_NAME and settings.S3_ACCESS_KEY.get_secret_value():
            client().upload_file(str(log_file), settings.S3_BUCKET_NAME, f"logs/{settings.run_date}/{log_file.name}")
    except Exception as e:  # noqa: BLE001
        logger.warning("Log upload failed: %s: %s", type(e).__name__, e)
