"""Raw csv.gz exports -> bronze parquet (one file at a time to keep memory low)."""

import gzip
import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl

from flats_pipeline.config import settings
from flats_pipeline.manifest import FileRecord, Manifest
from flats_pipeline.schemas import (
    OFFERS_FLOAT_COLUMNS,
    OFFERS_INT_COLUMNS,
    PRICES_SCHEMA,
    detect_kind,
)

logger = logging.getLogger(__name__)

ROW_GROUP_SIZE = 10_000  # small row groups = low memory when streaming later
RAW_GLOB = "*msk_habit.offer*.csv.gz"


def discover_raw_files() -> list[Path]:
    dirs = [*settings.INBOX_DIRS, settings.download_dir]
    files = {p.resolve() for d in dirs if d.exists() for p in d.glob(RAW_GLOB)}
    return sorted(files, key=lambda p: p.stat().st_mtime)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.readline()


def _read_prices(path: Path, downloaded_at: datetime) -> tuple[pl.DataFrame, str]:
    df = pl.read_csv(path, schema=PRICES_SCHEMA)
    # The export has no explicit date: the newest event in it is the snapshot date.
    snapshot = df.select(
        pl.max_horizontal(pl.col("start_pub_datetime").max(), pl.col("end_pub_datetime").max())
    ).item()
    if snapshot is None:
        snapshot = downloaded_at.date()
    return df, snapshot.isoformat()


def _read_offers(path: Path, downloaded_at: datetime) -> tuple[pl.LazyFrame, str]:
    """Offers files are big (~1 GB of text): decompress to disk and stream instead of loading."""
    tmp_csv = settings.DATA_DIR / "tmp" / (path.stem + ".csv")
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rb") as src, tmp_csv.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 24)
    lf = pl.scan_csv(tmp_csv, infer_schema=False)
    columns = lf.collect_schema().names()
    lf = lf.with_columns(
        [
            pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int32, strict=False)
            for c in OFFERS_INT_COLUMNS
            if c in columns
        ]
        + [pl.col(c).cast(pl.Float64, strict=False) for c in OFFERS_FLOAT_COLUMNS if c in columns]
    )
    # Offers carry no dates; the download time orders them.
    return lf, downloaded_at.date().isoformat()


def ingest_file(path: Path, manifest: Manifest) -> FileRecord | None:
    stat = path.stat()
    if stat.st_size == 0:
        logger.warning("Skip empty file (download not finished?): %s", path)
        return None
    if manifest.is_known(path, stat.st_size, stat.st_mtime):
        return None

    downloaded_at = datetime.fromtimestamp(stat.st_mtime)
    record = FileRecord(
        source_path=str(path),
        size=stat.st_size,
        mtime=stat.st_mtime,
        sha256="",
        kind="",
        status="error",
        downloaded_at=downloaded_at.isoformat(timespec="seconds"),
    )
    try:
        record.sha256 = _sha256(path)
        kind = detect_kind(_read_header(path))
        if kind is None:
            raise ValueError("unknown header, not a prices/offers export")
        record.kind = kind

        if (dup := manifest.find_ok_by_sha(record.sha256)) is not None:
            record.status = "duplicate"
            record.snapshot_date = dup.snapshot_date
            record.bronze_path = dup.bronze_path
            logger.info("Duplicate of %s: %s", Path(dup.source_path).name, path.name)
            manifest.add(record)
            return record

        logger.info("Ingest %s (%s, %.1f MB)", path.name, kind, stat.st_size / 2**20)
        reader = _read_prices if kind == "prices" else _read_offers
        data, snapshot_date = reader(path, downloaded_at)
        lf = data.lazy().with_columns(
            pl.lit(snapshot_date).str.to_date().alias("snapshot_date"),
            pl.lit(record.sha256).alias("source_sha"),
        )

        out = settings.bronze_dir / kind / f"{kind}_{snapshot_date}_{record.sha256[:10]}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        lf.sink_parquet(tmp, compression="zstd", row_group_size=ROW_GROUP_SIZE)
        tmp.replace(out)

        record.status = "ok"
        record.snapshot_date = snapshot_date
        record.rows = pl.scan_parquet(out).select(pl.len()).collect().item()
        record.bronze_path = str(out)
        if ("pub_list" in path.name) != (kind == "prices"):
            logger.warning("File name does not match its content (%s): %s", kind, path.name)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the weekly run
        record.error = f"{type(e).__name__}: {e}"[:500]
        logger.error("Failed to ingest %s: %s", path, record.error)
    finally:
        shutil.rmtree(settings.DATA_DIR / "tmp", ignore_errors=True)

    manifest.add(record)
    manifest.save()
    return record


def ingest_all(manifest: Manifest) -> list[FileRecord]:
    done = []
    memory_errors = 0
    for path in discover_raw_files():
        rec = ingest_file(path, manifest)
        if rec is not None:
            done.append(rec)
            low_memory = rec.status == "error" and "memory" in rec.error.lower()
            memory_errors = memory_errors + 1 if low_memory else 0
            if memory_errors >= 3:
                # The whole machine is out of memory; stop and let the next run retry.
                logger.error("Out of memory 3 times in a row, stopping ingest (will retry next run)")
                break
    manifest.save()
    new_ok = [r for r in done if r.status == "ok"]
    logger.info(
        "Ingest finished: %d new, %d duplicates, %d errors",
        len(new_ok),
        sum(r.status == "duplicate" for r in done),
        sum(r.status == "error" for r in done),
    )
    return done
