"""CLI: python -m flats_pipeline <command>

  run       download -> ingest -> upload raw parquets to S3 -> rebuild -> upload outputs
  ingest    only convert new raw files to bronze parquet
  upload    only upload bronze parquets missing in S3
  build     only rebuild silver/gold from bronze
  report    sales per month by sale-date source, outages, statuses -> DATA_DIR/reports
  review-relists  random flats with several listings -> DATA_DIR/reports/relist_review.csv
                  (--by-note: only flats merged by description text)

Modelling (DATA_DIR/modeling):
  dataset         sold flats -> prepared + cleaned training dataset
  features        dataset -> features (--variant iep: no text features, default; full: with them)
  hedonic-index   features -> hedonic price index (fixed basket, CatBoost) + Excel/chart
  hedonic-model   features -> CatBoost model with year_month for the API
  model-all       dataset -> features -> hedonic-index
  evaluate        out-of-time test: train before the last 3 months, estimate them
  export-app      model + precomputed indices + geo -> APP_BUNDLE_DIR (the web app bundle; --out PATH)

Automatic weekly run (auto.py, Airflow DAG hedonic_pipeline; version = RUN_DATE, default today):
  winner-fetch    baza-winner exports -> S3 cian_winner/new/<name>_<date>.csv.gz
  sync-raw        S3 cian_winner/{old,new}/ -> DATA_DIR/inbox (missing files only)
  export-bundle   web app bundle -> BUNDLES_DIR/<RUN_DATE>/
  publish         modeling outputs, dated model and bundle -> S3
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from flats_pipeline.config import settings


def _setup_logging() -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = _log_file()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


class _RunLock:
    """Prevents two runs (e.g. scheduled + manual) from writing the same files."""

    def __init__(self) -> None:
        self.path = settings.DATA_DIR / ".lock"

    def __enter__(self):
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            # OS-level lock: released automatically if the process crashes.
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            raise SystemExit("Another pipeline run is in progress")
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def main() -> None:
    parser = argparse.ArgumentParser(prog="flats_pipeline")
    parser.add_argument("command", choices=[
            "run", "ingest", "upload", "build", "report", "review-relists",
            "dataset", "features", "hedonic-index", "hedonic-model", "model-all", "evaluate",
            "export-app", "winner-fetch", "sync-raw", "export-bundle", "publish",
        ], default="run", nargs="?")
    parser.add_argument("--no-s3", action="store_true", help="skip all S3 uploads")
    parser.add_argument("--n", type=int, default=50, help="review-relists: number of flats")
    parser.add_argument("--by-note", action="store_true", help="review-relists: only flats merged by description")
    parser.add_argument("--variant", choices=["iep", "full"], default="iep", help="features: with or without text features")
    parser.add_argument("--start-month", default=None, help="hedonic: first month, e.g. 2025_06 (default: all)")
    parser.add_argument("--out", default=None, help="export-app: bundle folder (default: APP_BUNDLE_DIR = ../../iep_calculator_hedonic/data)")
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("flats_pipeline")
    # Trust the Windows certificate store, so HTTPS works behind antivirus HTTPS scanning (AVG).
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    # Must happen before polars is imported. Fewer threads = smaller buffers.
    os.environ.setdefault("POLARS_MAX_THREADS", str(settings.POLARS_MAX_THREADS))
    from flats_pipeline.memlimit import limit_process_memory, peak_memory_gb

    modelling = args.command in (
        "dataset", "features", "hedonic-index", "hedonic-model", "model-all", "evaluate", "export-app",
        "export-bundle",
    )
    limit_process_memory(settings.MODEL_MAX_MEMORY_GB if modelling else settings.MAX_MEMORY_GB)
    use_s3 = settings.UPLOAD_TO_S3 and not args.no_s3

    # Imported here so `--help` works without polars/boto3 loading.
    from flats_pipeline import auto

    if args.command == "winner-fetch":  # S3 only, no local state: no lock
        try:
            log.info("winner-fetch: %s", auto.winner_fetch())
        finally:
            auto.upload_log(_log_file())
        return

    try:
        with _RunLock():
            _run(args, log, use_s3, modelling)
        log.info("Done: %s (version %s, peak memory %.2f GB)", args.command, settings.run_date, peak_memory_gb())
    except BaseException:
        log.exception("Failed: %s (version %s)", args.command, settings.run_date)
        raise
    finally:
        auto.upload_log(_log_file())


def _log_file():
    return settings.logs_dir / f"pipeline_{datetime.now():%Y-%m-%d}.log"


def _run(args, log, use_s3: bool, modelling: bool) -> None:
    from flats_pipeline import auto, build, download, ingest, reports, s3
    from flats_pipeline.manifest import Manifest

    manifest = Manifest(settings.manifest_path)
    if args.command == "sync-raw":
        auto.sync_raw()
    if args.command == "run":
        download.download_weekly()
    if args.command in ("run", "ingest"):
        ingest.ingest_all(manifest)
    if use_s3 and args.command in ("run", "upload"):
        n = s3.upload_bronze(manifest)
        log.info("Uploaded %d raw parquet files to S3", n)
    if use_s3 and args.command == "upload":
        s3.upload_outputs([settings.silver_dir / build.LOTS_FILE, build.latest_sold_dir()])
    if args.command in ("run", "build"):
        outputs = build.build_all(manifest)
        if use_s3:
            s3.upload_outputs([outputs["lots"], outputs["sold_flats"]])
        reports.sales_report()
    if args.command == "report":
        reports.sales_report()
    if args.command == "review-relists":
        reports.review_relists(args.n, by_note=args.by_note)
    if modelling:
        from flats_pipeline.modeling import build_features, dataset, hedonic

        if args.command in ("dataset", "model-all"):
            dataset.build_dataset()
        if args.command in ("features", "model-all"):
            build_features.build_features(variant=args.variant)
        if args.command in ("hedonic-index", "model-all"):
            index = hedonic.build_index(args.start_month, variant=args.variant)
            hedonic.build_area_indices(variant=args.variant)
            with __import__("polars").Config(tbl_rows=60):
                print(index)
        if args.command == "evaluate":
            print(hedonic.evaluate_forward(variant=args.variant))
        if args.command == "hedonic-model":
            hedonic.train_served_model(args.start_month, variant=args.variant)
        if args.command in ("export-app", "model-all"):
            from pathlib import Path

            from flats_pipeline.modeling.export_app import DEFAULT_OUT, export_app

            export_app(Path(args.out) if args.out else DEFAULT_OUT, variant=args.variant)
        if args.command == "export-bundle":
            from flats_pipeline.modeling.export_app import export_app

            export_app(auto.bundle_dir(), variant=args.variant)
    if args.command == "publish":
        auto.publish()


if __name__ == "__main__":
    main()
