import logging
from pathlib import Path

import boto3
from botocore.config import Config

from flats_pipeline.config import settings
from flats_pipeline.manifest import Manifest

logger = logging.getLogger(__name__)


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        region_name=settings.S3_REGION,
        aws_access_key_id=settings.S3_ACCESS_KEY.get_secret_value(),
        aws_secret_access_key=settings.S3_SECRET_KEY.get_secret_value(),
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5, "mode": "standard"}),
    )


def _key(*parts: str) -> str:
    return "/".join([settings.S3_PREFIX.strip("/"), *parts])


def upload_bronze(manifest: Manifest) -> int:
    """Upload every ingested bronze parquet that is not in S3 yet."""
    pending = [r for r in manifest.records if r.status == "ok" and not r.s3_key]
    if not pending:
        return 0
    client = _client()
    for r in pending:
        path = Path(r.bronze_path)
        key = _key("bronze", r.kind, path.name)
        logger.info("Upload %s -> s3://%s/%s", path.name, settings.S3_BUCKET_NAME, key)
        client.upload_file(str(path), settings.S3_BUCKET_NAME, key)
        r.s3_key = key
        manifest.save()
    return len(pending)


def upload_outputs(paths: list[Path]) -> None:
    """Upload output files or folders of parts, keeping <layer>/<name>[/<part>] as the key."""
    client = _client()
    for path in paths:
        layer = path.parent.name  # silver | gold
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        prefix = [path.name] if path.is_dir() else []
        logger.info("Upload %s (%d files) -> s3://%s/%s", path.name, len(files),
                    settings.S3_BUCKET_NAME, _key(layer, path.name))
        for f in files:
            client.upload_file(str(f), settings.S3_BUCKET_NAME, _key(layer, *prefix, f.name))
