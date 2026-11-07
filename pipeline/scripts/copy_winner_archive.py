"""One-off migration: copy s3://<SRC>/cian_winner/ into the pipeline bucket with the same keys.

The two buckets belong to different accounts, so objects are streamed (download -> upload), not
copied server-side. Idempotent: an object already present with the same size is skipped.

Env: SRC_S3_BUCKET SRC_S3_ENDPOINT_URL SRC_S3_ACCESS_KEY SRC_S3_SECRET_KEY  (main bucket)
     PIPELINE_S3_BUCKET PIPELINE_S3_ENDPOINT_URL PIPELINE_S3_ACCESS_KEY PIPELINE_S3_SECRET_KEY
Run: python copy_winner_archive.py [--prefix cian_winner/] [--dry-run]
"""

import argparse
import os
import sys

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


def client(prefix: str):
    e = lambda k: os.environ[f"{prefix}_{k}"].strip()  # noqa: E731
    c = boto3.client(
        "s3", endpoint_url=e("S3_ENDPOINT_URL"), aws_access_key_id=e("S3_ACCESS_KEY"),
        aws_secret_access_key=e("S3_SECRET_KEY"), region_name=os.getenv(f"{prefix}_S3_REGION", "ru-1"),
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "standard"}),
    )
    return c, e("S3_BUCKET")


def listing(c, bucket: str, prefix: str) -> dict[str, int]:
    out = {}
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = o["Size"]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="cian_winner/")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    src, sb = client("SRC")
    dst, db = client("PIPELINE")
    have, want = listing(dst, db, a.prefix), listing(src, sb, a.prefix)
    todo = {k: s for k, s in want.items() if have.get(k) != s}
    print(f"source {len(want)} objects / {sum(want.values()) / 1e9:.2f} GB; to copy {len(todo)} "
          f"({sum(todo.values()) / 1e9:.2f} GB)", flush=True)
    cfg = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=4)
    for i, (key, size) in enumerate(sorted(todo.items()), 1):
        print(f"[{i}/{len(todo)}] {key} ({size / 1e6:.1f} MB)", flush=True)
        if a.dry_run:
            continue
        if key.endswith("/") and size == 0:  # folder marker
            dst.put_object(Bucket=db, Key=key, Body=b"")
            continue
        body = src.get_object(Bucket=sb, Key=key)["Body"]
        dst.upload_fileobj(body, db, key, Config=cfg)
    if a.dry_run:
        return 0
    final = listing(dst, db, a.prefix)
    bad = [k for k, s in want.items() if final.get(k) != s]
    print(f"verify: {len(want) - len(bad)}/{len(want)} objects match by key and size", flush=True)
    for k in bad:
        print("MISMATCH", k, want[k], final.get(k))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
