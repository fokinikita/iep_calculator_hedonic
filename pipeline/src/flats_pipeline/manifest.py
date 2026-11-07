"""Registry of every raw file the pipeline has seen.

One record per source file. Exact duplicates (same sha256) are recorded but not ingested twice.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl


@dataclass
class FileRecord:
    source_path: str
    size: int
    mtime: float
    sha256: str
    kind: str  # "prices" | "offers"
    status: str  # "ok" | "duplicate" | "error"
    downloaded_at: str = ""  # ISO datetime (file mtime)
    snapshot_date: str = ""  # ISO date; for prices = latest date inside the file
    rows: int = 0
    bronze_path: str = ""
    s3_key: str = ""
    error: str = ""


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[FileRecord] = []
        if path.exists():
            self.records = [FileRecord(**r) for r in json.loads(path.read_text("utf-8"))]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(r) for r in self.records], ensure_ascii=False, indent=1), "utf-8"
        )
        tmp.replace(self.path)

    def is_known(self, path: Path, size: int, mtime: float) -> bool:
        return any(
            r.source_path == str(path) and r.size == size and r.mtime == mtime and r.status != "error"
            for r in self.records
        )

    def find_ok_by_sha(self, sha: str) -> FileRecord | None:
        return next((r for r in self.records if r.sha256 == sha and r.status == "ok"), None)

    def add(self, record: FileRecord) -> None:
        # A retried file replaces its previous error record.
        self.records = [
            r for r in self.records if not (r.source_path == record.source_path and r.status == "error")
        ]
        self.records.append(record)

    def ok(self, kind: str) -> list[FileRecord]:
        return [r for r in self.records if r.kind == kind and r.status == "ok"]

    def files_frame(self, kind: str) -> pl.DataFrame:
        """Ingested files of one kind, ranked oldest -> newest (rank 1 = oldest)."""
        recs = self.ok(kind)
        df = pl.DataFrame(
            {
                "source_sha": [r.sha256 for r in recs],
                "snapshot_date": [r.snapshot_date for r in recs],
                "downloaded_at": [r.downloaded_at for r in recs],
                "bronze_path": [r.bronze_path for r in recs],
            },
            schema={
                "source_sha": pl.String,
                "snapshot_date": pl.String,
                "downloaded_at": pl.String,
                "bronze_path": pl.String,
            },
        ).with_columns(
            pl.col("snapshot_date").str.to_date(),
            pl.col("downloaded_at").str.to_datetime(),
        )
        return df.sort("snapshot_date", "downloaded_at").with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("file_rank")
        )
