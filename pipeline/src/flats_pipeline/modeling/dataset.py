"""Training dataset from the sold flats: prepared (StatsPreprocessor) and cleaned (DatasetCleaner).

One row per sold flat, with the columns the feature code expects (top_flip names):
guid, min_start (first listing), max_start_status3 (sale date), price, sell_days_count, the cleaned
flat attributes and `note`. Extra: sold_date_source, n_listings.

Memory: descriptions (`note`) are ~2 GB for all sold flats, so they are never all in RAM:
- StatsPreprocessor dedups on the full text; it gets a light stand-in instead (hash of the text, plus
  a "доля" mark so its share filter drops the same rows). Checked identical to the direct call.
- The output is written piece by piece, one gold part file at a time.
"""

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from flats_pipeline import build
from flats_pipeline.config import settings
from flats_pipeline.modeling.preprocessors.cleaner import DatasetCleaner
from flats_pipeline.modeling.preprocessors.stats_preprocessor import StatsPreprocessor

logger = logging.getLogger(__name__)

MIN_SELL_DAYS = 14  # sold in under 14 days: mostly glitches and duplicates
INT64_COLUMNS = ["total_room_count", "is_apartment", "storeys_count", "storey", "built_year"]
EXTRA_COLUMNS = ["sold_date_source", "n_listings"]


def dataset_path(as_of: date) -> Path:
    return settings.modeling_dir / f"dataset_{as_of.isoformat()}.parquet"


def latest_dataset_path() -> Path:
    files = sorted(settings.modeling_dir.glob("dataset_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No dataset in {settings.modeling_dir}: run `python -m flats_pipeline dataset`")
    return files[-1]


def build_dataset(
    sold_dir: Path | None = None,
    out_path: Path | None = None,
    min_sell_days: int = MIN_SELL_DAYS,
    min_sold_date: date | None = None,
    max_sold_date: date | None = None,
    only_confirmed: bool = False,
) -> Path:
    sold_dir = sold_dir or build.latest_sold_dir()
    parts = sorted(sold_dir.glob("*.parquet"))
    sold = pl.scan_parquet(sold_dir / "*.parquet")
    as_of = sold.select(pl.col("as_of_date").max()).collect().item()
    out_path = out_path or dataset_path(as_of)

    filters = [pl.col("sell_days_count") >= min_sell_days]
    if min_sold_date:
        filters.append(pl.col("sold_date") >= min_sold_date)
    if max_sold_date:
        filters.append(pl.col("sold_date") <= max_sold_date)
    if only_confirmed:
        filters.append(pl.col("sold_confirmed_by_snapshot"))
    sold = sold.filter(filters)

    # 1. Prepare flat attributes with a light stand-in for the description text.
    note = pl.col("note")
    is_share = note.str.contains("(?i)доля") | note.str.contains("(?i)долю")
    stand_in = pl.when(is_share).then(pl.lit("доля:")).otherwise(pl.lit("")) + note.hash().cast(pl.String)
    light = sold.with_columns(pl.when(note.is_null()).then(pl.lit(None, pl.String)).otherwise(stand_in).alias("note"))
    stats_lf = StatsPreprocessor(light).preprocess()
    stats_columns = stats_lf.collect_schema().names()
    stats = stats_lf.drop("note").collect(engine="streaming")

    # 2. Price columns under the names the feature code uses.
    prices = sold.select(
        "guid",
        pl.col("first_pub_date").alias("min_start"),
        pl.col("sold_date").alias("max_start_status3"),
        pl.col("price").cast(pl.Float64),
        pl.col("sell_days_count").cast(pl.Int64),
        *EXTRA_COLUMNS,
    ).collect(engine="streaming")
    columns = prices.columns + [c for c in stats_columns if c != "guid"]

    # 3. Join the real texts back part by part, clean, append to the output file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    writer, n_in, n_out = None, 0, 0
    try:
        for part in parts:
            piece = (
                pl.scan_parquet(part)
                .filter(filters)
                .select("guid", "note")
                .collect()
                .join(prices, on="guid", how="inner")
                .join(stats, on="guid", how="inner")
                .select(columns)
                .with_columns(pl.col(INT64_COLUMNS).cast(pl.Int64))
            )
            n_in += piece.height
            piece = DatasetCleaner(piece).clean()
            if piece.is_empty():
                continue
            table = piece.to_arrow()
            writer = writer or pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=10_000)
            n_out += piece.height
            del piece, table
    finally:
        if writer:
            writer.close()
    tmp_path.replace(out_path)
    logger.info(
        "Dataset written: %s (%d sold flats prepared, %d after cleaning)", out_path, n_in, n_out
    )
    return out_path
