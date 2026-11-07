"""Full recalculation from bronze: price history -> lot status -> sold flats dataset.

Rules
- Every weekly prices export holds the *full* history of each guid it contains, so for each guid
  the history is taken from the newest export that contains it ("newest wins").
- Status 1 = listed (price may change), status 3 = removed from the site (may come back).
- The exports are not snapshots of all listings: they hold the lots *closed* in the last ~90 days
  (a guid enters the first export after its closure and stays ~13 weeks). So the first export of
  a guid's latest run of consecutive exports dates its closure to within one export interval.
- A lot is closed when its last event is a removal (status 3), or when it entered the exports
  while its last event is an open listing (the source missed the status-3 row: it did so for
  ~138k lots in Jan-May 2026). A lot whose last event is a listing *after* it entered the exports
  came back and is active.
- A closed lot is sold when its latest possible closure date is more than SOLD_AFTER_DAYS days
  before the newest snapshot.
- Sale date (`sold_date`, with bounds `sold_date_min`..`sold_date_max`, source in `sold_date_source`):
  - `status_3`: the removal date, when the source recorded it normally;
  - `export_window`: the status-3 row is missing, so the closure is between the previous export
    (or the last listing event, if later) and the first export with the guid; midpoint;
  - `catch_up`: the removal was recorded in a catch-up batch right after an outage of the source's
    removal detection (weeks with almost no removals). The true date is between the outage start
    (or the last listing event, if later) and the batch date; midpoint.
- Flat attributes (offers) also come from the newest export containing the guid.
- Listings (guids) of the same flat are grouped into sale episodes (see episodes.py): the sold
  dataset has one row per sold episode, so re-posts and duplicates are not counted as sales.
"""

import logging
import shutil
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from flats_pipeline import episodes
from flats_pipeline.config import settings
from flats_pipeline.manifest import Manifest
from flats_pipeline.schemas import STATUS_ACTIVE, STATUS_REMOVED

logger = logging.getLogger(__name__)

HISTORY_FILE = "price_history.parquet"
PRESENCE_FILE = "export_presence.parquet"
LOTS_FILE = "lots.parquet"
OFFERS_DIR = "offers_latest"  # folder of parquet parts, one per source export
# Small row groups keep streaming reads of the wide offers tables cheap.
ROW_GROUP_SIZE = 10_000
# Removal-detection outage: a week with fewer removals than this share of the median week.
GAP_WEEK_SHARE = 0.2
# The week right after an outage is a catch-up batch when it has this many times the median week.
CATCH_UP_WEEK_FACTOR = 1.5
# A closed lot stays in the weekly exports for ~90 days after closure.
EXPORT_WINDOW_DAYS = 90


def _sink(lf: pl.LazyFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    lf.sink_parquet(tmp, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    tmp.replace(out)
    return out


def _newest_first_paths(manifest: Manifest, kind: str) -> list[str]:
    files = manifest.files_frame(kind)
    if files.is_empty():
        raise RuntimeError(f"No ingested {kind} files, nothing to build")
    return files.sort("file_rank", descending=True)["bronze_path"].to_list()


def _newest_per_guid(
    paths_newest_first: list[str],
    out: Path,
    prepare: Callable[[pl.LazyFrame], pl.LazyFrame],
    merge: bool = True,
) -> int:
    """Keep, for every guid, only the rows from the newest file that contains it.

    Files are processed one at a time (newest -> oldest); a file contributes only guids not
    already taken from a newer file. Memory ~ one file + the set of seen guids.
    With merge=False the result stays a folder of parts (`out` is that folder): polars reads one
    huge wide parquet file with a lot of memory, many small ones are cheap.
    Returns the number of unique guids written.
    """
    parts_dir = out.parent / f"{out.stem}_parts" if merge else out
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    seen = pl.DataFrame(schema={"guid": pl.String})
    for i, p in enumerate(paths_newest_first):
        part = parts_dir / f"part_{i:04d}.parquet"
        prepare(pl.scan_parquet(p)).join(seen.lazy(), on="guid", how="anti").sink_parquet(
            part, compression="zstd", row_group_size=ROW_GROUP_SIZE
        )
        new_guids = pl.read_parquet(part, columns=["guid"]).unique()
        if new_guids.is_empty():
            part.unlink()
        seen = pl.concat([seen, new_guids])
    if merge:
        _sink(pl.scan_parquet(parts_dir / "*.parquet"), out)
        shutil.rmtree(parts_dir, ignore_errors=True)
    return seen.height


def build_price_history(manifest: Manifest) -> Path:
    def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.select(
                "guid",
                "start_pub_datetime",
                "end_pub_datetime",
                "price",
                "deal_status_id",
                "snapshot_date",
            )
            .unique()
            .sort("guid", "start_pub_datetime", "end_pub_datetime", nulls_last=True)
            .with_columns(pl.int_range(pl.len(), dtype=pl.Int16).over("guid").alias("event_seq"))
        )

    out = settings.silver_dir / HISTORY_FILE
    n = _newest_per_guid(_newest_first_paths(manifest, "prices"), out, prepare)
    logger.info("Price history written: %s (%d guids)", out, n)
    return out


def build_export_presence(manifest: Manifest) -> Path:
    """Per guid: the first export of its latest run of consecutive exports, and the export before it.

    Exports hold recently closed lots, so `first_seen` is the first export after the closure and
    `prev_snapshot` the last export before it. Files are read one at a time, oldest first; only a
    small per-guid state is kept.
    """
    files = manifest.files_frame("prices")
    snapshots = files["snapshot_date"].unique().sort().to_list()
    k_of = {d: k for k, d in enumerate(snapshots)}
    state = pl.DataFrame(
        schema={"guid": pl.String, "run_start_k": pl.Int32, "last_k": pl.Int32, "n_export_runs": pl.Int32}
    )
    for snapshot in snapshots:
        k = k_of[snapshot]
        paths = files.filter(pl.col("snapshot_date") == snapshot)["bronze_path"].to_list()
        current = pl.scan_parquet(paths).select("guid").unique().collect()
        updated = (
            current.join(state, on="guid", how="left")
            .with_columns(continues=(pl.col("last_k") == k - 1).fill_null(False))
            .select(
                "guid",
                pl.when(pl.col("continues")).then(pl.col("run_start_k")).otherwise(pl.lit(k, pl.Int32)).alias("run_start_k"),
                pl.lit(k, pl.Int32).alias("last_k"),
                (pl.col("n_export_runs").fill_null(0) + (~pl.col("continues")).cast(pl.Int32)).alias("n_export_runs"),
            )
        )
        state = pl.concat([state.join(current, on="guid", how="anti"), updated])
    dates = pl.DataFrame({"run_start_k": range(len(snapshots)), "first_seen": snapshots},
                         schema={"run_start_k": pl.Int32, "first_seen": pl.Date})
    dates = dates.with_columns(pl.col("first_seen").shift(1).alias("prev_snapshot"))
    presence = state.join(dates, on="run_start_k", how="left").select(
        "guid", "first_seen", "prev_snapshot", "n_export_runs"
    )
    out = settings.silver_dir / PRESENCE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    presence.write_parquet(out, compression="zstd")
    logger.info("Export presence written: %s (%d guids, %d snapshots)", out, presence.height, len(snapshots))
    return out


def find_removal_outages(daily_removals: pl.DataFrame, first_snapshot: date, as_of: date) -> list[dict]:
    """Find outages of the source's removal detection and the catch-up batch after each.

    `daily_removals`: columns day (Date), n (removal events that day).
    An outage is a run of weeks with fewer than GAP_WEEK_SHARE x the median weekly removals,
    while listings kept being recorded. The week right after it is a catch-up batch when it has more
    than CATCH_UP_WEEK_FACTOR x the median. Only weeks the exports cover are considered
    (from ~12 weeks before the first export up to the last full week before `as_of`).
    Returns [{outage_start, catch_up_start, catch_up_end}] for outages followed by a catch-up batch.
    """
    start = first_snapshot - timedelta(weeks=12)
    start -= timedelta(days=start.weekday())
    end = as_of - timedelta(days=as_of.weekday())  # Monday of the (partial) last week, excluded
    if end <= start:
        return []
    weeks = pl.DataFrame({"week": pl.date_range(start, end - timedelta(days=7), "1w", eager=True)})
    weekly = weeks.join(
        daily_removals.group_by(pl.col("day").dt.truncate("1w").alias("week")).agg(pl.col("n").sum()),
        on="week",
        how="left",
    ).with_columns(pl.col("n").fill_null(0)).sort("week")
    median = weekly["n"].median() or 0
    if median == 0:
        return []
    is_gap = (weekly["n"] < GAP_WEEK_SHARE * median).to_list()
    week_list, counts = weekly["week"].to_list(), weekly["n"].to_list()
    outages, i = [], 0
    while i < len(week_list):
        if not is_gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(week_list) and is_gap[j + 1]:
            j += 1
        if j + 1 < len(week_list) and counts[j + 1] > CATCH_UP_WEEK_FACTOR * median:
            catch_up_start = week_list[j + 1]
            outages.append(
                {
                    "outage_start": week_list[i],
                    "catch_up_start": catch_up_start,
                    "catch_up_end": catch_up_start + timedelta(days=6),
                }
            )
        i = j + 1
    return outages


def _midpoint(lo: pl.Expr, hi: pl.Expr) -> pl.Expr:
    return lo + pl.duration(days=(hi - lo).dt.total_days() // 2)


def build_lots(as_of: date, first_snapshot: date) -> Path:
    """One row per guid with its final state and price-history summary."""
    days = settings.SOLD_AFTER_DAYS
    h = pl.scan_parquet(settings.silver_dir / HISTORY_FILE)

    daily = (
        h.filter(pl.col("deal_status_id") == STATUS_REMOVED)
        .group_by(pl.col("start_pub_datetime").alias("day"))
        .agg(pl.len().alias("n"))
        .collect()
    )
    outages = find_removal_outages(daily, first_snapshot, as_of)
    for o in outages:
        logger.info(
            "Removal-detection outage from %s, catch-up batch %s..%s",
            o["outage_start"], o["catch_up_start"], o["catch_up_end"],
        )

    def ordered(col: str | pl.Expr) -> pl.Expr:
        c = pl.col(col) if isinstance(col, str) else col
        return c.sort_by("event_seq")

    is_active = pl.col("deal_status_id") == STATUS_ACTIVE
    active_days = (
        (pl.col("end_pub_datetime").fill_null(as_of) - pl.col("start_pub_datetime")).dt.total_days()
    )

    lots = (
        h.group_by("guid")
        .agg(
            pl.col("start_pub_datetime").min().alias("first_pub_date"),
            ordered("start_pub_datetime").last().alias("last_event_date"),
            ordered("start_pub_datetime").filter(ordered(is_active)).last().alias("last_listing_date"),
            ordered("deal_status_id").last().alias("last_status"),
            ordered("price").first().alias("first_price"),
            ordered("price").last().alias("last_price"),
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
            (ordered("price").diff().fill_null(0) != 0).sum().alias("n_price_changes"),
            (pl.col("deal_status_id") == STATUS_REMOVED).sum().alias("n_removals"),
            active_days.filter(is_active).sum().alias("active_days"),
            pl.col("snapshot_date").max().alias("last_seen_snapshot"),
            ordered(
                pl.struct("start_pub_datetime", "end_pub_datetime", "price", "deal_status_id")
            ).alias("price_history"),
        )
        .join(pl.scan_parquet(settings.silver_dir / PRESENCE_FILE), on="guid", how="left")
    )

    removed = pl.col("last_status") == STATUS_REMOVED
    # Listed, but already in the closed-lots export with no later event: the status-3 row is missing.
    closed_no_row = (pl.col("last_status") == STATUS_ACTIVE) & (pl.col("last_event_date") < pl.col("first_seen"))
    in_catch_up = pl.lit(False)
    outage_start = pl.lit(None, pl.Date)
    for o in outages:
        hit = pl.col("removal_date").is_between(pl.lit(o["catch_up_start"]), pl.lit(o["catch_up_end"]))
        in_catch_up = in_catch_up | hit
        outage_start = pl.when(hit).then(pl.lit(o["outage_start"])).otherwise(outage_start)
    listing_or = lambda d: pl.max_horizontal(d, pl.col("last_listing_date"))  # noqa: E731

    lots = (
        lots.with_columns(
            pl.when(removed).then(pl.col("last_event_date")).alias("removal_date"),
            closed_no_row.fill_null(False).alias("closed_no_row"),
        )
        .with_columns(
            pl.when(removed & in_catch_up)
            .then(pl.lit("catch_up"))
            .when(removed)
            .then(pl.lit("status_3"))
            .when(pl.col("closed_no_row"))
            .then(pl.lit("export_window"))
            .alias("sold_date_source"),
            outage_start.alias("_outage_start"),
        )
        .with_columns(
            pl.when(pl.col("sold_date_source") == "status_3")
            .then(pl.col("removal_date"))
            .when(pl.col("sold_date_source") == "catch_up")
            .then(pl.min_horizontal(listing_or(pl.col("_outage_start")), pl.col("removal_date")))
            .when(pl.col("sold_date_source") == "export_window")
            .then(
                pl.min_horizontal(
                    # In the first export there is no previous one: exports cover ~90 days back.
                    listing_or(pl.col("prev_snapshot").fill_null(pl.col("first_seen") - pl.duration(days=EXPORT_WINDOW_DAYS))),
                    pl.col("first_seen"),
                )
            )
            .alias("sold_date_min"),
            pl.when(pl.col("sold_date_source") == "export_window")
            .then(pl.col("first_seen"))
            .otherwise(pl.col("removal_date"))
            .alias("sold_date_max"),
        )
        .with_columns(_midpoint(pl.col("sold_date_min"), pl.col("sold_date_max")).alias("closure_date"))
        .with_columns(
            (pl.lit(as_of) - pl.col("sold_date_max")).dt.total_days().alias("days_since_removal"),
            (pl.col("closure_date") - pl.col("first_pub_date")).dt.total_days().alias("sell_days_count"),
            # Still closed in a snapshot taken more than `days` after the closure.
            ((pl.col("last_seen_snapshot") - pl.col("sold_date_max")).dt.total_days() > days)
            .fill_null(False)
            .alias("sold_confirmed_by_snapshot"),
        )
        .with_columns(
            pl.when(pl.col("closure_date").is_not_null() & (pl.col("days_since_removal") > days))
            .then(pl.lit("sold"))
            .when(pl.col("closure_date").is_not_null())
            .then(pl.lit("removed_recently"))
            .when((pl.col("last_status") == STATUS_ACTIVE) & (pl.col("last_seen_snapshot") == as_of))
            .then(pl.lit("active"))
            .when(pl.col("last_status") == STATUS_ACTIVE)
            .then(pl.lit("active_not_in_latest_export"))
            .otherwise(pl.lit("other"))
            .alias("lot_status"),
            (pl.col("n_removals") - removed.cast(pl.UInt32)).alias("n_comebacks"),
            pl.lit(as_of).alias("as_of_date"),
        )
        .with_columns(
            pl.when(pl.col("lot_status") == "sold").then(pl.col("last_price")).alias("sold_price"),
            pl.when(pl.col("lot_status") == "sold").then(pl.col("closure_date")).alias("sold_date"),
        )
        .drop("_outage_start")
    )
    out = _sink(lots, settings.silver_dir / LOTS_FILE)
    logger.info("Lots written: %s", out)
    return out


def build_offers_latest(manifest: Manifest) -> Path:
    """Newest attributes per guid.

    Offers files are wide (long description texts), so they are processed one at a time from
    newest to oldest, keeping only guids not seen in a newer file. Memory ~ one file.
    """
    paths = _newest_first_paths(manifest, "offers")

    # Union of columns over all files (older exports may lack some), newest file's dtypes win.
    schema: dict[str, pl.DataType] = {}
    for p in paths:
        for name, dtype in pl.scan_parquet(p).collect_schema().items():
            schema.setdefault(name, dtype)
    schema.pop("source_sha", None)

    def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
        present = lf.collect_schema().names()
        return (
            lf.select(
                [
                    pl.col(c) if c in present else pl.lit(None, dtype=t).alias(c)
                    for c, t in schema.items()
                ]
            )
            .rename({"snapshot_date": "offer_snapshot_date"})
            .filter(pl.col("guid").is_first_distinct())
        )

    out = settings.silver_dir / OFFERS_DIR
    n = _newest_per_guid(paths, out, prepare, merge=False)
    logger.info("Latest offers written: %s (%d guids)", out, n)
    return out


def build_sold(as_of: date) -> Path:
    """One row per sold flat episode: the last-closing listing, with time on sale over the whole episode."""
    ep = pl.scan_parquet(settings.silver_dir / episodes.EPISODES_FILE).filter(
        pl.col("is_episode_final") & (pl.col("episode_status") == "sold")
    )
    lots = pl.scan_parquet(settings.silver_dir / LOTS_FILE)
    sold_lots = (
        lots.join(ep, on="guid", how="inner")
        .select(
            "guid",
            pl.col("sold_price").alias("price"),
            "sold_date",
            "sold_date_min",
            "sold_date_max",
            "sold_date_source",
            pl.col("episode_first_pub_date").alias("first_pub_date"),
            (pl.col("sold_date") - pl.col("episode_first_pub_date")).dt.total_days().alias("sell_days_count"),
            pl.col("first_pub_date").alias("lot_first_pub_date"),
            pl.col("sell_days_count").alias("lot_sell_days_count"),
            pl.col("episode_n_listings").alias("n_listings"),
            pl.col("episode_guids").alias("listing_guids"),
            "flat_id",
            "active_days",
            pl.col("episode_first_price").alias("first_price"),
            pl.col("episode_min_price").alias("min_price"),
            pl.col("episode_max_price").alias("max_price"),
            "n_price_changes",
            "n_comebacks",
            "sold_confirmed_by_snapshot",
            "as_of_date",
            "price_history",
        )
    ).collect()  # narrow table, ~0.2 GB for all sold lots
    lot_cols = sold_lots.columns

    # Join one offers part (one source export) at a time against the small sold-lots table.
    # The result stays a folder of parts: reading one huge parquet with all description texts
    # needs several GB, many small files stream cheaply. Read it with scan_parquet(folder / "*.parquet").
    out_dir = settings.gold_dir / f"sold_flats_{as_of.isoformat()}"
    tmp_dir = out_dir.with_name(out_dir.name + "_tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    for part in sorted((settings.silver_dir / OFFERS_DIR).glob("*.parquet")):
        sold_lots.lazy().join(pl.scan_parquet(part), on="guid", how="inner").select(
            *lot_cols, pl.exclude(lot_cols)
        ).sink_parquet(tmp_dir / part.name, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    logger.info("Sold flats written: %s", out_dir)
    return out_dir


def latest_sold_dir() -> Path:
    dirs = sorted(p for p in settings.gold_dir.glob("sold_flats_*") if p.is_dir() and not p.name.endswith("_tmp"))
    if not dirs:
        raise FileNotFoundError(f"No sold_flats_<date> folder in {settings.gold_dir}")
    return dirs[-1]


def build_all(manifest: Manifest) -> dict[str, Path]:
    snapshots = manifest.files_frame("prices")["snapshot_date"]
    as_of = snapshots.max()
    logger.info("Building as of %s (sold = closed > %d days)", as_of, settings.SOLD_AFTER_DAYS)
    outputs = {
        "price_history": build_price_history(manifest),
        "export_presence": build_export_presence(manifest),
    }
    outputs["lots"] = build_lots(as_of, snapshots.min())
    outputs["offers_latest"] = build_offers_latest(manifest)
    outputs["flat_episodes"] = episodes.build_episodes(as_of)
    outputs["sold_flats"] = build_sold(as_of)
    log_summary()
    return outputs


def log_summary() -> None:
    lots = pl.scan_parquet(settings.silver_dir / LOTS_FILE)
    counts = lots.group_by("lot_status").len().sort("len", descending=True).collect()
    for status, n in counts.iter_rows():
        logger.info("  %-28s %9d", status, n)
    by_source = (
        lots.filter(pl.col("lot_status") == "sold").group_by("sold_date_source").len().sort("len", descending=True).collect()
    )
    for source, n in by_source.iter_rows():
        logger.info("  sold, date from %-12s %9d", source, n)
    ep = pl.scan_parquet(settings.silver_dir / episodes.EPISODES_FILE)
    ep_counts = (
        ep.filter("is_episode_final").group_by("episode_status").agg(pl.len(), (pl.col("episode_n_listings") > 1).sum().alias("multi"))
        .sort("len", descending=True).collect()
    )
    for status, n, multi in ep_counts.iter_rows():
        logger.info("  flat episodes %-16s %9d (%d with several listings)", status, n, multi)
    sold = pl.scan_parquet(latest_sold_dir() / "*.parquet").select(pl.len()).collect().item()
    logger.info("  sold flats in dataset        %9d", sold)
