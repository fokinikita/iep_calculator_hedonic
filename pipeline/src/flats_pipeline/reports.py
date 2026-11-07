"""Reusable checks of the built data. Results go to DATA_DIR/reports/.

  python -m flats_pipeline report            sales per month (by sale-date source), outages, statuses
  python -m flats_pipeline review-relists    random flats with several listings -> CSV to check by hand
"""

import logging
from pathlib import Path

import polars as pl

from flats_pipeline import build, episodes
from flats_pipeline.config import settings
from flats_pipeline.schemas import STATUS_REMOVED

logger = logging.getLogger(__name__)


def _reports_dir() -> Path:
    d = settings.DATA_DIR / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def monthly_sales(min_sell_days: int = 0) -> pl.DataFrame:
    """Sold flats per month of sale, split by where the sale date comes from."""
    sold = pl.scan_parquet(build.latest_sold_dir() / "*.parquet").filter(pl.col("sell_days_count") >= min_sell_days)
    return (
        sold.group_by(pl.col("sold_date").dt.truncate("1mo").alias("month"))
        .agg(
            pl.len().alias("sold"),
            *[(pl.col("sold_date_source") == s).sum().alias(s) for s in ["status_3", "export_window", "catch_up"]],
            (pl.col("n_listings") > 1).sum().alias("several_listings"),
            pl.col("sell_days_count").median().alias("median_sell_days"),
            pl.col("price").median().alias("median_price"),
        )
        .sort("month")
        .collect()
    )


def removal_outages() -> list[dict]:
    h = pl.scan_parquet(settings.silver_dir / build.HISTORY_FILE)
    daily = (
        h.filter(pl.col("deal_status_id") == STATUS_REMOVED)
        .group_by(pl.col("start_pub_datetime").alias("day"))
        .agg(pl.len().alias("n"))
        .collect()
    )
    snaps = h.select(pl.col("snapshot_date").min().alias("lo"), pl.col("snapshot_date").max().alias("hi")).collect()
    return build.find_removal_outages(daily, snaps["lo"].item(), snaps["hi"].item())


def sales_report(min_sell_days: int = 14) -> Path:
    lots = pl.scan_parquet(settings.silver_dir / build.LOTS_FILE)
    as_of = lots.select(pl.col("as_of_date").max()).collect().item()
    lot_status = lots.group_by("lot_status").len().sort("len", descending=True).collect()
    ep = pl.scan_parquet(settings.silver_dir / episodes.EPISODES_FILE).filter("is_episode_final")
    ep_status = (
        ep.group_by("episode_status")
        .agg(pl.len().alias("episodes"), (pl.col("episode_n_listings") > 1).sum().alias("with_several_listings"))
        .sort("episodes", descending=True)
        .collect()
    )
    monthly = monthly_sales(min_sell_days)
    outages = removal_outages()

    out = _reports_dir() / f"sales_{as_of}.csv"
    monthly.write_csv(out)
    lines = [
        f"as of {as_of}",
        "",
        "Listings (guids) by status:",
        *[f"  {s:<28} {n:>9,}" for s, n in lot_status.iter_rows()],
        "",
        "Flat episodes (a flat's listings grouped, re-posts and duplicates merged):",
        *[f"  {s:<16} {n:>9,}  ({m:,} with several listings)" for s, n, m in ep_status.iter_rows()],
        "",
        "Outages of the source's removal detection followed by a catch-up batch:",
        *[f"  from {o['outage_start']}, catch-up {o['catch_up_start']}..{o['catch_up_end']}" for o in outages],
        "",
        f"Sold flats per month (sell_days_count >= {min_sell_days}), also in {out}:",
    ]
    with pl.Config(tbl_rows=100, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        lines.append(str(monthly))
    text = "\n".join(lines)
    (out.with_suffix(".txt")).write_text(text, encoding="utf-8")
    print(text)
    return out


def review_relists(n: int = 50, seed: int | None = None, by_note: bool = False) -> Path:
    """Random flats with several listings, one row per listing, to check the matching by hand.

    Columns to look at: area/floor/address are equal by construction, so compare `note` (description),
    `price`, coordinates and dates. Same flat = same layout and floor; often the same object code.
    by_note=True: only flats merged by description text (their address fields differ).
    """
    ep = pl.scan_parquet(settings.silver_dir / episodes.EPISODES_FILE)
    picked = (
        ep.filter(pl.col("is_episode_final") & (pl.col("episode_n_listings") > 1))
        .filter(pl.col("flat_merged_by_note") if by_note else pl.lit(True))
        .select("episode_id")
        .collect()
        .sample(n, seed=seed, shuffle=True)
    )
    rows = (
        ep.join(picked.lazy(), on="episode_id")
        .select("guid", "episode_id", "episode_status", "is_episode_final", "flat_merged_by_note")
        .collect()
    )
    guids = rows["guid"].implode()
    lots = (
        pl.scan_parquet(settings.silver_dir / build.LOTS_FILE)
        .filter(pl.col("guid").is_in(guids))
        .select("guid", "first_pub_date", "closure_date", "sold_date_source", "first_price", "last_price", "lot_status")
        .collect()
    )
    # One offers part at a time: reading all description texts at once needs several GB.
    offers = pl.concat(
        pl.scan_parquet(part)
        .filter(pl.col("guid").is_in(guids))
        .select(
            "guid", *episodes.FLAT_KEY, "total_room_count", "location.lat", "location.lon",
            pl.col("note").str.slice(0, 300).str.replace_all(r"\s+", " "),
        )
        .collect()
        for part in sorted((settings.silver_dir / build.OFFERS_DIR).glob("*.parquet"))
    )
    table = (
        rows.join(lots, on="guid", how="left")
        .join(offers, on="guid", how="left")
        .sort("episode_id", "first_pub_date")
    )
    out = _reports_dir() / ("relist_review_by_note.csv" if by_note else "relist_review.csv")
    table.write_csv(out)
    print(f"{table.height} listings of {picked.height} flats written to {out}")
    return out
