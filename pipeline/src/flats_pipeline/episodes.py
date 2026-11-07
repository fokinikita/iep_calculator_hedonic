"""Group listings (guids) of the same flat into sale episodes.

The same flat is often listed under several guids: an agency re-posts it with a new object code,
the developer and an agent both list it, or a deal falls through and the flat comes back under a new
guid. Removing one of these guids is not a sale. So:

- Same flat = equal region, street (`geo_cache_street_address_segment`), building
  (`geo_building`, missing = equal), `total_square`, `storey` and `storeys_count`. A listing more than
  MAX_DISTANCE_M away from the first listing with that key is a different flat (the same street
  name in another town). The town field is not used: in Moscow it holds a metro station, and sources
  disagree on which one.
- Also the same flat: the same description text (descriptions are often copy-pasted between sites
  while address fields differ: "29" vs "29с32Д", missing building, 18 vs 20 storeys), with area within
  NOTE_MAX_AREA_DIFF m2 and the same floor. Texts are compared after removing object codes, deposit
  notes and dates (they change between re-posts), by their first NOTE_PREFIX_CHARS characters; short
  texts and texts shared by many listings (developer templates) are not used.
- Checked by hand on random samples: almost all matches are the same flat.
- Listings of one flat form one episode while each next listing starts no later than RELIST_DAYS
  after all earlier ones closed (overlapping listings are duplicates, later ones are relists).
- The episode is sold when all its listings are closed; the sale is the last closure (its price and
  date), and time on sale counts from the first listing of the episode.
- Lots with no area or street can't be matched and form an episode on their own.

Note: the exports only contain closed lots, so a relist that is still on sale is invisible until it
closes. A recent "sale" can therefore turn into a relist in a later weekly run.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from flats_pipeline.config import settings

logger = logging.getLogger(__name__)

EPISODES_FILE = "flat_episodes.parquet"
FLAT_KEY = [
    "geo_cache_region_name",
    "geo_cache_street_address_segment",
    "geo_building",
    "total_square",
    "storey",
    "storeys_count",
]
_STRING_KEYS = {"geo_cache_region_name", "geo_cache_street_address_segment", "geo_building"}
RELIST_DAYS = 30
MAX_DISTANCE_M = 200
MIN_NOTE_CHARS = 100  # shorter descriptions are too generic to match on
NOTE_PREFIX_CHARS = 300
MAX_NOTE_GROUP = 10  # a text shared by more listings is a template, not one flat
NOTE_MAX_AREA_DIFF = 0.5  # 1.0 let developer templates link mirror flats on one floor
# Parts of a description that change between re-posts of the same flat.
_NOTE_NOISE = [
    r"(код объекта|номер в базе|номер лота|арт\.?|артикул|id|объект|лот)\s*[:№#]?\s*[\w-]*\d+",
    r"(внесен[а-я]*|внесли)\s+(аванс|задаток|депозит)[^.!\n]*",
    r"(аванс|задаток)\s+до[^.!\n]*",
    r"\d{1,2}[.]\d{1,2}[.]\d{2,4}",
]
_OPEN_END = date(2999, 1, 1)


def _distance_m(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    """Planar approximation, good enough within Moscow region at these distances."""
    dy = (lat2 - lat1) * 111_320
    dx = (lon2 - lon1) * 111_320 * (lat1.radians().cos())
    return (dx**2 + dy**2).sqrt()


def flat_attributes(offers_dir: Path) -> pl.LazyFrame:
    """guid + matching attributes from the latest offers (missing columns become null)."""
    lf = pl.scan_parquet(offers_dir / "*.parquet")
    present = lf.collect_schema().names()
    cols = FLAT_KEY + ["location.lat", "location.lon"]
    return lf.select(
        "guid",
        *[pl.col(c) if c in present else pl.lit(None, pl.String if c in _STRING_KEYS else pl.Float64).alias(c) for c in cols],
    )


def note_fingerprint(note: pl.Expr) -> pl.Expr:
    """Hash of the normalized start of a description; null when the text is too short to trust."""
    t = note.str.to_lowercase()
    for pattern in _NOTE_NOISE:
        t = t.str.replace_all(pattern, " ")
    t = t.str.replace_all(r"[^0-9a-zа-яё]+", " ").str.strip_chars()
    return pl.when(t.str.len_chars() >= MIN_NOTE_CHARS).then(t.str.slice(0, NOTE_PREFIX_CHARS).hash())


def note_fingerprints(offers_dir: Path) -> pl.DataFrame:
    """guid -> note_fp, one offers part at a time (all description texts at once need GBs)."""
    return pl.concat(
        pl.scan_parquet(part)
        .select("guid", note_fingerprint(pl.col("note")).alias("note_fp"))
        .collect(engine="streaming")  # texts are processed in small batches
        for part in sorted(offers_dir.glob("*.parquet"))
    )


def _merge_flats_by_note(x: pl.DataFrame) -> pl.DataFrame:
    """Merge flat_ids whose listings share a description, with close area and the same floor."""
    c = x.filter(pl.col("note_fp").is_not_null()).select("flat_id", "note_fp", "total_square", "storey")
    small = c.group_by("note_fp").len().filter(pl.col("len").is_between(2, MAX_NOTE_GROUP)).select("note_fp")
    c = c.join(small, on="note_fp")
    edges = (
        c.join(c, on="note_fp", suffix="_b")
        .filter(
            (pl.col("flat_id") < pl.col("flat_id_b"))
            & ((pl.col("total_square") - pl.col("total_square_b")).abs() <= NOTE_MAX_AREA_DIFF)
            & (pl.col("storey") == pl.col("storey_b"))
        )
        .select("flat_id", "flat_id_b")
        .unique()
    )
    # Union-find over the linked flats (tens of thousands); the smallest id names the group.
    parent: dict[int, int] = {}

    def root(a: int) -> int:
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for a, b in edges.iter_rows():
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    if not parent:
        return x.with_columns(pl.lit(False).alias("flat_merged_by_note"))
    mapping = pl.DataFrame(
        {"flat_id": list(parent), "_root": [root(a) for a in parent]},
        schema={"flat_id": pl.UInt64, "_root": pl.UInt64},
    )
    merged = pl.concat([mapping["flat_id"], mapping["_root"]]).unique().implode()
    return (
        x.join(mapping, on="flat_id", how="left")
        .with_columns(pl.col("_root").fill_null(pl.col("flat_id")).alias("flat_id"))
        .with_columns(pl.col("flat_id").is_in(merged).alias("flat_merged_by_note"))
        .drop("_root")
    )


def assign_episodes(lots: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Per guid: flat_id, episode_id and episode fields.

    `lots` needs: guid, first_pub_date, closure_date (null = not closed), lot_status, first_price,
    last_price, min_price, max_price, FLAT_KEY columns, location.lat, location.lon, and optionally
    note_fp (description fingerprint, see note_fingerprint).
    """
    if "note_fp" not in lots.columns:
        lots = lots.with_columns(pl.lit(None, pl.UInt64).alias("note_fp"))
    has_key = pl.all_horizontal(
        pl.col(c).is_not_null() for c in ["total_square", "storey", "storeys_count", "geo_cache_street_address_segment"]
    )
    end = pl.col("closure_date").fill_null(_OPEN_END)
    x = (
        lots.with_columns(
            pl.when(has_key).then(pl.struct(FLAT_KEY).hash()).otherwise(pl.col("guid").hash()).alias("flat_id")
        )
        .sort("flat_id", "first_pub_date", "guid")
        .with_columns(
            _distance_m(
                pl.col("location.lat").first().over("flat_id"),
                pl.col("location.lon").first().over("flat_id"),
                pl.col("location.lat"),
                pl.col("location.lon"),
            ).alias("_dist")
        )
        .with_columns(
            # Far from the first listing with this key: a different flat (same address in another place).
            pl.when((pl.col("_dist") > MAX_DISTANCE_M).fill_null(False))
            .then(pl.struct("flat_id", pl.col("location.lat").round(3), pl.col("location.lon").round(3)).hash())
            .otherwise(pl.col("flat_id"))
            .alias("flat_id")
        )
    )
    x = (
        _merge_flats_by_note(x)
        .sort("flat_id", "first_pub_date", "guid")
        .with_columns(end.cum_max().shift(1).over("flat_id").alias("_prev_end"))
        .with_columns(
            (
                pl.col("_prev_end").is_null()
                | (pl.col("first_pub_date") > pl.col("_prev_end") + timedelta(days=RELIST_DAYS))
            ).alias("_new")
        )
        .with_columns(pl.struct("flat_id", pl.col("_new").cum_sum().over("flat_id")).hash().alias("episode_id"))
        .with_columns(end.alias("_end"))
    )
    by_end = lambda c: pl.col(c).sort_by("_end", "first_pub_date").last()  # noqa: E731  # value of the last-closing listing
    episodes = x.group_by("episode_id").agg(
        pl.len().alias("episode_n_listings"),
        by_end("guid").alias("episode_final_guid"),
        (pl.col("closure_date").is_null().any()).alias("_open"),
        by_end("lot_status").alias("_final_status"),
        pl.col("first_pub_date").min().alias("episode_first_pub_date"),
        pl.col("first_price").sort_by("first_pub_date").first().alias("episode_first_price"),
        pl.col("min_price").min().alias("episode_min_price"),
        pl.col("max_price").max().alias("episode_max_price"),
        pl.col("guid").sort_by("first_pub_date").alias("episode_guids"),
    ).with_columns(
        pl.when(pl.col("_open"))
        .then(pl.lit("on_sale"))
        .when(pl.col("_final_status") == "sold")
        .then(pl.lit("sold"))
        .otherwise(pl.lit("removed_recently"))
        .alias("episode_status")
    ).drop("_open", "_final_status")
    return (
        x.select("guid", "flat_id", "flat_merged_by_note", "episode_id")
        .join(episodes, on="episode_id")
        .with_columns((pl.col("guid") == pl.col("episode_final_guid")).alias("is_episode_final"))
        .with_columns(
            # Keep the guid list only on the final row: it is what the sold dataset needs.
            pl.when(pl.col("is_episode_final")).then(pl.col("episode_guids")).alias("episode_guids")
        )
        .drop("episode_final_guid")
    )


def build_episodes(as_of: date) -> Path:
    lots = (
        pl.scan_parquet(settings.silver_dir / "lots.parquet")
        .select("guid", "first_pub_date", "closure_date", "lot_status", "first_price", "last_price", "min_price", "max_price")
        .join(flat_attributes(settings.silver_dir / "offers_latest"), on="guid", how="left")
        .collect()
        .join(note_fingerprints(settings.silver_dir / "offers_latest"), on="guid", how="left")
    )
    episodes = assign_episodes(lots, as_of)
    out = settings.silver_dir / EPISODES_FILE
    tmp = out.with_suffix(".tmp")
    episodes.write_parquet(tmp, compression="zstd")
    tmp.replace(out)
    n_ep = episodes["episode_id"].n_unique()
    logger.info(
        "Flat episodes written: %s (%d guids -> %d episodes; %d guids in flats merged by description)",
        out, episodes.height, n_ep, episodes["flat_merged_by_note"].sum(),
    )
    return out
