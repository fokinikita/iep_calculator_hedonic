import gzip
import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from flats_pipeline import build, episodes, ingest
from flats_pipeline.config import settings
from flats_pipeline.manifest import Manifest

PRICES_HEADER = "guid,start_pub_datetime,end_pub_datetime,price,deal_status_id"
OFFERS_HEADER = (
    "guid,sale_type_name,note,total_square,storey,storeys_count,"
    "geo_cache_region_name,geo_cache_town_name,geo_cache_street_address_segment,geo_building,location.lat,location.lon"
)


def _write_gz(path: Path, lines: list[str], mtime: float) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox_src"
    inbox.mkdir()
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "INBOX_DIRS", [inbox])

    # Week 1 export (newest event 2026-01-05)
    _write_gz(
        inbox / "for-analytics.msk_habit.offer_pub_list(1).csv.gz",
        [
            PRICES_HEADER,
            # A: sold on 01-05 (stays removed in week 2 too)
            "A,2026-01-05,,100,3",
            "A,2025-12-01,2026-01-05,100,1",
            "A,2025-11-01,2025-12-01,110,1",
            # B: removed on 01-05, comes back in week 2
            "B,2026-01-05,,200,3",
            "B,2025-12-15,2026-01-05,200,1",
            # E: only in week 1, removed long ago -> sold
            "E,2025-10-01,,500,3",
            "E,2025-09-01,2025-10-01,500,1",
            # G: closed, but the source never wrote the status-3 row (still "listed" in the export)
            "G,2025-12-10,,700,1",
            "G,2025-11-20,2025-12-10,720,1",
            # H: the same flat as E, re-posted 19 days after E was removed -> one sale (H), not two
            "H,2025-12-01,,520,3",
            "H,2025-10-20,2025-12-01,520,1",
        ],
        mtime=1_768_000_000,
    )
    _write_gz(
        inbox / "for-analytics.msk_habit.offer(1).csv.gz",
        [
            OFFERS_HEADER,
            "A,прямая,old note,30,2,9,R,T,Ленина ул.,1,55.70,37.60",
            "B,прямая,b,40,3,9,R,T,Ленина ул.,1,55.70,37.60",
            "E,прямая,e,50,4,9,R,T,Мира пр-т,5,55.80,37.50",
            "H,прямая,h,50,4,9,R,T,Мира пр-т,5,55.8001,37.5001",
        ],
        mtime=1_768_000_000,
    )
    # Week 2 export, as of 2026-02-01
    _write_gz(
        inbox / "for-analytics.msk_habit.offer_pub_list(2).csv.gz",
        [
            PRICES_HEADER,
            "A,2026-01-05,,100,3",
            "A,2025-12-01,2026-01-05,100,1",
            "A,2025-11-01,2025-12-01,110,1",
            "B,2026-01-20,,190,1",
            "B,2026-01-05,2026-01-20,200,3",
            "B,2025-12-15,2026-01-05,200,1",
            # C: removed 5 days before snapshot -> not sold yet
            "C,2026-01-27,,300,3",
            "C,2026-01-01,2026-01-27,300,1",
            # D: removed, came back, removed again for good
            "D,2026-01-10,,400,3",
            "D,2025-12-20,2026-01-10,390,1",
            "D,2025-12-18,2025-12-20,400,3",
            "D,2025-12-01,2025-12-18,400,1",
            # F: latest date of the snapshot
            "F,2026-02-01,,600,1",
            "G,2025-12-10,,700,1",
            "G,2025-11-20,2025-12-10,720,1",
        ],
        mtime=1_769_900_000,
    )
    _write_gz(
        inbox / "for-analytics.msk_habit.offer(2).csv.gz",
        [
            OFFERS_HEADER,
            "A,прямая,new note,31,2,9,R,T,Ленина ул.,1,55.70,37.60",
            "D,прямая,d,45,1,9,R,T,Ленина ул.,1,55.70,37.60",
            "F,прямая,f,60,9,9,R,T,Ленина ул.,1,55.70,37.60",
            "G,прямая,g,70,5,9,R,T,Ленина ул.,1,55.70,37.60",
        ],
        mtime=1_769_900_000,
    )
    # Exact duplicate download + empty (unfinished) download
    dup = inbox / "for-analytics.msk_habit.offer_pub_list(3).csv.gz"
    dup.write_bytes((inbox / "for-analytics.msk_habit.offer_pub_list(2).csv.gz").read_bytes())
    (inbox / "for-analytics.msk_habit.offer(3).csv.gz").write_bytes(b"")

    manifest = Manifest(settings.manifest_path)
    ingest.ingest_all(manifest)
    build.build_all(manifest)
    return manifest


def test_manifest(pipeline):
    statuses = sorted(r.status for r in pipeline.records)
    assert statuses == ["duplicate", "ok", "ok", "ok", "ok"]
    assert {r.snapshot_date for r in pipeline.ok("prices")} == {"2026-01-05", "2026-02-01"}


def test_lot_statuses(pipeline):
    lots = pl.read_parquet(settings.silver_dir / build.LOTS_FILE)
    status = dict(zip(lots["guid"], lots["lot_status"]))
    assert status == {
        "A": "sold",
        "B": "active",
        "C": "removed_recently",
        "D": "sold",
        "E": "sold",
        "F": "active",
        "G": "sold",
        "H": "sold",
    }
    row = lots.filter(pl.col("guid") == "D").row(0, named=True)
    assert row["removal_date"] == date(2026, 1, 10)
    assert row["n_comebacks"] == 1
    assert row["sell_days_count"] == (date(2026, 1, 10) - date(2025, 12, 1)).days
    assert row["sold_price"] == 400
    assert row["n_price_changes"] == 2
    assert len(row["price_history"]) == 4
    assert row["sold_date_source"] == "status_3"
    assert row["sold_date"] == date(2026, 1, 10)

    # G entered the closed-lots export on 2026-01-05 with no status-3 row: closed between its
    # last listing event (12-10) and that export, dated by the midpoint.
    g = lots.filter(pl.col("guid") == "G").row(0, named=True)
    assert g["sold_date_source"] == "export_window"
    assert (g["sold_date_min"], g["sold_date_max"]) == (date(2025, 12, 10), date(2026, 1, 5))
    assert g["sold_date"] == date(2025, 12, 23)
    assert g["sold_price"] == 700
    # B came back after entering the exports: active, not a missed removal.
    b = lots.filter(pl.col("guid") == "B").row(0, named=True)
    assert b["closed_no_row"] is False


def test_sold_dataset(pipeline):
    sold = pl.read_parquet(build.latest_sold_dir() / "*.parquet")
    # E and H are one flat: only the last listing (H) is a sale
    assert sorted(sold["guid"]) == ["A", "D", "G", "H"]
    a = sold.filter(pl.col("guid") == "A").row(0, named=True)
    assert a["price"] == 100
    assert a["first_price"] == 110
    assert a["note"] == "new note"  # attributes from the newest offers export
    assert a["total_square"] == 31.0
    assert a["sold_confirmed_by_snapshot"] is True  # still removed 27 days later in week 2
    h = sold.filter(pl.col("guid") == "H").row(0, named=True)
    assert h["n_listings"] == 2
    assert h["listing_guids"] == ["E", "H"]
    assert h["first_pub_date"] == date(2025, 9, 1)  # first listing of the flat (E)
    assert h["sell_days_count"] == (date(2025, 12, 1) - date(2025, 9, 1)).days
    assert h["lot_sell_days_count"] == (date(2025, 12, 1) - date(2025, 10, 20)).days
    assert h["first_price"] == 500
    assert h["price"] == 520


def test_rerun_is_idempotent(pipeline):
    before = len(pipeline.records)
    ingest.ingest_all(pipeline)
    assert len(pipeline.records) == before


def test_removal_outages():
    # Steady 100 removals a week, two weeks with almost none, then a catch-up batch of 400.
    weeks = pl.date_range(date(2025, 1, 6), date(2025, 6, 30), "1w", eager=True).to_list()
    n = [100] * len(weeks)
    gap = weeks.index(date(2025, 4, 7))
    n[gap], n[gap + 1], n[gap + 2] = 5, 3, 400
    daily = pl.DataFrame({"day": weeks, "n": n})
    outages = build.find_removal_outages(daily, first_snapshot=date(2025, 4, 1), as_of=date(2025, 7, 9))
    assert outages == [
        {"outage_start": date(2025, 4, 7), "catch_up_start": date(2025, 4, 21), "catch_up_end": date(2025, 4, 27)}
    ]
    # An outage not followed by a spike (the backlog never got recorded) is not a catch-up.
    n[gap + 2] = 100
    daily = pl.DataFrame({"day": weeks, "n": n})
    assert build.find_removal_outages(daily, date(2025, 4, 1), date(2025, 7, 9)) == []


def test_assign_episodes():
    key = {"geo_cache_region_name": "R", "geo_cache_town_name": "T", "total_square": 50.0, "storey": 4, "storeys_count": 9, "geo_cache_street_address_segment": "X", "geo_building": None}
    rows = [
        # guid, first_pub, closure, lat
        ("L1", date(2025, 1, 1), date(2025, 2, 1), 55.0),
        ("L2", date(2025, 1, 15), date(2025, 2, 10), 55.0),  # duplicate, overlaps L1
        ("L3", date(2025, 3, 1), date(2025, 4, 1), 55.0),  # relist 19 days after -> same episode
        ("L4", date(2025, 6, 1), None, 55.0),  # 61 days later -> new episode, still on sale
        ("L5", date(2025, 1, 2), date(2025, 1, 20), 55.1),  # same key, 11 km away -> another flat
    ]
    lots = pl.DataFrame(
        [
            {"guid": g, "first_pub_date": p, "closure_date": c, "lot_status": "sold" if c else "active",
             "first_price": 1.0, "last_price": 1.0, "min_price": 1.0, "max_price": 1.0,
             **key, "location.lat": lat, "location.lon": 37.0}
            for g, p, c, lat in rows
        ]
    )
    ep = episodes.assign_episodes(lots, as_of=date(2025, 7, 1))
    by = {r["guid"]: r for r in ep.iter_rows(named=True)}
    assert by["L1"]["episode_id"] == by["L2"]["episode_id"] == by["L3"]["episode_id"]
    assert len({by[g]["episode_id"] for g in ["L1", "L4", "L5"]}) == 3
    assert sorted(g for g in by if by[g]["is_episode_final"]) == ["L3", "L4", "L5"]
    assert by["L3"]["episode_status"] == "sold"
    assert by["L3"]["episode_first_pub_date"] == date(2025, 1, 1)
    assert by["L3"]["episode_guids"] == ["L1", "L2", "L3"]
    assert by["L4"]["episode_status"] == "on_sale"
    assert not any(r["flat_merged_by_note"] for r in by.values())


def test_note_matching():
    text = "Продается уютная двухкомнатная квартира с ремонтом, окна во двор, рядом метро и парк. " * 2
    fp = pl.select(
        a=episodes.note_fingerprint(pl.lit("Код объекта: 1920767. " + text)),
        b=episodes.note_fingerprint(pl.lit("Код объекта: 2236044. Внесен аванс до 14 августа. " + text.upper())),
        short=episodes.note_fingerprint(pl.lit("Продается квартира")),
    ).row(0, named=True)
    assert fp["a"] == fp["b"]  # object code, deposit note and case are ignored
    assert fp["short"] is None

    base = {"geo_cache_region_name": "R", "geo_cache_town_name": "T", "storeys_count": 9, "geo_building": None,
            "first_price": 1.0, "last_price": 1.0, "min_price": 1.0, "max_price": 1.0, "location.lon": 37.0,
            "location.lat": 55.0, "lot_status": "sold", "note_fp": fp["a"]}
    rows = [
        # guid, street, area, floor, first_pub, closure
        ("M1", "Ленина ул.", 50.0, 4, date(2025, 1, 1), date(2025, 2, 1)),
        ("M2", "ул. Ленина", 50.4, 4, date(2025, 2, 10), date(2025, 3, 1)),  # same text, address written differently
        ("M3", "Ленина ул.", 62.0, 4, date(2025, 2, 12), date(2025, 3, 5)),  # same text, other area -> template
    ]
    lots = pl.DataFrame(
        [{**base, "guid": g, "geo_cache_street_address_segment": st, "total_square": sq, "storey": fl,
          "first_pub_date": p, "closure_date": c} for g, st, sq, fl, p, c in rows]
    )
    by = {r["guid"]: r for r in episodes.assign_episodes(lots, as_of=date(2025, 7, 1)).iter_rows(named=True)}
    assert by["M1"]["episode_id"] == by["M2"]["episode_id"]
    assert by["M1"]["flat_merged_by_note"] and by["M2"]["flat_merged_by_note"]
    assert by["M3"]["episode_id"] != by["M1"]["episode_id"]
    assert not by["M3"]["flat_merged_by_note"]
