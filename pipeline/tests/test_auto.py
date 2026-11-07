import datetime as dt

from flats_pipeline.auto import dated_name, file_time


def test_dated_name():
    assert dated_name("for-analytics.msk_habit.offer.csv.gz", dt.date(2026, 11, 11)) == \
        "for-analytics.msk_habit.offer_2026-11-11.csv.gz"


def test_file_time_orders_exports():
    a = file_time("for-analytics.msk_habit.offer_2025-09-10.csv.gz")
    b = file_time("for-analytics.msk_habit.offer_2025-09-10_2.csv.gz")
    c = file_time("for-analytics.msk_habit.offer_pub_list_2025-09-20.csv.gz")
    assert a < b < c and b - a == 3600
    assert file_time("for-analytics.msk_habit.offer(21).csv.gz") is None
