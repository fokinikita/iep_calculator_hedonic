from typing import Final

import polars as pl

PRICES_HEADER: Final[str] = "guid,start_pub_datetime,end_pub_datetime,price,deal_status_id"

PRICES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "guid": pl.String(),
    "start_pub_datetime": pl.Date(),
    "end_pub_datetime": pl.Date(),
    "price": pl.Float64(),
    "deal_status_id": pl.Int8(),
}

# Offers (flat attributes) are read as strings and these columns are cast afterwards,
# so every bronze file gets the same schema no matter what CSV inference would guess.
OFFERS_INT_COLUMNS: Final[list[str]] = [
    "built_year",
    "offer_part_count",
    "offer_room_count",
    "total_part_count",
    "total_room_count",
    "is_studio",
    "is_apartment",
    "is_free_planning",
    "storey",
    "storeys_count",
    "geo_town_transport_access",
    "geo_cache_building_in_renovation",
    "walking_access_1",
    "transport_access_1",
]
OFFERS_FLOAT_COLUMNS: Final[list[str]] = [
    "total_square",
    "kitchen_square",
    "life_square",
    "price_rub",
    "location.lat",
    "location.lon",
]

STATUS_ACTIVE: Final[int] = 1
STATUS_REMOVED: Final[int] = 3


def detect_kind(header_line: str) -> str | None:
    header = header_line.strip().lstrip("﻿")
    if header == PRICES_HEADER:
        return "prices"
    if header.startswith("guid,sale_type_name"):
        return "offers"
    return None
