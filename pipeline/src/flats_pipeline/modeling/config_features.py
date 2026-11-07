from typing import Final

BASE_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "sale_type_name",
    "total_room_count",
    "is_apartment",
    "apartment_condition",
    "building_batch_name",
)

BASE_CONTINOUS_FEATURES: Final[tuple[str, ...]] = (
    "total_square",
    "kitchen_square",
    "life_square",
    "balcony_count",
    "storeys_count",
    "storey",
    "storey_first",
    "storey_last",
    "storey_relative",
    "bathroom_count",
    "elevator_count",
    "built_year",
)

BASE_TEXT_FEATURES: Final[tuple[str, ...]] = ("_text",)

GEO_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "region_type",
    "city_name",
    "region_name",
    "metro_name",
    "district_name",
    "neighbors_median_total_rooms",
)
