"""Monotone price model rules.

- The model predicts the log of the *total* price; price per m2 = total / area. Then a larger total, kitchen
  or living area can only raise the price (monotone constraints), with the other inputs fixed.
  Price per m2 itself falls with size, so it cannot carry such a constraint.
- Features derived from the areas (ratios, non-living area, area per room) are not used: through them
  the model could make a larger area cheaper and break the constraints.
- More balconies / bathrooms can only raise the price (raw counts instead of counts per room).
- Condition is merged into 4 ordered levels; a worse condition can only lower the price.
"""

from typing import Final

import polars as pl

# Raw condition (after StatsPreprocessor) -> merged group
CONDITION_GROUPS: Final[dict[str, str]] = {
    "без отделки": "требуется ремонт",
    "первичная отделка": "требуется ремонт",
    "требуется капитальный ремонт": "требуется ремонт",
    "требуется ремонт": "требуется ремонт",
    "среднее состояние": "среднее состояние",
    "хорошее состояние": "среднее состояние",
    "отличное состояние": "хорошее состояние",
    "евроремонт": "дизайнерский",
    "дизайнерский": "дизайнерский",
}
# Worse -> better
CONDITION_LEVELS: Final[list[str]] = ["требуется ремонт", "среднее состояние", "хорошее состояние", "дизайнерский"]
CONDITION_DEFAULT: Final[str] = "среднее состояние"

DROPPED_AREA_FEATURES: Final[set[str]] = {
    "nonlife_square",
    "nonlife_non_kitchen_square",
    "total_room_total_square_ratio",
    "kitchen_square_total_square_ratio",
    "life_square_total_square_ratio",
    "balcony_count_total_square_ratio",
    "bathroom_count_total_square_ratio",
    "elevator_count_total_square_ratio",
    "built_year_total_square_ratio",
    "total_room_total_room_count_ratio",
    "kitchen_square_total_room_count_ratio",
    "life_square_total_room_count_ratio",
    # replaced by the raw counts, which carry monotone constraints
    "balcony_count_total_room_count_ratio",
    "bathroom_count_total_room_count_ratio",
}
CONDITION_FEATURE: Final[str] = "condition_level"
MONOTONE_INCREASING: Final[list[str]] = [
    "total_square", "kitchen_square", "life_square", "balcony_count", "bathroom_count", CONDITION_FEATURE,
]
MODEL_TARGET: Final[str] = "log_price"  # log of the total price


def add_condition_level(df: pl.DataFrame) -> pl.DataFrame:
    """apartment_condition -> merged group; condition_level 0 (needs repair) .. 3 (designer)."""
    merged = (
        pl.col("apartment_condition").fill_null(CONDITION_DEFAULT)
        .replace_strict(CONDITION_GROUPS, default=CONDITION_DEFAULT, return_dtype=pl.String)
    )
    return df.with_columns(merged.alias("apartment_condition")).with_columns(
        pl.col("apartment_condition")
        .replace_strict({c: i for i, c in enumerate(CONDITION_LEVELS)}, return_dtype=pl.Int8)
        .alias(CONDITION_FEATURE)
    )


def model_feature_lists(continuous: list[str], categorical: list[str]) -> tuple[list[str], list[str]]:
    continuous = [c for c in continuous if c not in DROPPED_AREA_FEATURES]
    for c in MONOTONE_INCREASING:
        if c not in continuous:
            continuous.append(c)
    categorical = [c for c in categorical if c != "apartment_condition"]
    return continuous, categorical
