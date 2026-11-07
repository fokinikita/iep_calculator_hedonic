from datetime import date
from typing import Final

TARGET_NAME = "log_price_sq"
VALID_START_DATE: date = date(2025, 10, 1)
TEST_START_DATE: date = date(2025, 12, 1)

CATBOOST_PRICE_PARAMS: Final[dict[str, int | float | str]] = {
    "iterations": 780,
    "learning_rate": 0.176,
    "depth": 5,
    "l2_leaf_reg": 0.024,
    "random_seed": 228,
    "verbose": 25,
    "thread_count": 4,
    "loss_function": "MAE",
}
