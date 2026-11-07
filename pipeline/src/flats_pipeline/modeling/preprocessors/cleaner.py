"""Extra cleaning after StatsPreprocessor, for price models and the hedonic index.

StatsPreprocessor (copied from top_flip) cleans the flat attributes. On top of it:
- broken encodings ("РїСЂСЏРјР°СЏ РїСЂРѕРґР°Р¶Р°" = "прямая продажа" read in the wrong codepage) are fixed;
- price outliers are dropped: price per m2 outside MIN/MAX_PRICE_SQ (typos, shares, parking
  places, prices in thousands), ~0.4% of rows;
- implausible flats are dropped: area > MAX_TOTAL_SQUARE, buildings over MAX_STOREYS;
- a built year before MIN_BUILT_YEAR is set to missing.
"""

from typing import Final

import attr
import polars as pl

MIN_PRICE: Final[int] = 500_000
MIN_PRICE_SQ: Final[int] = 30_000
MAX_PRICE_SQ: Final[int] = 3_000_000
MAX_TOTAL_SQUARE: Final[int] = 500
MAX_STOREYS: Final[int] = 100
MIN_BUILT_YEAR: Final[int] = 1800
TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "sale_type_name",
    "apartment_condition",
    "building_batch_name",
)


def fix_mojibake(value: str | None) -> str | None:
    """'РїСЂСЏРјР°СЏ' -> 'прямая': UTF-8 bytes that were decoded as cp1251."""
    if value is None or not any(ch in value for ch in "РС"):
        return value
    try:
        fixed = value.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return fixed


@attr.s(auto_attribs=True, slots=True)
class DatasetCleaner:
    data: pl.DataFrame

    def _fix_texts(self) -> None:
        for col in TEXT_COLUMNS:
            if col not in self.data.columns:
                continue
            values = self.data[col].unique().to_list()
            mapping = {v: fix_mojibake(v) for v in values if v is not None and fix_mojibake(v) != v}
            if mapping:
                self.data = self.data.with_columns(pl.col(col).replace(mapping))

    def _drop_outliers(self) -> None:
        price_sq = pl.col("price") / pl.col("total_square")
        self.data = self.data.filter(
            (pl.col("price") >= MIN_PRICE)
            & price_sq.is_between(MIN_PRICE_SQ, MAX_PRICE_SQ)
            & (pl.col("total_square") <= MAX_TOTAL_SQUARE)
            & (pl.col("storeys_count") <= MAX_STOREYS)
        )

    def _fix_built_year(self) -> None:
        self.data = self.data.with_columns(
            pl.when(pl.col("built_year") >= MIN_BUILT_YEAR).then(pl.col("built_year")).alias("built_year")
        )

    def clean(self) -> pl.DataFrame:
        self._fix_texts()
        self._drop_outliers()
        self._fix_built_year()
        return self.data
