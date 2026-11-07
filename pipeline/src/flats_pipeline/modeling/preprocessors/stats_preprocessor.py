from typing import Final, Optional

import attr
import polars as pl

MIN_TOTAL_SQUARE: Final[int] = 15
MIN_KITCHEN_SQUARE: Final[int] = 4
MIN_LIFE_SQUARE: Final[int] = 9


BALCONY_LOGGIA_MAP: Final[dict[Optional[str], Optional[int]]] = {
    "нет": 0,
    "балкон": 1,
    "лоджия": 1,
    "балкон + лоджия": 2,
    "два балкона": 2,
    "две лоджии": 2,
    "балкон + две лоджии": 3,
    "три балкона": 3,
    "три лоджии": 3,
    "два балкона + две лоджии": 4,
    "четыре лоджии": 4,
    None: None,
}

BATHROOM_MAP: Final[dict[Optional[str], Optional[int]]] = {
    "отсутствует": 0,
    "есть санузел": 1,
    "совмещенный": 1,
    "раздельный": 1,
    "два санузла": 2,
    "два совмещенных санузла": 2,
    "два раздельных санузла": 2,
    "три санузла": 3,
    "три совмещенных санузла": 3,
    "три раздельных санузла": 3,
    "четыре санузла": 4,
    "четыре совмещенных санузла": 4,
    None: 1,
}

ELEVATOR_MAP: Final[dict[str, int]] = {
    "лифт пассажирский и лифт грузовой": 2,
    "лифт грузовой": 1,
    "лифт пассажирский": 1,
    "есть лифт": 1,
    "нет лифта": 0,
}

CONDITION_MAP: Final[dict[str, str]] = {
    "евроремонт": "евроремонт",
    "дизайнерский ремонт": "дизайнерский",
    "среднее состояние": "среднее состояние",
    "требуется ремонт": "требуется ремонт",
    "отличное состояние": "отличное состояние",
    "хорошее состояние": "хорошее состояние",
    "требуется капитальный ремонт": "требуется капитальный ремонт",
    "без отделки": "без отделки",
    "первичная отделка": "первичная отделка",
}


@attr.s(auto_attribs=True, slots=True)
class StatsPreprocessor:
    stats: pl.LazyFrame

    def _process_sale_type_name(self):

        self.stats = self.stats.filter(pl.col("sale_type_name").is_not_null())

    def _process_total_room_count(self):

        self.stats = self.stats.with_columns(
            pl.when(pl.col("total_room_count").is_null() & (pl.col("is_studio") == 1))
            .then(0)
            .otherwise(pl.col("total_room_count"))
            .alias("total_room_count")
        ).filter(pl.col("total_room_count").is_not_null())

    def _process_squares(self):

        self.stats = self.stats.filter(
            pl.col("total_square").is_not_null()
            & (pl.col("total_square") >= MIN_TOTAL_SQUARE)
            & (
                pl.col("kitchen_square").is_null()
                | (pl.col("kitchen_square") >= MIN_KITCHEN_SQUARE)
            )
            & (
                pl.col("life_square").is_null()
                | (pl.col("life_square") >= MIN_LIFE_SQUARE)
            )
        )

        ratios = (
            self.stats.filter(
                pl.col("kitchen_square").is_not_null()
                & pl.col("life_square").is_not_null()
            )
            .select(
                (pl.col("kitchen_square") / pl.col("total_square"))
                .mean()
                .alias("k_ratio"),
                (pl.col("life_square") / pl.col("total_square"))
                .mean()
                .alias("l_ratio"),
            )
            .collect(engine="streaming")
        )

        avg_kitchen_to_total = ratios["k_ratio"][0]
        avg_life_total = ratios["l_ratio"][0]

        self.stats = self.stats.with_columns(
            pl.col("kitchen_square").fill_null(
                avg_kitchen_to_total * pl.col("total_square")
            ),
            pl.col("life_square").fill_null(avg_life_total * pl.col("total_square")),
        )

    def _process_balcony_col(self):

        self.stats = self.stats.with_columns(
            pl.col("balcony_type_name")
            .replace_strict(BALCONY_LOGGIA_MAP, default=None, return_dtype=pl.Int8)
            .alias("balcony_count")
        ).drop("balcony_type_name")

    def _process_apartment(self):

        self.stats = self.stats.with_columns(pl.col("is_apartment").fill_null(0))

    def _process_storeys(self):

        self.stats = self.stats.filter(
            pl.col("storey").is_not_null()
            & pl.col("storeys_count").is_not_null()
            & (pl.col("storey") <= pl.col("storeys_count"))
        ).with_columns(
            [
                (pl.col("storey") == 1).cast(pl.Int8).alias("storey_first"),
                (pl.col("storey") == pl.col("storeys_count"))
                .cast(pl.Int8)
                .alias("storey_last"),
                (pl.col("storey") / pl.col("storeys_count")).alias("storey_relative"),
            ]
        )

    def _process_bathroom(self):

        self.stats = self.stats.with_columns(
            pl.col("water_closet_type_name")
            .replace_strict(BATHROOM_MAP, default=None, return_dtype=pl.Int8)
            .alias("bathroom_count")
        )

    def _process_elevator(self):

        self.stats = self.stats.with_columns(
            pl.col("elevator_type_name")
            .replace_strict(ELEVATOR_MAP, default=None, return_dtype=pl.Int8)
            .alias("elevator_count")
        ).with_columns(
            pl.when(
                pl.col("elevator_count").is_null()
                & (pl.col("storeys_count") <= 5)
                & (pl.col("built_year") <= 1980)
            )
            .then(0)
            .otherwise(pl.col("elevator_count"))
            .fill_null(1)
            .alias("elevator_count")
        )

    def _process_condition(self):

        self.stats = self.stats.with_columns(
            pl.col("apartment_condition_type_name")
            .fill_null("среднее состояние")
            .replace(CONDITION_MAP)
            .alias("apartment_condition")
        )

    def _drop_duplicated_notes(self):

        self.stats = self.stats.unique(
            subset=["note"], keep="first", maintain_order=True
        )

    def _drop_share_notes(self):
        self.stats = self.stats.filter(
            ~pl.col("note").str.contains("(?i)доля"),
            ~pl.col("note").str.contains("(?i)долю"),
        )

    def _select_final_columns(self):

        self.stats = self.stats.select(
            [
                "guid",
                "sale_type_name",
                "total_room_count",
                "total_square",
                "kitchen_square",
                "life_square",
                "balcony_count",
                "is_apartment",
                "storeys_count",
                "storey",
                "storey_first",
                "storey_last",
                "storey_relative",
                "bathroom_count",
                "elevator_count",
                "apartment_condition",
                "note",
                "built_year",
                "building_batch_name",
                "location.lat",
                "location.lon",
            ]
        )

    def preprocess(self) -> pl.LazyFrame:

        self._drop_duplicated_notes()
        self._drop_share_notes()
        self._process_total_room_count()
        self._process_sale_type_name()
        self._process_squares()
        self._process_balcony_col()
        self._process_apartment()
        self._process_storeys()
        self._process_bathroom()
        self._process_elevator()
        self._process_condition()
        self._select_final_columns()

        return self.stats
