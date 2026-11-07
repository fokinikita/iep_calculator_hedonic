from typing import Callable, List

import attr
import polars as pl


@attr.attrs(slots=True, auto_attribs=True)
class AdditionalBaseFeaturesService:

    features: pl.DataFrame
    additional_base_features_names: List[str] = attr.attrib(init=False, default=None)

    def _get_cols_features(self) -> List[str]:
        return self.features.columns

    def _get_nonlife_squares(self):
        self.features = self.features.with_columns(
            (pl.col("total_square") - pl.col("life_square")).alias("nonlife_square"),
            (
                pl.col("total_square")
                - pl.col("life_square")
                - pl.col("kitchen_square")
            ).alias("nonlife_non_kitchen_square"),
        )

    def _get_total_square_ratios(self):
        self.features = self.features.with_columns(
            (pl.col("total_room_count") / pl.col("total_square")).alias(
                "total_room_total_square_ratio"
            ),
            (pl.col("kitchen_square") / pl.col("total_square")).alias(
                "kitchen_square_total_square_ratio"
            ),
            (pl.col("life_square") / pl.col("total_square")).alias(
                "life_square_total_square_ratio"
            ),
            (pl.col("balcony_count") / pl.col("total_square")).alias(
                "balcony_count_total_square_ratio"
            ),
            (pl.col("bathroom_count") / pl.col("total_square")).alias(
                "bathroom_count_total_square_ratio"
            ),
            (pl.col("elevator_count") / pl.col("total_square")).alias(
                "elevator_count_total_square_ratio"
            ),
            (pl.col("built_year") / pl.col("total_square")).alias(
                "built_year_total_square_ratio"
            ),
        )

    def _get_total_room_count_ratios(self):
        self.features = self.features.with_columns(
            (pl.col("total_square") / pl.col("total_room_count")).alias(
                "total_room_total_room_count_ratio"
            ),
            (pl.col("kitchen_square") / pl.col("total_room_count")).alias(
                "kitchen_square_total_room_count_ratio"
            ),
            (pl.col("life_square") / pl.col("total_room_count")).alias(
                "life_square_total_room_count_ratio"
            ),
            (pl.col("balcony_count") / pl.col("total_room_count")).alias(
                "balcony_count_total_room_count_ratio"
            ),
            (pl.col("bathroom_count") / pl.col("total_room_count")).alias(
                "bathroom_count_total_room_count_ratio"
            ),
            (pl.col("elevator_count") / pl.col("total_room_count")).alias(
                "elevator_count_total_room_count_ratio"
            ),
            (pl.col("built_year") / pl.col("total_room_count")).alias(
                "built_year_total_room_count_ratio"
            ),
        )

    def _get_build_year_additional_features(self):
        self.features = self.features.with_columns(
            pl.col("built_year").log().alias("built_year_log"),
            (pl.col("built_year") ** 0.5).alias("built_year_sqrt"),
            (pl.col("built_year") ** 2).alias("built_year_sqrd"),
        )

    def _get_storey_additional_features(self):
        self.features = self.features.with_columns(
            pl.col("storey").log().alias("storey_log"),
            (pl.col("storey") ** 0.5).alias("storey_sqrt"),
            (pl.col("storey") ** 2).alias("storey_sqrd"),
        )

        self.features = self.features.with_columns(
            pl.col("storeys_count").log().alias("storeys_count_log"),
            (pl.col("storeys_count") ** 0.5).alias("storeys_count_sqrt"),
            (pl.col("storeys_count") ** 2).alias("storeys_count_sqrd"),
        )

        self.features = self.features.with_columns(
            pl.col("storey_relative").log().alias("storey_relative_log"),
            (pl.col("storey_relative") ** 0.5).alias("storey_relative_sqrt"),
            (pl.col("storey_relative") ** 2).alias("storey_relative_sqrd"),
        )

    def calculate_features(self) -> pl.DataFrame:

        feature_pipeline: List[Callable[[], None]] = [
            self._get_nonlife_squares,
            self._get_total_square_ratios,
            self._get_total_room_count_ratios,
            self._get_build_year_additional_features,
            self._get_storey_additional_features,
        ]

        features_before = self._get_cols_features()

        for feature in feature_pipeline:
            feature()

        features_after = self._get_cols_features()

        self.additional_base_features_names = list(
            set(features_after) - set(features_before)
        )

        return self.features
