from typing import Callable, List

import attr
import polars as pl


@attr.define(slots=True)
class NLPFeaturesService:
    features: pl.DataFrame
    text_col: str = "note"
    nlp_features_names: List[str] = attr.attrib(init=False, default=None)

    def _get_cols_features(self) -> List[str]:
        return self.features.columns

    # =========================
    # PUBLIC PIPELINE
    # =========================
    def calculate_features(self) -> pl.DataFrame:

        feature_pipeline: List[Callable[[], None]] = [
            self.prepare_text,
            self.add_basic_text_features,
            self.add_legal_features,
            self.add_condition_features,
            self.add_layout_features,
            self.add_house_features,
            self.add_location_features,
            self.add_counts_features,
        ]

        features_before = self._get_cols_features()

        for feature in feature_pipeline:
            feature()

        self.features = self.features.drop(self.text_col)
        features_after = self._get_cols_features()

        self.nlp_features_names = list(set(features_after) - set(features_before))
        return self.features

    # =========================
    # TEXT PREPROCESSING
    # =========================
    def prepare_text(self) -> None:
        self.features = self.features.with_columns(
            pl.col(self.text_col)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all("ё", "е")
            .alias("_text")
        )

    # =========================
    # BASIC FEATURES
    # =========================
    def add_basic_text_features(self) -> None:
        self.features = self.features.with_columns(
            [
                pl.col("_text").str.len_chars().alias("note_len_chars"),
                pl.col("_text").str.split(" ").list.len().alias("note_len_words"),
                pl.col("_text").str.count_matches(r"[.!?]").alias("note_sent_count"),
                pl.col("_text").str.count_matches(r"\d").alias("digit_count"),
                pl.col("_text").str.count_matches("!").alias("exclamation_count"),
                pl.col("_text").str.count_matches(",").alias("comma_count"),
            ]
        )

    # =========================
    # HELPER
    # =========================
    def _has_pattern(self, patterns: List[str]) -> pl.Expr:
        return pl.any_horizontal(
            [pl.col("_text").str.contains(p) for p in patterns]
        ).cast(pl.Int8)

    def _count_pattern(self, patterns: List[str]) -> pl.Expr:
        return sum([pl.col("_text").str.count_matches(p) for p in patterns])

    # =========================
    # LEGAL FEATURES
    # =========================
    def add_legal_features(self) -> None:
        self.features = self.features.with_columns(
            [
                self._has_pattern(
                    [r"более\s*5\s*лет", r"5\s*лет\s*в\s*собственности"]
                ).alias("has_more_5_years"),
                self._has_pattern([r"полная\s*стоимость"]).alias("has_full_price"),
                self._has_pattern([r"альтернатив", r"альтернатива"]).alias(
                    "has_alternative"
                ),
                self._has_pattern([r"свободн\w*\s*продаж"]).alias("has_free_sale"),
                self._has_pattern([r"один\s*собственник"]).alias("has_one_owner"),
                self._has_pattern([r"без\s*обременени"]).alias("has_no_encumbrance"),
                self._has_pattern([r"документы\s*готовы"]).alias("has_documents_ready"),
            ]
        )

    # =========================
    # CONDITION
    # =========================
    def add_condition_features(self) -> None:
        self.features = self.features.with_columns(
            [
                self._has_pattern([r"дизайн", r"дизайнерск"]).alias(
                    "has_design_renovation"
                ),
                self._has_pattern([r"евроремонт"]).alias("has_euro_renovation"),
                self._has_pattern(
                    [r"отличн\w*\s*состояни", r"хорош\w*\s*состояни"]
                ).alias("has_good_condition"),
                self._has_pattern([r"требует\s*ремонта", r"под\s*ремонт"]).alias(
                    "has_need_repair"
                ),
                self._has_pattern([r"мебель"]).alias("has_furnished"),
                self._has_pattern([r"техник"]).alias("has_appliances"),
                self._has_pattern([r"хранени", r"шкаф"]).alias("has_storage"),
                self._has_pattern([r"тепл"]).alias("has_warm"),
                self._has_pattern([r"светл"]).alias("has_bright"),
            ]
        )

    # =========================
    # LAYOUT
    # =========================
    def add_layout_features(self) -> None:
        self.features = self.features.with_columns(
            [
                self._has_pattern([r"изолирован"]).alias("has_isolated_rooms"),
                self._has_pattern([r"кухн\w*[-\s]*гостин"]).alias("has_kitchen_living"),
                self._has_pattern([r"высок\w*\s*потолк"]).alias("has_high_ceilings"),
                self._has_pattern([r"гардероб"]).alias("has_walk_in_closet"),
                self._has_pattern([r"балкон"]).alias("has_balcony"),
                self._has_pattern([r"лоджи"]).alias("has_loggia"),
                self._has_pattern([r"панорам"]).alias("has_panoramic"),
                self._has_pattern([r"видов"]).alias("has_view"),
            ]
        )

    # =========================
    # HOUSE
    # =========================
    def add_house_features(self) -> None:
        self.features = self.features.with_columns(
            [
                self._has_pattern([r"кирпичн"]).alias("has_brick_house"),
                self._has_pattern([r"монолит"]).alias("has_monolith_house"),
                self._has_pattern([r"панель"]).alias("has_panel_house"),
                self._has_pattern([r"сталин"]).alias("has_stalin_house"),
                self._has_pattern([r"новострой"]).alias("has_new_building"),
                self._has_pattern([r"лифт"]).alias("has_elevator"),
                self._has_pattern([r"консьерж"]).alias("has_concierge"),
                self._has_pattern([r"парковк"]).alias("has_parking"),
            ]
        )

    # =========================
    # LOCATION
    # =========================
    def add_location_features(self) -> None:
        self.features = self.features.with_columns(
            [
                self._has_pattern([r"метро"]).alias("has_near_metro"),
                self._has_pattern([r"транспортн"]).alias("has_transport_access"),
                self._has_pattern([r"инфраструктур"]).alias("has_infrastructure"),
                self._has_pattern([r"парк"]).alias("has_park"),
                self._has_pattern([r"школ", r"сад"]).alias("has_school"),
                self._has_pattern([r"сосед"]).alias("has_good_neighbors"),
            ]
        )

        # минуты до метро (regex extract)
        self.features = self.features.with_columns(
            pl.col("_text")
            .str.extract(r"(\d+)\s*мин\w*\s*до\s*метро", 1)
            .cast(pl.Float32)
            .alias("metro_minutes_text")
        )

    # =========================
    # COUNTS / SENTIMENT
    # =========================
    def add_counts_features(self) -> None:
        positive = [r"отличн", r"просторн", r"уютн", r"светл", r"тепл"]
        premium = [r"дизайнерск", r"элит", r"премиальн", r"индивидуальн"]
        negative = [r"требует\s*ремонта", r"под\s*ремонт", r"срочно"]

        self.features = self.features.with_columns(
            [
                self._count_pattern(positive).alias("positive_words_count"),
                self._count_pattern(premium).alias("premium_words_count"),
                self._count_pattern(negative).alias("negative_words_count"),
            ]
        )
