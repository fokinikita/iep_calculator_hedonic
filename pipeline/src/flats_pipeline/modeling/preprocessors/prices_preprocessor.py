import attr
import polars as pl


@attr.s(auto_attribs=True, slots=True)
class PricesPreprocessor:
    prices: pl.LazyFrame

    def preprocess(self) -> pl.LazyFrame:

        return self.prices
