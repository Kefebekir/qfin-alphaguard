import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices


def test_columns():
    df = generate_prices(Config())
    assert df.columns == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjustment",
        "split_factor",
        "sp500",
    ]


def test_shape():
    df = generate_prices(Config())
    assert len(df) == df["date"].n_unique() * df["ticker"].n_unique()


def test_positive_prices():
    df = generate_prices(Config())
    lowest = df.select(pl.min_horizontal("open", "high", "low", "close")).to_series()
    assert (lowest > 0).all()


def test_open_and_close_lie_between_low_and_high():
    df = generate_prices(Config())
    outside = df.filter(
        (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.max_horizontal("open", "close"))
    )
    assert outside.is_empty()
