"""Load price data, either from the network or from the synthetic generator."""

import polars as pl
import yfinance as yf

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices

# Adjusting for splits and dividends multiplies all four prices by a factor,
# and rounding can leave a close a hair above the high: about 1e-16 relative,
# 22 of 88,740 rows when checked in October 2026. Gaps up to this size are
# rounding; anything larger is a real data error and is left for validation.
ROUNDING_TOLERANCE = 1e-9


def load_prices(config: Config) -> pl.DataFrame:
    """Return daily bars in long format: date, ticker, open, high, low, close, volume.

    Prices are adjusted for splits and dividends.
    """
    if config.synthetic:
        return generate_prices(config)
    return _download_prices(config)


def _download_prices(config: Config) -> pl.DataFrame:
    raw = yf.download(
        list(config.tickers),
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
    )
    # Columns are (field, ticker) pairs; stacking the ticker level gives one
    # row per date and ticker. Days before a stock was listed are all NaN.
    long = raw.stack(level="Ticker").reset_index()

    df = (
        pl.from_pandas(long)
        .rename(str.lower)
        .select("date", "ticker", "open", "high", "low", "close", "volume")
        .drop_nulls()
        .with_columns(pl.col("date").cast(pl.Date), pl.col("volume").cast(pl.Int64))
        .sort(["ticker", "date"])
    )
    return snap_rounding(df)


def snap_rounding(df: pl.DataFrame) -> pl.DataFrame:
    """Move high and low onto open or close where only rounding puts them outside."""
    body_high = pl.max_horizontal("open", "close")
    body_low = pl.min_horizontal("open", "close")
    high_is_rounding = (pl.col("high") < body_high) & (
        body_high - pl.col("high") <= ROUNDING_TOLERANCE * body_high
    )
    low_is_rounding = (pl.col("low") > body_low) & (
        pl.col("low") - body_low <= ROUNDING_TOLERANCE * body_low
    )
    return df.with_columns(
        pl.when(high_is_rounding)
        .then(body_high)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(low_is_rounding).then(body_low).otherwise(pl.col("low")).alias("low"),
    )
