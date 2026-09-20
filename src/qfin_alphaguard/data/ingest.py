"""Load price data,either from the network or from the synthetic generator."""

import polars as pl
import yfinance as yf

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.synthetic import generate_prices


def load_prices(config: Config) -> pl.DataFrame:
    """Return daily close prices in long format:date,ticker,close."""
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
    close = raw["Close"].reset_index()

    return (
        pl.from_pandas(close)
        .rename({"Date": "date"})
        .unpivot(index="date", variable_name="ticker", value_name="close")
        .with_columns(pl.col("date").cast(pl.Date))
        .drop_nulls()
        .sort(["ticker", "date"])
    )
