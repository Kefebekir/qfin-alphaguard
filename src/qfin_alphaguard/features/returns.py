"""Turn prices into returns, and returns into a matrix for analysis."""

import numpy as np
import polars as pl


def log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Daily log returns in long format: date, ticker, log_return.
    The first day of each ticker has no previous price, so it is dropped."""
    return (
        prices.sort(["ticker", "date"])
        .with_columns(pl.col("close").log().diff().over("ticker").alias("log_return"))
        .drop_nulls("log_return")
        .select(["date", "ticker", "log_return"])
    )


def to_matrix(returns: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Pivot long returns to a days x tickers matrix.
    Only dates where every ticker has a return are kept.
    Returns the matrix and the ticker order of its columns."""
    wide = (
        returns.pivot(on="ticker", index="date", values="log_return")
        .sort("date")
        .drop_nulls()
    )
    tickers = [c for c in wide.columns if c != "date"]
    return wide.select(tickers).to_numpy(), tickers
