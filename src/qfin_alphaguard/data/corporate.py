"""Splits and dividends, derived from the daily bars.

The daily bars carry two factors. `split_factor` is how many of today's shares
one share of that day became through later splits; `adjustment` is the
adjusted over the traded close, which carries splits and dividends together.
Between two trading days in a row:

- a split shows as a fall in split_factor: the ratio is yesterday's over
  today's (Netflix: 10 on 17 November 2025)
- a dividend shows as a rise in the dividend part of the adjustment,
  adjustment x split_factor, on its ex-date: the rise as a share of the
  factor, times the previous traded close, is the cash per share

On the real data this gives Netflix's 7-for-1 (2015) and 10-for-1 (2025)
splits and Apple's four 2025 dividends (0.25, 0.26, 0.26, 0.26 USD) exactly.
"""

from collections.abc import Iterable

import polars as pl

from qfin_alphaguard.events import Dividend, Split

# EODHD rounds adjusted closes, so the dividend factor jitters by tiny amounts
# almost every day. A real dividend is far larger: Apple's quarterly one is
# 0.11% of its price. Changes below this share of the price are rounding.
MIN_DIVIDEND_SHARE = 1e-4


def corporate_actions(
    daily: pl.DataFrame, tickers: Iterable[str] | None = None
) -> list[Split | Dividend]:
    """The splits and dividends of `tickers` (all, if None), by day then ticker."""
    if tickers is not None:
        daily = daily.filter(pl.col("ticker").is_in(list(tickers)))
    steps = (
        daily.sort("ticker", "date")
        .with_columns(
            (pl.col("adjustment") * pl.col("split_factor")).alias("dividend_factor"),
            (pl.col("close") / pl.col("adjustment")).alias("traded_close"),
        )
        .with_columns(
            (
                pl.col("split_factor").shift(1).over("ticker") / pl.col("split_factor")
            ).alias("ratio"),
            (
                1
                - pl.col("dividend_factor").shift(1).over("ticker")
                / pl.col("dividend_factor")
            ).alias("dividend_share"),
            pl.col("traded_close").shift(1).over("ticker").alias("previous_close"),
        )
        .drop_nulls("ratio")
    )
    odd = steps.filter(pl.col("dividend_share") < -MIN_DIVIDEND_SHARE)
    if not odd.is_empty():
        # A fall in the dividend factor is neither a split nor a dividend: a
        # spin-off or merger EODHD encoded another way. Stop rather than guess.
        cases = ", ".join(
            f"{t} {d}" for t, d in odd.select("ticker", "date").head(5).rows()
        )
        raise ValueError(
            f"{odd.height} adjustments that are neither split nor dividend: {cases}"
        )
    actions: list[Split | Dividend] = [
        Split(ticker, day, ratio)
        for ticker, day, ratio in steps.filter(pl.col("ratio") != 1)
        .select("ticker", "date", "ratio")
        .rows()
    ]
    actions += [
        Dividend(ticker, day, share * previous_close)
        for ticker, day, share, previous_close in steps.filter(
            pl.col("dividend_share") > MIN_DIVIDEND_SHARE
        )
        .select("ticker", "date", "dividend_share", "previous_close")
        .rows()
    ]
    return sorted(
        actions, key=lambda action: (action.day, action.ticker, type(action).__name__)
    )
