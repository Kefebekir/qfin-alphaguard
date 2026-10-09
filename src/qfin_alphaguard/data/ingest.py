"""Load price data, either from EODHD or from the synthetic generator."""

from collections import defaultdict
from datetime import date

import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.eodhd import BARS_SCHEMA, EodhdClient, api_key, eodhd_code
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.data.universe import IndexHistory, Membership, sp500

# Scaling prices for splits and dividends can leave a close a hair above the
# high: about 1e-16 relative, from floating-point rounding. Gaps up to this
# size are rounding; anything larger is a real data error and is left for
# validation.
ROUNDING_TOLERANCE = 1e-9


def load_prices(config: Config) -> pl.DataFrame:
    """Daily bars: date, ticker, open, high, low, close, volume, adjustment, sp500.

    Prices are adjusted for splits and dividends; the price as traded is the
    adjusted price divided by `adjustment`. `sp500` says whether the stock was
    in the S&P 500 that day.
    """
    if config.synthetic:
        return generate_prices(config)
    first = date.fromisoformat(config.start_date)
    last = date.fromisoformat(config.end_date)
    return download_sp500(EodhdClient(api_key()), sp500(), first, last)


def download_sp500(
    client: EodhdClient, history: IndexHistory, first: date, last: date
) -> pl.DataFrame:
    """Daily bars for every stock in the index on any day from `first` to `last`.

    One ticker can stand for different companies over time, and EODHD keeps
    the earlier ones as TICKER_old, TICKER_old1 and so on. For each stretch of
    membership, the code with the most bars inside the stretch is taken, and
    its bars on those days are marked `sp500`.
    """
    old_codes = client.old_codes()
    frames = []
    for ticker in history.tickers_between(first, last):
        code = eodhd_code(ticker)
        bars: dict[str, pl.DataFrame] = {}  # downloaded once per code
        stretches: dict[str, list[Membership]] = defaultdict(list)
        for span in history.spans_of(ticker):
            if not span.overlaps(first, last):
                continue
            best, most = None, 0
            for candidate in (code, *old_codes.get(code, ())):
                if candidate not in bars:
                    bars[candidate] = client.daily_bars(candidate, first, last)
                inside = bars[candidate].filter(_inside(span)).height
                if inside > most:
                    best, most = candidate, inside
            if best is not None:
                stretches[best].append(span)
        for candidate, spans in stretches.items():
            member = pl.any_horizontal([_inside(span) for span in spans])
            frames.append(bars[candidate].with_columns(member.alias("sp500")))

    if not frames:
        return pl.DataFrame(schema={**BARS_SCHEMA, "sp500": pl.Boolean})
    return snap_rounding(pl.concat(frames).sort(["ticker", "date"]))


def sp500_coverage(prices: pl.DataFrame, history: IndexHistory) -> float:
    """Share of index member-days, on the dates in `prices`, that have a bar.

    A drop shows missing data, or index changes after the membership history's
    `as_of` date that the file does not know about yet.
    """
    days = prices["date"].unique()
    expected = sum(len(history.members_on(day)) for day in days)
    return prices.filter(pl.col("sp500")).height / expected if expected else 0.0


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


def _inside(span: Membership) -> pl.Expr:
    inside = pl.col("date") >= span.start
    if span.end is not None:
        inside = inside & (pl.col("date") < span.end)
    return inside
