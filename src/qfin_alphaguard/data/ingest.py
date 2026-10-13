"""Load price data, either from EODHD or from the synthetic generator."""

from collections import defaultdict
from datetime import date

import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.eodhd import BARS_SCHEMA, EodhdClient, api_key, eodhd_code
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.data.universe import IndexHistory, Membership, sp500
from qfin_alphaguard.sessions import session_days

# Scaling prices for splits and dividends can leave a close a hair above the
# high: about 1e-16 relative, from floating-point rounding. Gaps up to this
# size are rounding; anything larger is a real data error and is left for
# validation.
ROUNDING_TOLERANCE = 1e-9

# The S&P 500 ETF the backtest report measures the plan against (step 10). A
# UK investor would hold a UCITS ETF of the same index; SPY's history is the
# longest.
BENCHMARKS = ("SPY",)

# A split shows up as a jump in `adjustment` (10x for a 10-for-1 split). Split
# histories are fetched only for codes whose adjustment jumps by more than
# this from one day to the next; a 5-for-4 split is a 1.25 jump, while
# ordinary dividends move it by a few percent at most.
SPLIT_JUMP = 1.15


def load_prices(config: Config) -> pl.DataFrame:
    """Daily bars with prices and volume adjusted for splits and dividends.

    Columns: date, ticker, open, high, low, close, volume, adjustment,
    split_factor, sp500. The price as traded that day is the adjusted price
    divided by `adjustment`; the shares traded are `volume / split_factor`.
    `sp500` says whether the stock was in the S&P 500 that day; the benchmark
    ETFs in BENCHMARKS never are. The synthetic data has no benchmark.
    """
    if config.synthetic:
        return generate_prices(config)
    first = date.fromisoformat(config.start_date)
    last = date.fromisoformat(config.end_date)
    client = EodhdClient(api_key())
    members = download_sp500(client, sp500(), first, last)
    etfs = download_benchmarks(client, BENCHMARKS, first, last)
    return pl.concat([members, etfs]).sort(["ticker", "date"])


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
            with_splits = _with_split_factor(client, candidate, bars[candidate])
            frames.append(with_splits.with_columns(member.alias("sp500")))

    if not frames:
        return pl.DataFrame(
            schema={**BARS_SCHEMA, "split_factor": pl.Float64, "sp500": pl.Boolean}
        )
    prices = keep_sessions(pl.concat(frames))
    return snap_rounding(prices.sort(["ticker", "date"]))


def download_benchmarks(
    client: EodhdClient, codes: tuple[str, ...], first: date, last: date
) -> pl.DataFrame:
    """Daily bars for ETFs such as SPY, in the members' columns, never `sp500`."""
    frames = [
        _with_split_factor(client, code, client.daily_bars(code, first, last))
        for code in codes
    ]
    if not frames:
        return pl.DataFrame(
            schema={**BARS_SCHEMA, "split_factor": pl.Float64, "sp500": pl.Boolean}
        )
    prices = pl.concat(frames).with_columns(pl.lit(False).alias("sp500"))
    return snap_rounding(keep_sessions(prices).sort(["ticker", "date"]))


def keep_sessions(prices: pl.DataFrame) -> pl.DataFrame:
    """Drop bars on days the exchange was closed.

    EODHD has a few: by its data, Nordstrom traded on New Year's Day 2025 and
    on 9 January 2025, when the exchange closed for President Carter's funeral.
    """
    if prices.is_empty():
        return prices
    days = pl.Series(session_days(prices["date"].min(), prices["date"].max()))
    return prices.filter(pl.col("date").is_in(days.implode()))


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


def repair_broken_bars(prices: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Bars whose open or close lies outside low..high, widened to cover them.

    EODHD sometimes sends a bar whose high and low are wrong while its open and
    close are right: LUMN's of 12 October 2026 came as open 5.25, high and low
    5.05, close 5.05, where the SIP minutes say 5.25, 5.28, 4.99 and 5.055
    (decision 0016). The open and close carry the returns and are kept; the
    high and low are moved out to them. Dropping the bar instead would leave
    the stock a day short and out of the plans for a year.

    Returns the bars, repaired, and the broken ones as they came.
    """
    broken = (pl.col("low") > pl.min_horizontal("open", "close")) | (
        pl.col("high") < pl.max_horizontal("open", "close")
    )
    repaired = prices.with_columns(
        pl.max_horizontal("open", "high", "low", "close").alias("high"),
        pl.min_horizontal("open", "high", "low", "close").alias("low"),
    )
    return repaired, prices.filter(broken)


def _with_split_factor(
    client: EodhdClient, code: str, bars: pl.DataFrame
) -> pl.DataFrame:
    """Add `split_factor`: how many of today's shares one share of that day became.

    It is 10 for Netflix before its 10-for-1 split in November 2025 and 1 after,
    so dividing the split-adjusted volume by it gives the shares traded that day.
    """
    bars = bars.sort("date")
    step = pl.col("adjustment") / pl.col("adjustment").shift(1)
    jumps = bars.select(((step > SPLIT_JUMP) | (step < 1 / SPLIT_JUMP)).any()).item()
    factor = pl.lit(1.0)
    if jumps:
        for day, ratio in client.splits(code):
            factor = factor * pl.when(pl.col("date") < day).then(ratio).otherwise(1.0)
    return bars.with_columns(factor.alias("split_factor"))


def _inside(span: Membership) -> pl.Expr:
    inside = pl.col("date") >= span.start
    if span.end is not None:
        inside = inside & (pl.col("date") < span.end)
    return inside
