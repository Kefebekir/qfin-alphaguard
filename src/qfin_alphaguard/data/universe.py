"""S&P 500 membership over time.

A backtest on today's index members leaves out every company that has since
left the index, often because it shrank, failed or was bought, so the past
looks better than it was (survivorship bias). This module says which stocks
were in the S&P 500 on any given day, using only what was known that day.

The history ships with the package in reference/sp500_membership.csv; see
reference/README.md for its source, version and licence.

The trading universe, the stocks the system may hold in a year, is the most
liquid of the members at the start of that year (decision 0005).
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib import resources
from itertools import pairwise

import polars as pl

# A member needs this many days of trading in the previous year to be ranked,
# so a new listing's busy first weeks cannot buy it a place.
MIN_TRADING_DAYS = 126

# Two share classes of one company move as one. On a typical day the returns
# of GOOG and GOOGL differed by 0.06-0.15% in every year since 2015, while the
# closest pairs of different companies among the most traded members (KO and
# PEP, MA and V) differed by 0.30% or more. Of two stocks whose median daily
# return gap is below this, the trading universe keeps the more traded one.
# A median is not thrown off by a few bad prices; a correlation is: one bad
# GOOG close on 28 July 2021 pulled that year's correlation down to 0.978.
SAME_COMPANY_GAP = 0.002


@dataclass(frozen=True)
class Membership:
    """One continuous stretch of a ticker in the index."""

    ticker: str
    start: date  # first day in the index
    end: date | None  # first day out of the index; None while still a member

    def __post_init__(self) -> None:
        if not self.ticker or self.ticker != self.ticker.strip().upper():
            raise ValueError(
                f"ticker must be upper case without spaces, got {self.ticker!r}"
            )
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"{self.ticker}: end {self.end} is not after start {self.start}"
            )

    def contains(self, day: date) -> bool:
        return self.start <= day and (self.end is None or day < self.end)

    def overlaps(self, first: date, last: date) -> bool:
        """In the index on at least one day from `first` to `last`."""
        return self.start <= last and (self.end is None or self.end > first)


@dataclass(frozen=True)
class IndexHistory:
    """Every membership stretch of one index."""

    spans: tuple[Membership, ...]

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("an index history needs at least one membership")
        by_ticker: dict[str, list[Membership]] = defaultdict(list)
        for span in self.spans:
            by_ticker[span.ticker].append(span)
        for ticker, spans in by_ticker.items():
            spans.sort(key=lambda span: span.start)
            # A ticker can leave and come back, or pass to another company
            # (DOW was Dow Chemical until 2017 and Dow Inc. from 2019), but it
            # cannot be in the index twice at the same time.
            for earlier, later in pairwise(spans):
                if earlier.end is None or earlier.end > later.start:
                    raise ValueError(
                        f"{ticker}: memberships starting {earlier.start} "
                        f"and {later.start} overlap"
                    )

    @property
    def as_of(self) -> date:
        """The last change the history knows about; later changes are missing."""
        ends = [span.end for span in self.spans if span.end is not None]
        return max([span.start for span in self.spans] + ends)

    def members_on(self, day: date) -> frozenset[str]:
        """Tickers in the index on `day`. After `as_of`, the members on `as_of`."""
        return frozenset(span.ticker for span in self.spans if span.contains(day))

    def tickers_between(self, first: date, last: date) -> tuple[str, ...]:
        """Tickers in the index on at least one day from `first` to `last`, sorted."""
        if last < first:
            raise ValueError(f"last ({last}) is before first ({first})")
        return tuple(
            sorted({span.ticker for span in self.spans if span.overlaps(first, last)})
        )

    def spans_of(self, ticker: str) -> tuple[Membership, ...]:
        """Every stretch of `ticker` in the index, oldest first."""
        return tuple(
            sorted(
                (span for span in self.spans if span.ticker == ticker),
                key=lambda span: span.start,
            )
        )


@cache
def sp500() -> IndexHistory:
    """The S&P 500 history shipped with the package, read once."""
    path = resources.files("qfin_alphaguard.data").joinpath(
        "reference", "sp500_membership.csv"
    )
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return IndexHistory(
        tuple(
            Membership(
                ticker=row["ticker"],
                start=date.fromisoformat(row["start_date"]),
                end=date.fromisoformat(row["end_date"]) if row["end_date"] else None,
            )
            for row in rows
        )
    )


def dollar_volume() -> pl.Expr:
    """Value traded in a day, at that day's price and share count.

    Adjusted prices and volumes carry the dividends and splits that came later.
    Ranking by them would let events after the ranking date decide the ranking:
    a stock that went on to pay high dividends would look less traded.
    """
    price = pl.col("close") / pl.col("adjustment")
    shares = pl.col("volume") / pl.col("split_factor")
    return price * shares


def trading_universe(prices: pl.DataFrame, year: int, size: int) -> tuple[str, ...]:
    """The `size` most liquid index members on the first trading day of `year`.

    Members are the codes marked `sp500` that day. They are ranked by average
    daily dollar volume over the previous calendar year, and need at least
    MIN_TRADING_DAYS of trading in it. Of two stocks whose daily returns that
    year differed by less than SAME_COMPANY_GAP on a typical day, only the more
    traded is kept. Nothing after the first trading day of `year` is used, so the answer
    does not change when more data arrives.
    """
    in_year = prices.filter(pl.col("date").dt.year() == year)
    if in_year.is_empty():
        raise ValueError(f"no prices in {year}")
    first_day = in_year["date"].min()
    members = in_year.filter((pl.col("date") == first_day) & pl.col("sp500"))
    last_year = prices.filter(
        (pl.col("date").dt.year() == year - 1)
        & pl.col("ticker").is_in(members["ticker"].implode())
    )
    if last_year.is_empty():
        raise ValueError(f"no prices in {year - 1} to rank the members of {year} by")
    ranked = (
        last_year.group_by("ticker")
        .agg(dollar_volume().mean().alias("dollar_volume"), pl.len().alias("days"))
        .filter(pl.col("days") >= MIN_TRADING_DAYS)
        .sort(["dollar_volume", "ticker"], descending=[True, False])
    )
    returns = (
        last_year.sort("date")
        .with_columns(pl.col("close").pct_change().over("ticker").alias("return"))
        .pivot(on="ticker", index="date", values="return")
    )
    chosen: list[str] = []
    for ticker in ranked["ticker"]:
        if len(chosen) == size:
            break
        if all(
            _median_gap(returns, ticker, other) >= SAME_COMPANY_GAP for other in chosen
        ):
            chosen.append(ticker)
    return tuple(chosen)


def _median_gap(returns: pl.DataFrame, a: str, b: str) -> float:
    """Median absolute difference of two stocks' daily returns."""
    pair = returns.select(a, b).drop_nulls()
    if pair.is_empty():
        return math.inf
    return pair.select((pl.col(a) - pl.col(b)).abs().median()).item()
