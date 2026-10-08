"""S&P 500 membership over time.

A backtest on today's index members leaves out every company that has since
left the index, often because it shrank, failed or was bought, so the past
looks better than it was (survivorship bias). This module says which stocks
were in the S&P 500 on any given day, using only what was known that day.

The history ships with the package in reference/sp500_membership.csv; see
reference/README.md for its source, version and licence.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib import resources
from itertools import pairwise


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
