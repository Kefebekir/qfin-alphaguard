"""The nightly plan replayed on the stored data: what `qfin backtest` runs.

Everything comes from the stores, never from a fresh download (decision
0015): the daily bars for the plans and the corporate actions, the 1-minute
bars for the trades.

- The days run from the first session with a year of daily returns before
  it, 5 January 2016, to the last with both daily bars and minutes.
- One plan a day, built from the daily bars before it (plan.py). Above a few
  dozen dollars the capital does not change a plan, so runs at several
  capitals share their plans.
- The minutes of each year's trading universe, and of the stocks that left
  it, until the end of January (minute_feed).
- The splits and dividends of the stocks the backtest can hold, on the days
  it can hold them. EODHD's daily bars encode a few events of other stocks
  as neither split nor dividend (NVDA's rounding in 2015, CHTR's merger in
  2016); data/corporate.py stops on those, and they never matter here.
- A Delisting on the first session a stock has no minutes after having had
  them: TWTR after its takeover, VIAC and DWDP after their codes changed, and
  each stock that left the universe, after its January. A stock still held
  then is paid out at its last price; one sold in time is not touched.
- EqualSpaced, the simulated broker at Alpaca's costs, and Guard with the
  limits of guard.yaml.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import polars as pl

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import BacktestResult, run_backtest
from qfin_alphaguard.data.corporate import corporate_actions
from qfin_alphaguard.data.intraday import intraday_path, minute_feed, minute_windows
from qfin_alphaguard.data.universe import TRADING_UNIVERSE_SIZE, trading_universes
from qfin_alphaguard.events import (
    DailyPlan,
    Delisting,
    Dividend,
    ExecutionSettings,
    Split,
)
from qfin_alphaguard.execution import EqualSpaced
from qfin_alphaguard.guard.config import GuardConfig
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.plan import LOOKBACK_DAYS, build_plans, positions_for
from qfin_alphaguard.sessions import session_days

Action = Split | Dividend | Delisting


@dataclass(frozen=True)
class Inputs:
    """What every run of one backtest shares, whatever its capital."""

    days: tuple[date, ...]  # the sessions it trades
    universes: Mapping[int, tuple[str, ...]]  # each year's trading universe
    minute_days: Mapping[str, frozenset[date]]  # the days each stock has minutes
    actions: tuple[Action, ...]  # splits, dividends and delistings, by day


def first_day(daily: pl.DataFrame) -> date:
    """The first session with a year of daily returns before it."""
    days = session_days(daily["date"].min(), daily["date"].max())
    if len(days) <= LOOKBACK_DAYS + 1:
        raise ValueError("the daily bars do not cover a year and a day")
    return days[LOOKBACK_DAYS + 1]


def prepare(
    daily: pl.DataFrame,
    root: Path,
    first: date | None = None,
    last: date | None = None,
) -> Inputs:
    """The days, universes, minutes and corporate actions of one backtest.

    `first` defaults to first_day; `last`, to the last day with both daily
    bars and minutes. Neither may go beyond the data.
    """
    first = max(first or first_day(daily), first_day(daily))
    last = min(last or daily["date"].max(), daily["date"].max())
    if last < first:
        raise ValueError(f"no backtest from {first} to {last}")
    universes = trading_universes(daily, range(first.year, last.year + 1))
    windows = minute_windows(universes)
    minute_days = stored_minute_days(root, windows, first, last)
    if not minute_days:
        raise ValueError(f"no stored minutes from {first} to {last}")
    last = min(last, max(max(found) for found in minute_days.values()))
    days = tuple(session_days(first, last))
    minute_days = {
        ticker: frozenset(day for day in found if day <= last)
        for ticker, found in minute_days.items()
    }
    actions = holdable_actions(daily, windows, first, last)
    actions += delistings(minute_days, days)
    # By day and stock; a delisting comes last, so a dividend of the same day
    # is paid on the shares before they are paid out. The sort is stable, so
    # a dividend still comes before a split, as corporate_actions puts it.
    actions.sort(key=lambda a: (a.day, a.ticker, isinstance(a, Delisting)))
    return Inputs(days, universes, minute_days, tuple(actions))


def stored_minute_days(
    root: Path, windows: Mapping[int, Mapping[str, date]], first: date, last: date
) -> dict[str, frozenset[date]]:
    """The days each stock has minutes in the store, as minute_feed reads them."""
    found: dict[str, set[date]] = defaultdict(set)
    for year in range(first.year, last.year + 1):
        for ticker, until in windows.get(year, {}).items():
            path = intraday_path(root, ticker, year)
            if not path.exists():
                continue
            days = (
                pl.scan_parquet(path)
                .select(pl.col("start").dt.date().alias("day"))
                .filter(pl.col("day").is_between(first, min(until, last)))
                .unique()
                .collect()["day"]
            )
            found[ticker].update(days)
    return {ticker: frozenset(days) for ticker, days in found.items() if days}


def holdable_actions(
    daily: pl.DataFrame,
    windows: Mapping[int, Mapping[str, date]],
    first: date,
    last: date,
) -> list[Action]:
    """Splits and dividends of each stock on the days the backtest can hold it.

    Each year's stocks are taken from the first day to the end of their
    window, with the session before, so an action on the first day is seen.
    """
    actions: list[Action] = []
    for year in range(first.year, last.year + 1):
        window = windows.get(year, {})
        if not window:
            continue
        start = max(first, date(year, 1, 1))
        until = pl.DataFrame(
            {
                "ticker": list(window),
                "until": [min(day, last) for day in window.values()],
            },
            schema={"ticker": pl.String, "until": pl.Date},
        )
        rows = daily.join(until, on="ticker")
        before = (
            rows.filter(pl.col("date") < start)
            .sort("date")
            .group_by("ticker")
            .last()
            .select(daily.columns)
        )
        inside = rows.filter(pl.col("date").is_between(start, pl.col("until")))
        actions += corporate_actions(pl.concat([before, inside.select(daily.columns)]))
    return actions


def delistings(
    minute_days: Mapping[str, frozenset[date]], days: Iterable[date]
) -> list[Delisting]:
    """A Delisting on each session a stock has no minutes after having had them."""
    pairs = list(pairwise(days))
    return [
        Delisting(ticker, day)
        for ticker, found in sorted(minute_days.items())
        for previous, day in pairs
        if previous in found and day not in found
    ]


class Backtest:
    """One backtest's inputs, run at as many capitals as asked."""

    def __init__(
        self,
        daily: pl.DataFrame,
        root: Path,
        limits: GuardConfig,
        first: date | None = None,
        last: date | None = None,
    ) -> None:
        self.daily = daily
        self.root = root
        self.limits = limits
        self.costs = CostModel()  # Alpaca's fees, and a half spread of 1 bp
        self.inputs = prepare(daily, root, first, last)
        self._plans: dict[int, dict[date, DailyPlan]] = {}

    def plans(self, capital: float) -> dict[date, DailyPlan]:
        """Each day's plan at `capital`, built once for every capital alike.

        A capital limits a plan only by how many stocks it can build; past the
        universe's size, every capital gets the same plans.
        """
        holds = min(positions_for(capital, ExecutionSettings()), TRADING_UNIVERSE_SIZE)
        if holds not in self._plans:
            self._plans[holds] = build_plans(
                self.daily, self.inputs.days, capital, self.limits.max_weight
            )
        return self._plans[holds]

    def run(self, capital: float) -> BacktestResult:
        """The nightly plan from `capital` in cash, through Guard and the broker."""
        inputs = self.inputs
        strategy = EqualSpaced(
            self.plans(capital),
            self.limits.price_collar_bps,
            self.limits.max_orders_per_min,
        )
        feed = minute_feed(self.root, inputs.universes, inputs.days[0], inputs.days[-1])
        broker = NextBarBroker(self.costs)
        return run_backtest(
            feed, strategy, broker, Guard(self.limits), capital, inputs.actions
        )
