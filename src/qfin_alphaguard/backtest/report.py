"""What `qfin backtest` reports: the plan's results against two baselines.

Decision 0015 sets what is measured and how.

- Performance, from the value at each session's close and the starting
  value: the return a year, compounded over 252 sessions; the volatility, the
  daily returns' standard deviation times the square root of 252; the Sharpe
  ratio, their mean over that deviation, times the same, with no risk-free
  rate; the maximum drawdown, the deepest fall from an earlier high; the
  share of winning days; the worst and the best day.
- Trading, for the plan alone: the orders sent; the turnover, the value of
  every fill over the account's average value, a year; the fees, as charged;
  the spread, as the cost model's half spread on every fill's value, an
  estimate; Guard's events by rule; the held stocks paid out at a delisting.
- Two baselines over the same sessions, before costs:
  - the same stocks, equal weight: on the first session of each year, the
    year's trading universe in equal parts, bought at the previous close and
    held through the year. A stock whose bars end is held as cash at its last
    close.
  - SPY, the S&P 500 ETF, bought at the previous close and held.

Both baselines use the daily bars, adjusted for splits and dividends: their
returns include the dividends.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import polars as pl

from qfin_alphaguard.backtest.broker import CostModel
from qfin_alphaguard.backtest.engine import BacktestResult
from qfin_alphaguard.backtest.run import Inputs
from qfin_alphaguard.data.ingest import BENCHMARKS
from qfin_alphaguard.events import DailyPlan, Delisting, Fill, Order, RiskEvent

TRADING_DAYS = 252


@dataclass(frozen=True)
class Performance:
    """How a value did, session by session, from where it started."""

    total_return: float
    annual_return: float  # compounded over 252 sessions a year
    annual_volatility: float
    sharpe: float  # no risk-free rate
    max_drawdown: float  # negative: the deepest fall from an earlier high
    winning_days: float  # the share of sessions with a gain
    worst_day: float
    best_day: float


@dataclass(frozen=True)
class Trading:
    """What the plan's trading did and cost."""

    orders: int
    turnover: float  # every fill's value over the average account value, a year
    fees: float  # USD, as the broker charged them
    spread: float  # USD: the cost model's half spread on every fill, an estimate
    costs_a_year: float  # fees and spread over the average account value, a year
    guard: dict[str, int]  # Guard's events, "rule action": count
    delistings: int  # held stocks paid out at their last price


def performance(values: Sequence[float], start: float) -> Performance:
    """Performance of the closing `values` of consecutive sessions from `start`."""
    series = np.array([start, *values], dtype=float)
    if len(series) < 3 or not (series > 0).all():
        raise ValueError("performance needs two sessions or more of positive values")
    returns = series[1:] / series[:-1] - 1
    deviation = returns.std(ddof=1)
    peaks = np.maximum.accumulate(series)
    return Performance(
        total_return=float(series[-1] / start - 1),
        annual_return=float((series[-1] / start) ** (TRADING_DAYS / len(returns)) - 1),
        annual_volatility=float(deviation * math.sqrt(TRADING_DAYS)),
        sharpe=float(returns.mean() / deviation * math.sqrt(TRADING_DAYS))
        if deviation > 0
        else math.nan,
        max_drawdown=float((series / peaks - 1).min()),
        winning_days=float((returns > 0).mean()),
        worst_day=float(returns.min()),
        best_day=float(returns.max()),
    )


def trading(result: BacktestResult, costs: CostModel, start: float) -> Trading:
    """The orders, turnover, costs and Guard's events of one run."""
    fills = [record for record in result.log if isinstance(record, Fill)]
    traded = sum(fill.quantity * fill.price for fill in fills)
    values = [start, *(close.value for close in result.closes)]
    average = sum(values) / len(values)
    years = len(result.closes) / TRADING_DAYS
    fees = sum(fill.commission for fill in fills)
    spread = traded * costs.half_spread_bps / 10_000
    events = Counter(
        f"{event.rule} {event.action.value}"
        for event in result.log
        if isinstance(event, RiskEvent)
    )
    return Trading(
        orders=sum(isinstance(record, Order) for record in result.log),
        turnover=traded / average / years,
        fees=fees,
        spread=spread,
        costs_a_year=(fees + spread) / average / years,
        guard=dict(sorted(events.items())),
        delistings=sum(isinstance(record, Delisting) for record in result.log),
    )


def previous_session(daily: pl.DataFrame, day: date) -> date:
    """The last date with daily bars before `day`."""
    before = daily.filter(pl.col("date") < day)["date"]
    if before.is_empty():
        raise ValueError(f"no daily bars before {day}")
    return before.max()


def closes(
    daily: pl.DataFrame, tickers: Sequence[str], first: date, last: date
) -> pl.DataFrame:
    """Adjusted closes of `tickers` from `first` to `last`, one column each.

    A close missing after a stock's last bar keeps the last one: the stock is
    held as cash at its last close.
    """
    return (
        daily.filter(
            pl.col("ticker").is_in(list(tickers))
            & pl.col("date").is_between(first, last)
        )
        .pivot(on="ticker", index="date", values="close")
        .sort("date")
        .with_columns(pl.exclude("date").forward_fill())
    )


def equal_weight(
    daily: pl.DataFrame, universes: Mapping[int, Sequence[str]], days: Sequence[date]
) -> list[float]:
    """Each session's close of the equal-weight baseline, starting from 1."""
    values: list[float] = []
    value = 1.0
    for year in sorted({day.year for day in days}):
        sessions = [day for day in days if day.year == year]
        bought = previous_session(daily, sessions[0])
        prices = closes(daily, universes[year], bought, sessions[-1])
        first = prices.filter(pl.col("date") == bought)
        held = [ticker for ticker in prices.columns[1:] if first[ticker][0] is not None]
        if not held:
            raise ValueError(f"no stock of the {year} universe has a close on {bought}")
        shares = {ticker: value / len(held) / first[ticker][0] for ticker in held}
        by_day = dict(
            zip(
                prices["date"],
                sum(prices[ticker] * n for ticker, n in shares.items()),
                strict=True,
            )
        )
        for day in sessions:
            value = by_day[day]
            values.append(value)
    return values


def benchmark(daily: pl.DataFrame, ticker: str, days: Sequence[date]) -> list[float]:
    """Each session's close of `ticker` bought at the previous close, from 1."""
    bought = previous_session(daily, days[0])
    prices = closes(daily, [ticker], bought, days[-1])
    if ticker not in prices.columns:
        raise ValueError(f"no daily bars for {ticker}")
    by_day = dict(zip(prices["date"], prices[ticker], strict=True))
    start = by_day.get(bought)
    if start is None or any(by_day.get(day) is None for day in days):
        raise ValueError(f"{ticker} lacks daily bars between {bought} and {days[-1]}")
    return [by_day[day] / start for day in days]


def wanted_without_minutes(
    plans: Mapping[date, DailyPlan], minute_days: Mapping[str, frozenset[date]]
) -> int:
    """Stock-days a plan held a stock that had no minutes that day."""
    return sum(
        day not in minute_days.get(ticker, frozenset())
        for day, plan in plans.items()
        for ticker in plan.target_weights
    )


def summary(
    daily: pl.DataFrame,
    inputs: Inputs,
    plans: Mapping[date, DailyPlan],
    runs: Mapping[float, BacktestResult],
    costs: CostModel,
) -> dict:
    """The whole report as plain data, for the JSON file and the screen.

    Nothing in it depends on when or where it was made, so the same inputs
    give the same report; each run's fingerprint is a hash of its whole log.
    """
    days = list(inputs.days)
    plan = {}
    for capital, result in runs.items():
        if [close.day for close in result.closes] != days:
            raise ValueError(f"the run at {capital:g} USD missed some of the sessions")
        values = [close.value for close in result.closes]
        plan[f"{capital:g}"] = {
            "performance": asdict(performance(values, capital)),
            "trading": asdict(trading(result, costs, capital)),
            "fingerprint": result.fingerprint(),
        }
    baselines = {
        "equal weight": asdict(
            performance(equal_weight(daily, inputs.universes, days), 1.0)
        )
    }
    missing = {}
    for ticker in BENCHMARKS:
        try:
            baselines[ticker] = asdict(performance(benchmark(daily, ticker, days), 1.0))
        except ValueError as error:
            missing[ticker] = str(error)
    return {
        "first": days[0].isoformat(),
        "last": days[-1].isoformat(),
        "sessions": len(days),
        "plan": plan,
        "baselines": baselines,
        "missing baselines": missing,
        "data": {
            "delistings": sum(isinstance(a, Delisting) for a in inputs.actions),
            "plan stock-days": sum(len(p.target_weights) for p in plans.values()),
            "plan stock-days without minutes": wanted_without_minutes(
                plans, inputs.minute_days
            ),
        },
    }


def render(report: dict) -> str:
    """The report as a few tables of text."""
    title = (
        f"Backtest from {report['first']} to {report['last']}, "
        f"{report['sessions']:,} sessions: the nightly plan with EqualSpaced, "
        "Alpaca's costs and Guard; the baselines before costs."
    )
    header = (
        f"{'':28}{'a year':>9}{'volatility':>12}{'Sharpe':>8}"
        f"{'max drawdown':>14}{'winning days':>14}{'worst day':>11}"
    )
    lines = [title, "", header]
    rows = [
        (f"Plan, {float(capital):,.0f} USD", run["performance"])
        for capital, run in report["plan"].items()
    ]
    rows += [
        ("Same stocks, equal weight" if name == "equal weight" else name, values)
        for name, values in report["baselines"].items()
    ]
    for name, p in rows:
        lines.append(
            f"{name:28}{p['annual_return']:>+9.1%}{p['annual_volatility']:>12.1%}"
            f"{p['sharpe']:>8.2f}{p['max_drawdown']:>14.1%}"
            f"{p['winning_days']:>14.1%}{p['worst_day']:>11.1%}"
        )
    for ticker, reason in report["missing baselines"].items():
        lines.append(f"{ticker:28}not measured: {reason}")

    header = (
        f"{'':28}{'orders':>9}{'turnover':>12}{'fees':>12}{'spread':>12}"
        f"{'costs a year':>14}"
    )
    lines += ["", header]
    for capital, run in report["plan"].items():
        t = run["trading"]
        lines.append(
            f"{f'Plan, {float(capital):,.0f} USD':28}{t['orders']:>9,}"
            f"{t['turnover']:>12.0%}{t['fees']:>12,.2f}{t['spread']:>12,.2f}"
            f"{t['costs_a_year']:>14.2%}"
        )

    lines.append("")
    for capital, run in report["plan"].items():
        t = run["trading"]
        events = ", ".join(f"{rule} {n:,}" for rule, n in t["guard"].items())
        lines.append(
            f"Guard at {float(capital):,.0f} USD: {events or 'nothing'}; "
            f"{t['delistings']} held stocks paid out at a delisting"
        )
    data = report["data"]
    lines.append(
        f"Data: {data['delistings']} delistings; the plan held a stock without "
        f"minutes on {data['plan stock-days without minutes']:,} of its "
        f"{data['plan stock-days']:,} stock-days, and could not trade it then."
    )
    lines.append(
        "Universe: each year's 50 most liquid S&P 500 members, chosen on 1 January "
        "from that day's members (decision 0005), so stocks that later left the "
        "index are in it, without survivorship bias; the membership history is "
        "kept by one person, not by S&P."
    )
    lines.append("")
    for capital, run in report["plan"].items():
        lines.append(f"Fingerprint at {float(capital):,.0f} USD: {run['fingerprint']}")
    return "\n".join(lines)
