"""Does the engine's QQQ strategy reproduce the research? (Phase 2, step 2)

Runs momentum.py's NoiseAreaMomentum through run_backtest on the stored QQQ
minutes, from 30,000 USD, and compares each day with qqq_momentum.py's.

Guard is a stand-in that sends every intent: today's guard.yaml forbids
shorts, leverage and reversals, and the limits for this strategy are step 4,
Efe's, by hand. Nothing here trades.

Three brokers:
- at no cost, so only the decisions and fills differ from the research;
- at Alpaca's fees and QQQ's half spread: 0.3 bps, the average of NBBO
  quotes at the decision times on 101 days of 2016-2026 (about 0.36 bps at
  10:00 and 0.23 at 15:59; the spread is a cent);
- at the CostModel's default, 1 bp of half spread, measured for the trading
  universe's stocks (decision 0012): about three times QQQ's.

What is expected to differ, and why:
- the research leaves at the close, the engine on the last minute's open;
- the research turns from long to short in one fill, the engine closes
  first and opens the other side a minute later (Alpaca takes no order that
  crosses zero);
- the engine sizes at the decision's price and sells short whole shares;
- if the bar a decision needs is missing, the engine decides a bar later.

The days the two trade must be the same; their results close.

Usage: engine_check.py ROOT QQQ_DAYS
QQQ_DAYS is qqq_momentum.py's output.
"""

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import SessionClose, run_backtest
from qfin_alphaguard.data.intraday import minute_feed
from qfin_alphaguard.events import Fill
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.momentum import MomentumSettings, NoiseAreaMomentum

ROOT, QQQ_DAYS = Path(sys.argv[1]), Path(sys.argv[2])
CASH = 30_000.0


class SendsEverything:
    """Not Guard: lets every intent through, for this comparison only."""

    def check(self, intent, view):
        return GuardDecision(intent)


research = pl.read_parquet(QQQ_DAYS).sort("day")
first, last = date(2016, 1, 1), research["day"].max()
years = range(first.year, last.year + 1)


def engine_days(costs: CostModel) -> pl.DataFrame:
    feed = minute_feed(ROOT, {year: ["QQQ"] for year in years}, first, last)
    result = run_backtest(
        feed,
        NoiseAreaMomentum(MomentumSettings()),
        NextBarBroker(costs),
        SendsEverything(),
        CASH,
    )
    closes = [r for r in result.log if isinstance(r, SessionClose)]
    traded = {r.time.date() for r in result.log if isinstance(r, Fill)}
    values = np.array([c.value for c in closes])
    returns = values / np.concatenate([[CASH], values[:-1]]) - 1
    return pl.DataFrame(
        {
            "day": [c.day for c in closes],
            "engine": returns,
            "engine_traded": [c.day in traded for c in closes],
        }
    )


def stats(r: np.ndarray) -> str:
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    return (
        f"{growth:+.1%} a year, Sharpe {r.mean() / r.std(ddof=1) * math.sqrt(252):.2f}"
    )


for label, costs, column in (
    ("no cost", CostModel(0.0, 0.0, 0.0), None),
    (
        "Alpaca's fees and QQQ's 0.3 bps half spread",
        CostModel(half_spread_bps=0.3),
        "net_2x",
    ),
    ("the default 1 bp half spread", CostModel(), "net_2x"),
):
    days = engine_days(costs).join(research, on="day", how="inner")
    gross = (days["gross_1x"] * days["size_2x"]).to_numpy()
    reference = gross if column is None else days[column].to_numpy()
    engine = days["engine"].to_numpy()
    research_traded = (days["traded_1x"] > 0).to_numpy()
    engine_traded = days["engine_traded"].to_numpy()
    disagree = days.filter(pl.Series(research_traded != engine_traded))
    diff = engine - reference
    print(f"-- {label}: {days.height} days, {days['day'].min()} .. {days['day'].max()}")
    print(
        f"   research {'before' if column is None else 'after'} costs: {stats(reference)}"
    )
    print(f"   engine: {stats(engine)}")
    print(
        f"   days traded: research {research_traded.sum()}, engine "
        f"{engine_traded.sum()}, in only one of them {disagree.height}"
    )
    print(
        f"   daily difference: mean {diff.mean() * 1e4:+.2f} bps, mean absolute "
        f"{np.abs(diff).mean() * 1e4:.2f} bps, correlation "
        f"{np.corrcoef(engine, reference)[0, 1]:.4f}"
    )
    worst = days.with_columns(pl.Series("diff", diff)).sort(
        pl.col("diff").abs(), descending=True
    )
    print("   largest differences (bps):")
    for row in worst.head(5).iter_rows(named=True):
        print(
            f"     {row['day']}: engine {row['engine'] * 1e4:+.1f}, research "
            f"{(row['gross_1x'] * row['size_2x'] if column is None else row[column]) * 1e4:+.1f}"
        )
    if disagree.height:
        print("   days in only one:", disagree["day"].to_list()[:20])
