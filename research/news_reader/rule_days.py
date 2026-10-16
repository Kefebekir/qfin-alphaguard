"""The QQQ rule's every session, 2016 to 9 October 2026: the target of pre-registration N1.

momentum.py's NoiseAreaMomentum in the engine, behind guard_qqq.yaml with max_order_usd lifted
(the account grows past the size the cap was set for, as in guard_needs.py), at QQQ's half
spread of 0.3 bps, from 30,000 USD. For each session:

- day: the session's date (New York);
- rule: the account's return that day, from its closing values (shadow.daily_returns), traded
  or not;
- vol14: the standard deviation of QQQ's last 14 daily returns, close to close, up to the
  session before (known at the open), from the dividend-adjusted daily bars.

N1's target is rule / vol14.

Usage: rule_days.py ROOT GUARD_QQQ_YAML QQQ_DAILY_PARQUET OUT_PARQUET
"""

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.backtest.shadow import daily_returns
from qfin_alphaguard.data.intraday import minute_feed
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.momentum import NoiseAreaMomentum

ROOT, LIMITS, QQQ_DAILY, OUT = (
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    Path(sys.argv[4]),
)
FIRST, LAST, CASH = date(2016, 1, 1), date(2026, 10, 9), 30_000.0

feed = minute_feed(
    ROOT, {y: ["QQQ"] for y in range(FIRST.year, LAST.year + 1)}, FIRST, LAST
)
guard = Guard(replace(load_guard_config(LIMITS), max_order_usd=1e12))
result = run_backtest(
    feed,
    NoiseAreaMomentum(),
    NextBarBroker(CostModel(half_spread_bps=0.3)),
    guard,
    CASH,
)
rule = pl.DataFrame(
    {
        "day": list(daily_returns(result, CASH)),
        "rule": list(daily_returns(result, CASH).values()),
    }
)

daily = (
    pl.read_parquet(QQQ_DAILY, columns=["date", "close"])
    .sort("date")
    .with_columns(ret=pl.col("close").pct_change())
    .with_columns(vol14=pl.col("ret").rolling_std(14).shift(1))
    .select(pl.col("date").alias("day"), "vol14")
)
days = rule.join(daily, on="day", how="left").sort("day")
missing = days.filter(pl.col("vol14").is_null())
if missing.height:
    sys.exit(
        f"no 14-day volatility for {missing.height} sessions, e.g. {missing['day'][0]}"
    )
days.write_parquet(OUT)
print(
    f"{days.height} sessions {days['day'].min()} to {days['day'].max()}; "
    f"the rule traded on {(days['rule'] != 0).sum()}; "
    f"{days['rule'].mean() * 252:.1%} a year, Sharpe "
    f"{days['rule'].mean() / days['rule'].std() * 252**0.5:.2f}"
)
