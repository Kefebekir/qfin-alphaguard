"""Before N2 and the network that will carry news to the FPGA: what news tells, in our data.

Three measurements on the development years only, 2016-2021: N2's holdout (2022 - October
2026) is not read, so nothing here can steer it.

1. Direction, an upper bound. Among S&P 500 members with news in N2's window, those whose
   day went best against those whose day went worst: what the next day pays, before costs,
   and the same for members without news. Day t's return runs to its close, 30 minutes past
   N2's window: a bound on the continuation a news model could find, not a strategy.
2. Size. Does a member's news count tell how far it moves the next day, beyond its own
   recent volatility? The added R-squared of log(1 + count) on the next day's absolute
   return, and the next day's move for members with news against those without.
3. QQQ. Does the morning's news count (N1's window, to 09:55) tell the day's range, beyond
   the 14-day volatility the rule sizes by?

Usage: architecture_checks.py   (from the repository, after features.py)
"""

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

FIRST, LAST = date(2016, 1, 1), date(2021, 12, 31)
EMB = Path("data/news_reader/emb")
PRICES = Path("data/raw/prices.parquet")


def t_stat(v: np.ndarray) -> float:
    return float(v.mean() / v.std(ddof=1) * math.sqrt(len(v)))


def added_r2(y: np.ndarray, base: np.ndarray, extra: np.ndarray) -> tuple[float, float]:
    """R-squared of y on base, and how much extra adds to it (least squares)."""

    def r2(x: np.ndarray) -> float:
        x = np.column_stack([np.ones(len(y)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        return 1 - resid.var() / y.var()

    with_base = r2(base)
    return with_base, r2(np.column_stack([base, extra])) - with_base


# The members' days: returns, the next day's, recent volatility, and the news count.
prices = (
    pl.read_parquet(PRICES, columns=["date", "ticker", "close", "sp500"])
    .filter(pl.col("date").is_between(date(2015, 11, 1), date(2022, 1, 10)))
    .sort("ticker", "date")
    .with_columns(ret=pl.col("close").pct_change().over("ticker"))
    .with_columns(
        next_ret=pl.col("ret").shift(-1).over("ticker"),
        vol20=pl.col("ret").rolling_std(20).over("ticker"),
    )
    .filter(pl.col("date").is_between(FIRST, LAST) & pl.col("sp500"))
    .drop_nulls(["ret", "next_ret", "vol20"])
)
news = (
    pl.concat([pl.read_parquet(f) for f in sorted(EMB.glob("n2_v*.parquet"))])
    .unique(["ticker", "day"])
    .rename({"day": "date"})
)
days = prices.join(news, on=["ticker", "date"], how="left").with_columns(
    pl.col("count").fill_null(0)
)
print(
    f"S&P 500 member-days {FIRST} to {LAST}: {days.height:,}, "
    f"{(days['count'] > 0).sum():,} with news in N2's window"
)

# 1. Direction: the next day after the best and worst days, with and without news.
print("\n1. The next day after the day's best tenth less its worst tenth, before costs")
for label, part in (
    ("with news", days.filter(pl.col("count") > 0)),
    ("without news", days.filter(pl.col("count") == 0)),
):
    spreads = []
    for _, day in part.group_by("date"):
        if day.height < 20:
            continue
        ranked = day.sort("ret")
        n = day.height // 10
        spreads.append(ranked["next_ret"][-n:].mean() - ranked["next_ret"][:n].mean())
    s = np.array(spreads)
    print(
        f"   {label:13s} {s.mean() * 1e4:+6.2f} bps a day (t {t_stat(s):+.2f}) over "
        f"{len(s):,} days; a year {s.mean() * 252:+.1%}"
    )
by_size = days.filter(pl.col("count") > 0).with_columns(
    pl.col("vol20").rank().over("date").alias("calm_rank"),
    pl.len().over("date").alias("n"),
)

# 2. Size: does news tell the next day's move beyond recent volatility?
print(
    "\n2. The next day's absolute move: R-squared on 20-day volatility, and what news adds"
)
y = days["next_ret"].abs().to_numpy()
base = days["vol20"].to_numpy()
count = np.log1p(days["count"].to_numpy())
r2_base, gain = added_r2(y, base, count)
print(f"   volatility alone R2 {r2_base:.3f}; log(1 + count) adds {gain:+.4f}")
moved = days.group_by(pl.col("count") > 0).agg(
    (pl.col("next_ret").abs() / pl.col("vol20")).mean().alias("move_in_vol"),
    pl.len().alias("days"),
)
for row in moved.sort("count").iter_rows(named=True):
    who = "with news" if row["count"] else "without news"
    print(
        f"   {who:13s} next day moves {row['move_in_vol']:.3f} of its 20-day volatility ({row['days']:,} days)"
    )

# 3. QQQ: the morning's news and the day's range.
print("\n3. QQQ: the morning's news count (to 09:55) and the day's range")
n1 = (
    pl.concat([pl.read_parquet(f) for f in sorted(EMB.glob("n1_v*.parquet"))])
    .unique("day")
    .rename({"day": "date"})
)
qqq = (
    pl.read_parquet("data/raw/etf_daily/QQQ.parquet")
    .sort("date")
    .with_columns(ret=pl.col("close").pct_change())
    .with_columns(
        vol14=pl.col("ret").rolling_std(14).shift(1),
        rng=(pl.col("high") / pl.col("low")).log(),
    )
    .filter(pl.col("date").is_between(FIRST, LAST))
    .join(n1, on="date", how="inner")
    .drop_nulls(["vol14", "rng"])
)
r2_base, gain = added_r2(
    qqq["rng"].to_numpy(), qqq["vol14"].to_numpy(), np.log1p(qqq["count"].to_numpy())
)
print(
    f"   {qqq.height:,} sessions: 14-day volatility R2 {r2_base:.3f}; the morning's count adds {gain:+.4f}"
)
