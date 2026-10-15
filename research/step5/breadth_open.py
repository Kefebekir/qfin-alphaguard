"""Exploration: does the market's breadth in its first 15 minutes mark the QQQ rule's good days?

From every S&P 500 member's first 16 minutes (research/intraday_pilot/opening_minutes.py), at
09:45: coherence (|share of members up minus share down|), co-movement (variance of the average
1-minute return over the total), and dispersion. The rule's daily result by tercile, 2016-2022.
"""

import math
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "research/step5")
from rule_daily import rule_daily

o = pl.concat(
    [
        pl.read_parquet(f"data/intraday_pilot/opening/{y}.parquet")
        for y in range(2016, 2023)
    ]
).sort("day", "ticker", "start")
o = o.with_columns(
    (pl.col("start") - pl.col("start").min().over("day")).dt.total_minutes().alias("m")
)
first = o.filter(pl.col("m") == 0).select("day", "ticker", pl.col("open").alias("o0"))
at15 = (
    o.filter(pl.col("m") <= 14)
    .group_by("day", "ticker")
    .agg(pl.col("close").last().alias("c15"), pl.col("m").max().alias("mm"))
)
r = (
    first.join(at15, on=["day", "ticker"])
    .filter(pl.col("mm") >= 10)
    .with_columns((pl.col("c15") / pl.col("o0") - 1).alias("r"))
)
o = o.filter(pl.col("m") <= 14).with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("day", "ticker") - 1).alias("mr")
)
avg = o.group_by("day", "m").agg(
    pl.col("mr").mean().alias("avg"), pl.col("mr").var().alias("xvar")
)
co = avg.group_by("day").agg(
    (pl.col("avg").var() / (pl.col("avg").var() + pl.col("xvar").mean())).alias(
        "comove"
    )
)
b = (
    r.group_by("day")
    .agg(
        pl.col("r").sign().mean().abs().alias("coherence"),
        pl.col("r").std().alias("disp"),
    )
    .join(co, on="day")
)
days = rule_daily().filter(pl.col("day").dt.year() <= 2022).join(b, on="day")
for col in ("coherence", "comove", "disp"):
    e = np.quantile(days[col].to_numpy(), [1 / 3, 2 / 3])
    print(f"{col}: tercile edges {np.round(e, 4)}")
    for i, (lo, hi) in enumerate(((-np.inf, e[0]), (e[0], e[1]), (e[1], np.inf))):
        v = days.filter((pl.col(col) > lo) & (pl.col(col) <= hi))["pnl"].to_numpy()
        print(
            f"   tercile {i + 1}: n {len(v)}, mean {v.mean() * 1e4:+.2f} bps, t {v.mean() / v.std(ddof=1) * math.sqrt(len(v)):+.2f}"
        )
