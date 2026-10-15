"""Exploration: the rule's timing luck. policy_data.py with OFFSET=15 decides at 10:15, 10:45 and
so on (data/intraday_pilot/policy/qqq_offset15); against the :00/:30 grid, and both at half size."""

import math
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "research/step5")
from rule_daily import rule_daily

a = rule_daily("data/intraday_pilot/policy/qqq_offset0/decisions.parquet").rename(
    {"pnl": "a"}
)
b = rule_daily("data/intraday_pilot/policy/qqq_offset15/decisions.parquet").rename(
    {"pnl": "b"}
)
d = (
    a.join(b, on="day")
    .sort("day")
    .with_columns(((pl.col("a") + pl.col("b")) / 2).alias("both"))
)
part = d.filter(pl.col("day").dt.year() <= 2022)
print(
    "2016-2022, corr of the two grids:",
    round(np.corrcoef(part["a"], part["b"])[0, 1], 3),
)
for c, n in (
    ("a", "grid :00/:30 (the rule)"),
    ("b", "grid :15/:45"),
    ("both", "both, half size each"),
):
    x = part[c].to_numpy()
    print(
        f"  {n:24} {np.prod(1 + x) ** (252 / len(x)) - 1:+6.1%} a year, Sharpe {x.mean() / x.std(ddof=1) * math.sqrt(252):5.2f}"
    )
