"""Intraday pilot: end-of-day reversal (Baltussen, Da and Soebhag, 2025) in our
trading universe.

Each day the universe is sorted on ROD3 = on + fh + mid, the return from the
previous close to 60 minutes before the close. The second-to-last half hour
is skipped, as in the paper, and each group's last-half-hour return (lh) is
measured. Long only: buy the day's biggest losers 30 minutes before the
close and sell them at the close, against holding the whole universe over
the same half hour. Costs are a round trip of c bps on the money traded.
Sorting uses log returns; a portfolio's return is the average of its
stocks' simple returns.
"""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

SLICES = Path(sys.argv[1])
COSTS = (0.0, 1.0, 2.0, 3.0)

days = (
    pl.read_parquet(SLICES)
    .with_columns((pl.col("on") + pl.col("fh") + pl.col("mid")).alias("rod3"))
    .drop_nulls(["rod3", "lh", "slh"])
    .filter(pl.col("minutes") >= pl.col("length") - 5)
)
days = days.with_columns(
    (pl.col("rod3").rank("ordinal").over("day") * 5 / (pl.len().over("day") + 1))
    .floor()
    .cast(pl.Int32)
    .alias("quintile"),
    pl.col("lh").exp().sub(1).alias("lh_simple"),
)


def summary(name: str, daily: np.ndarray) -> str:
    mean, sd = daily.mean(), daily.std(ddof=1)
    t = mean / sd * math.sqrt(len(daily))
    return f"{name:34} {mean * 1e4:+6.2f} bps a day (t {t:+5.1f})"


by_q = (
    days.group_by("day", "quintile")
    .agg(pl.col("lh_simple").mean())
    .pivot(on="quintile", index="day", values="lh_simple")
    .sort("day")
    .drop_nulls()
)
print("Last half hour by ROD3 quintile (0 = biggest losers), 2016-2026:")
for q in range(5):
    print("  " + summary(f"quintile {q}", by_q[str(q)].to_numpy()))
print("  " + summary("losers minus winners", (by_q["0"] - by_q["4"]).to_numpy()))

universe = days.group_by("day").agg(pl.col("lh_simple").mean().alias("all")).sort("day")
for k in (5, 3, 1):
    picks = (
        days.sort("rod3")
        .group_by("day", maintain_order=True)
        .head(k)
        .group_by("day")
        .agg(pl.col("lh_simple").mean().alias("pick"))
        .join(universe, on="day")
        .sort("day")
    )
    excess = (picks["pick"] - picks["all"]).to_numpy()
    raw = picks["pick"].to_numpy()
    print(f"Buy the {k} biggest losers at close-30, sell at the close:")
    print("  " + summary("excess over the universe", excess))
    for c in COSTS:
        net = raw - c / 1e4
        print(
            "  "
            + summary(f"absolute, after {c:g} bps round trip", net)
            + f", {net.mean() * 252:+.1%} a year on money in for 30 minutes a day"
        )
    by_year = (
        picks.with_columns(pl.col("day").dt.year().alias("y"))
        .group_by("y")
        .agg(((pl.col("pick") - pl.col("all")).mean() * 1e4).alias("bps"))
        .sort("y")
    )
    print("  excess by year (bps a day):", {y: round(v, 2) for y, v in by_year.rows()})
