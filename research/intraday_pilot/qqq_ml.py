"""Intraday pilot: can ML tell, at 10:00, how good a day QQQ momentum will have?

Each day's features use only what is known at 10:00: the opening gap, the
first 30 minutes' move, range and volume against their 14-day averages, the
noise band at 10:00, recent volatility, and the strategy's own recent days.
The target is the day's return of the strategy at 1x. Walk-forward by year:
each year's models are fitted on the years before it only.

Policies, each sized as decision 0011 (cap 2x) and charged the same costs:
- plain: every day;
- skip: no trading on days the model ranks in the bottom third (of its own
  training predictions);
- scale: half size in the bottom third, 1.5x in the top third (still capped).
"""

import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

DAYS = Path(sys.argv[1])
CAP = 2.0
PER_SHARE, PER_VALUE = 0.001, 0.15e-4

d = pl.read_parquet(DAYS).sort("day")
d = d.with_columns(
    (pl.col("volume30") / pl.col("volume30").shift(1).rolling_mean(14))
    .log()
    .alias("rel_volume30"),
    (pl.col("first30") / pl.col("sigma30")).alias("first30_in_sigma"),
    (pl.col("gap").abs() / pl.col("sigma30")).alias("gap_in_sigma"),
    (pl.col("range30") / pl.col("range30").shift(1).rolling_mean(14))
    .log()
    .alias("rel_range30"),
    pl.col("gross_1x").shift(1).alias("strategy_1d"),
    pl.col("gross_1x").shift(1).rolling_sum(5).alias("strategy_5d"),
    pl.col("gross_1x").shift(1).rolling_sum(20).alias("strategy_20d"),
    pl.col("trend_day").shift(1).alias("prev_day"),
    (pl.col("vol14") / pl.col("vol14").shift(1).rolling_mean(60))
    .log()
    .alias("vol_regime"),
    pl.col("day").dt.weekday().alias("weekday"),
).drop_nulls()
FEATURES = [
    "gap",
    "first30",
    "range30",
    "rel_volume30",
    "first30_in_sigma",
    "gap_in_sigma",
    "rel_range30",
    "sigma30",
    "vol14",
    "vol_regime",
    "strategy_1d",
    "strategy_5d",
    "strategy_20d",
    "prev_day",
    "weekday",
]
size = np.minimum(CAP, 0.02 / d["vol14"].to_numpy())
unit_cost = d["traded_1x"].to_numpy() * (PER_VALUE + PER_SHARE / d["price"].to_numpy())
gross = d["gross_1x"].to_numpy()
years = d["day"].dt.year().to_numpy()

policies = {
    name: np.zeros(len(d))
    for name in ("plain", "skip_gbt", "scale_gbt", "skip_ridge", "scale_ridge")
}
tested = np.zeros(len(d), dtype=bool)
for year in range(2018, 2027):
    train, test = years < year, years == year
    if not test.any():
        continue
    X = d.select(FEATURES).to_numpy()
    for name, model in (
        (
            "gbt",
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.03,
                max_leaf_nodes=8,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=0,
            ),
        ),
        ("ridge", Ridge(alpha=10.0)),
    ):
        mean, sd = X[train].mean(0), X[train].std(0) + 1e-12
        model.fit((X[train] - mean) / sd, gross[train])
        fitted = model.predict((X[train] - mean) / sd)
        low, high = np.quantile(fitted, [1 / 3, 2 / 3])
        pred = model.predict((X[test] - mean) / sd)
        skip = np.where(pred < low, 0.0, 1.0)
        scale = np.where(pred < low, 0.5, np.where(pred > high, 1.5, 1.0))
        for policy, mult in (("skip", skip), ("scale", scale)):
            s = np.minimum(CAP, size[test] * mult)
            policies[f"{policy}_{name}"][test] = gross[test] * s - unit_cost[test] * s
    policies["plain"][test] = gross[test] * size[test] - unit_cost[test] * size[test]
    tested |= test

norm = NormalDist()
results = {}
for name, r in policies.items():
    r = r[tested]
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
    worst = np.min(np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r)) - 1)
    results[name] = r
    print(
        f"{name:12} {growth:+.1%} a year, Sharpe {sharpe:.2f}, max drawdown {worst:.1%}, "
        f"days traded {np.mean(r != 0):.0%}"
    )
# Is a policy's daily return better than plain's? Paired t-statistic.
plain = results["plain"]
for name, r in results.items():
    if name == "plain":
        continue
    diff = r - plain
    t = diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))
    print(f"{name:12} minus plain: {diff.mean() * 252:+.2%} a year, t {t:+.2f}")
by_year = (
    pl.DataFrame({"year": years[tested], **{k: v for k, v in results.items()}})
    .group_by("year")
    .agg([((pl.col(k) + 1).product() - 1).alias(k) for k in results])
    .sort("year")
)
print(by_year)
