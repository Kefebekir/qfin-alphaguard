"""Intraday pilot: Ghosh, Neufeld and Sahoo (2022) on the S&P 500, 2018-2026.

Each morning, for every S&P 500 member that day (point in time), the model
sees what is known at the open: the open-to-close returns of the last 1-20
days and of each 20th day back to 240, the returns between closes over the
same spans up to yesterday's close, and today's open against each of those
closes (the opening gap, for one day). It predicts whether the stock's
return from today's open to today's close beats the day's median. The trade
buys at the opening auction the k stocks the model likes most (and, long and
short, sells the k it likes least) and closes them at the closing auction.

The paper (Finance Research Letters, 2022; 1993-2018, Bloomberg prices) used
a random forest of 1,000 trees of depth 10 on 93 such features, trained on 3
years and traded for the next 1, k = 10 long and short, and reports 0.44% a
day after 5 bps a half-turn, with returns shrinking after 2010. Here: the
same features from EODHD's adjusted daily bars, a random forest and
gradient-boosted trees, trained on the 3 years before each year; as simple
rivals, the opening gap alone (buy the largest fall overnight) and yesterday's
open-to-close return alone. The scores are written to OUT.

Usage: open_to_close.py PRICES OUT
"""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from qfin_alphaguard.data.universe import dollar_volume

DAILY, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LAGS = [*range(1, 21), *range(40, 241, 20)]
KS = (5, 10)
COSTS = (0.0, 1.0, 2.0, 5.0)  # bps a half-turn

prices = pl.read_parquet(DAILY).sort("ticker", "date")
c, o = pl.col("close"), pl.col("open")
features = []
exprs = []
for m in LAGS:
    exprs += [
        (c / o - 1).shift(m).over("ticker").alias(f"ir_{m}"),
        (c.shift(1) / c.shift(1 + m) - 1).over("ticker").alias(f"cr_{m}"),
        (o / c.shift(m) - 1).over("ticker").alias(f"or_{m}"),
    ]
    features += [f"ir_{m}", f"cr_{m}", f"or_{m}"]
d = (
    prices.with_columns(
        *exprs,
        (c / o - 1).alias("target"),
        (c / pl.col("adjustment")).shift(1).over("ticker").alias("raw_close"),
        dollar_volume().shift(1).rolling_mean(20).over("ticker").alias("dollar_volume"),
    )
    .filter(pl.col("sp500") & (pl.col("volume") > 0) & (pl.col("raw_close") > 5))
    .drop_nulls([*features, "target", "dollar_volume"])
    .with_columns(pl.col("date").dt.year().alias("year"))
    .with_columns(
        (pl.col("target") > pl.col("target").median().over("date")).alias("label"),
        (pl.col("dollar_volume").rank(descending=True).over("date") <= 100).alias(
            "top100"
        ),
    )
)
print(
    f"{d.height} member-days, {d['date'].min()} .. {d['date'].max()}, "
    f"{d.group_by('date').len()['len'].mean():.0f} members a day"
)

X, label, years = (
    d.select(features).to_numpy(),
    d["label"].to_numpy(),
    d["year"].to_numpy(),
)
scores = {"rf": np.full(len(d), np.nan), "gbt": np.full(len(d), np.nan)}
for year in range(2018, 2027):
    train, test = (years >= year - 3) & (years < year), years == year
    if not test.any():
        continue
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=0
    )
    gbt = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=1000,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    for name, model in (("rf", rf), ("gbt", gbt)):
        model.fit(X[train], label[train])
        scores[name][test] = model.predict_proba(X[test])[:, 1]
    print(f"  {year}: trained on {train.sum()} rows", flush=True)

d = d.with_columns(
    pl.Series("rf", scores["rf"]),
    pl.Series("gbt", scores["gbt"]),
    (-pl.col("or_1")).alias("gap_reversal"),
    (-pl.col("ir_1")).alias("yesterday_reversal"),
).filter(pl.col("year") >= 2018)
d.select(
    "date", "ticker", "target", "rf", "gbt", "gap_reversal", "top100"
).write_parquet(OUT)


def summary(r: np.ndarray) -> str:
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    return (
        f"{growth:+.1%} a year, Sharpe {r.mean() / r.std(ddof=1) * math.sqrt(252):+.2f}"
    )


print("Open to close, S&P 500 members, 2018-2026 out of sample")
for universe, frame in (
    ("all members", d),
    ("100 most traded", d.filter(pl.col("top100"))),
):
    print(f"-- {universe}")
    for signal in ("rf", "gbt", "gap_reversal", "yesterday_reversal"):
        ic = (
            frame.group_by("date")
            .agg(pl.corr(pl.col(signal).rank(), pl.col("target").rank()).alias("ic"))[
                "ic"
            ]
            .drop_nulls()
            .to_numpy()
        )
        print(
            f"{signal:19} rank IC {ic.mean():+.4f} (t {ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)):+.1f})"
        )
        for k in KS:
            ranked = frame.with_columns(
                pl.col(signal)
                .rank("ordinal", descending=True)
                .over("date")
                .alias("top"),
                pl.col(signal).rank("ordinal").over("date").alias("bottom"),
            )
            legs = (
                ranked.group_by("date")
                .agg(
                    pl.col("target").filter(pl.col("top") <= k).mean().alias("long"),
                    pl.col("target")
                    .filter(pl.col("bottom") <= k)
                    .mean()
                    .alias("short"),
                    pl.col("target").mean().alias("all"),
                )
                .sort("date")
            )
            long, short = legs["long"].to_numpy(), legs["short"].to_numpy()
            excess = long - legs["all"].to_numpy()
            line = (
                f"   k={k:2}: long {long.mean() * 1e4:+.1f} bps, short leg {-short.mean() * 1e4:+.1f}, "
                f"long over all {excess.mean() * 1e4:+.1f} (t {excess.mean() / excess.std(ddof=1) * math.sqrt(len(excess)):+.1f})"
            )
            print(line)
            for cost in COSTS:
                print(
                    f"         {cost:g} bps a half-turn: long only {summary(long - 2 * cost / 1e4)}; "
                    f"long and short {summary(long - short - 4 * cost / 1e4)}"
                )
