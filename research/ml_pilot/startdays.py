"""Pilot for decision 0017: a monthly result over every day the month can start.

A portfolio rebalanced every h sessions depends on which session it starts
on. This scores the top 10 of the year's 50 against the 50 for each of the
h possible start days, net of `cost` bps a side.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PRED = Path(sys.argv[1])
UNIVERSES = {int(k): v for k, v in json.loads(Path(sys.argv[2]).read_text()).items()}
H = int(sys.argv[3])
SIGNAL = sys.argv[4]
COST = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0

pred = pl.read_parquet(PRED)
label = f"fwd_{H}"
dates = pred["date"].unique().sort().to_list()
by_date = {day: rows for (day,), rows in pred.group_by("date")}
nets, ratios = [], []
for start in range(H):
    active, turnover, held = [], [], {}
    for day in dates[start::H]:
        cross = (
            by_date[day]
            .filter(pl.col("ticker").is_in(UNIVERSES[day.year]))
            .drop_nulls(label)
        )
        if cross.height < 20:
            continue
        returns = np.expm1(cross[label].to_numpy())
        top = np.argsort(-cross[SIGNAL].to_numpy(), kind="stable")[:10]
        weights = dict.fromkeys(cross["ticker"].to_numpy()[top].tolist(), 0.1)
        names = set(weights) | set(held)
        turnover.append(sum(abs(weights.get(n, 0) - held.get(n, 0)) for n in names))
        held = weights
        active.append(returns[top].mean() - returns.mean())
    net = np.array(active) - np.array(turnover) * COST / 10_000
    nets.append(net.mean() * 252 / H)
    ratios.append(net.mean() / net.std(ddof=1) * np.sqrt(252 / H))
print(
    f"{SIGNAL}, net of {COST:g} bps, over {H} start days: "
    f"mean {np.mean(nets):+.2%}, from {min(nets):+.2%} to {max(nets):+.2%}; "
    f"information ratio mean {np.mean(ratios):.2f}, "
    f"from {min(ratios):.2f} to {max(ratios):.2f}"
)
