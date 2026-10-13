"""Pilot for decision 0017: score the walk-forward predictions.

- Rank IC: each day's Spearman correlation of a signal with the next h-day
  return; its t-statistic uses only every h-th day, so no two returns overlap.
- Long-only portfolios rebalanced every h sessions (non-overlapping):
  * S&P 500: the top 10% of members by signal, equal weight, against all
    members at equal weight;
  * our universe: the top 10 of the year's 50 most liquid, against the 50.
  Active return = portfolio minus benchmark. Costs: every rebalance pays
  c bps on the sum of absolute weight changes (buys and sells).
- Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) of the best active
  Sharpe ratio, given every signal and universe tried.
"""

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

PRED = Path(sys.argv[1])
UNIVERSES = {
    int(k): set(v) for k, v in json.loads(Path(sys.argv[2]).read_text()).items()
}
H = int(sys.argv[3])
SIGNALS = ["ridge", "gbt", "ensemble", "momentum", "reversal", "low_vol"]
COSTS = (2.0, 5.0, 10.0)
PPY = 252 / H
N = NormalDist()

pred = pl.read_parquet(PRED)
label = f"fwd_{H}"
dates = pred["date"].unique().sort().to_list()
rebalance = dates[::H]


def rank_ic(signal):
    per_day = (
        pred.group_by("date")
        .agg(pl.corr(pl.col(signal).rank(), pl.col(label).rank()).alias("ic"))
        .sort("date")
    )
    ic = per_day["ic"].drop_nulls().to_numpy()
    spaced = (
        per_day.filter(pl.col("date").is_in(rebalance))["ic"].drop_nulls().to_numpy()
    )
    by_year = (
        per_day.group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.col("ic").mean())
        .sort("year")
    )
    return {
        "mean_ic": float(ic.mean()),
        "t_stat": float(spaced.mean() / spaced.std(ddof=1) * math.sqrt(len(spaced))),
        "by_year": {int(y): round(float(v), 4) for y, v in by_year.rows()},
    }


def portfolio(signal, universe):
    active, absolute, turnover = [], [], []
    held: dict[str, float] = {}
    for day in rebalance:
        cross = pred.filter(pl.col("date") == day).drop_nulls(label)
        if universe == "top50":
            cross = cross.filter(pl.col("ticker").is_in(list(UNIVERSES[day.year])))
            k = 10
        else:
            k = max(1, cross.height // 10)
        if cross.height < k * 2:
            continue
        ret = np.expm1(cross[label].to_numpy())
        order = np.argsort(-cross[signal].to_numpy(), kind="stable")[:k]
        chosen = cross["ticker"].to_numpy()[order]
        weights = dict.fromkeys(chosen.tolist(), 1 / k)
        names = set(weights) | set(held)
        turnover.append(sum(abs(weights.get(n, 0) - held.get(n, 0)) for n in names))
        held = weights
        port = ret[order].mean()
        bench = ret.mean()
        active.append(port - bench)
        absolute.append(port)
    a = np.array(active)
    tv = np.array(turnover)
    out = {
        "periods": len(a),
        "active_a_year": float(a.mean() * PPY),
        "active_vol": float(a.std(ddof=1) * math.sqrt(PPY)),
        "info_ratio": float(a.mean() / a.std(ddof=1) * math.sqrt(PPY)),
        "turnover_a_year": float(tv.mean() * PPY),
        "period_sharpe": float(a.mean() / a.std(ddof=1)),
        "skew": float(((a - a.mean()) ** 3).mean() / a.std() ** 3),
        "kurt": float(((a - a.mean()) ** 4).mean() / a.std() ** 4),
    }
    for c in COSTS:
        net = a - tv * c / 10_000
        out[f"net_active_{c:g}bps"] = float(net.mean() * PPY)
        out[f"net_ir_{c:g}bps"] = float(net.mean() / net.std(ddof=1) * math.sqrt(PPY))
    return out


def deflated_sharpe(best, trials_sr, n_obs, skew, kurt):
    """Probability that the best trial's true Sharpe ratio is above zero, after
    selecting it from len(trials_sr) trials (Bailey and Lopez de Prado, 2014)."""
    gamma = 0.5772156649
    var = np.var(trials_sr, ddof=1)
    k = len(trials_sr)
    sr0 = math.sqrt(var) * (
        (1 - gamma) * N.inv_cdf(1 - 1 / k) + gamma * N.inv_cdf(1 - 1 / (k * math.e))
    )
    z = (
        (best - sr0)
        * math.sqrt(n_obs - 1)
        / math.sqrt(1 - skew * best + (kurt - 1) / 4 * best**2)
    )
    return sr0, N.cdf(z)


report = {"horizon": H, "ic": {}, "portfolio": {}}
for s in SIGNALS:
    report["ic"][s] = rank_ic(s)
    for u in ("sp500", "top50"):
        report["portfolio"][f"{s}/{u}"] = portfolio(s, u)
trials = {key: v["period_sharpe"] for key, v in report["portfolio"].items()}
best_key = max(trials, key=trials.get)
best = report["portfolio"][best_key]
sr0, dsr = deflated_sharpe(
    best["period_sharpe"],
    list(trials.values()),
    best["periods"],
    best["skew"],
    best["kurt"],
)
report["deflated"] = {
    "best": best_key,
    "trials": len(trials),
    "sr0_period": sr0,
    "dsr": dsr,
}
print(json.dumps(report, indent=1))
