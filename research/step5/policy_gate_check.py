"""Intraday pilot: is the gate's gain over the rule real?

policy_nets.py found one model ahead of the rule on 2019-2022: the gate fitted
on the four index ETFs, t +3.35. Before the holdout years are looked at,
this refits it the same way (same seeds, so it must reproduce the same daily
results) and asks:

- leverage: what factor does it apply, and does the rule scaled by the same
  average factor, or to the same volatility, do as well? Scaling up raises
  the return but not the Sharpe ratio; a real gain raises both;
- years: is the gain spread over the four years or one;
- costs: does it survive the costs doubled;
- what it does: the factor by time of day, and against the features.

The same checks run on the holdout years, 2023-2026, once their run is done
(HOLDOUT=1 and its directory).

Usage: policy_gate_check.py DATA_DIR RUN_DIR [HOLDOUT]
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

DATA, SELECTION = Path(sys.argv[1]), Path(sys.argv[2])
HOLDOUT = len(sys.argv) > 3 and sys.argv[3] == "1"
YEARS = range(2023, 2027) if HOLDOUT else range(2019, 2023)
TAG = "holdout" if HOLDOUT else "selection"
sys.argv = [sys.argv[0], str(DATA), str(SELECTION)]
spec = importlib.util.spec_from_file_location(
    "policy_nets", Path(__file__).with_name("policy_nets.py")
)
pn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pn)

torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
data = pn.DATASET
qqq = data["ticker"] == "QQQ"
index = np.isin(data["ticker"], ["QQQ", "SPY", "IWM", "DIA"])
saved = pl.read_parquet(SELECTION / f"daily_{TAG}.parquet")
ARGS = ("ret", "mask", "size", "cost")

gate_pos, rule_pos, rows = [], [], []
for year in YEARS:
    train = np.where(index & (data["year"] < year - 1))[0]
    valid = np.where(index & (data["year"] == year - 1))[0]
    test = np.where(qqq & (data["year"] == year))[0]
    pos, info = pn.neural("gate", data, train, valid, test)
    pnl = pn.day_pnl(pos, *(data[k][test] for k in ARGS))
    before = (
        saved.filter(
            (pl.col("model") == "gate")
            & (pl.col("pooled") == "index")
            & (pl.col("year") == year)
        )
        .sort("day")["pnl"]
        .to_numpy()
    )
    print(
        f"{year}: refit reproduces the stored daily results: "
        f"{np.allclose(pnl, before, atol=1e-9)}"
    )
    gate_pos.append(pos)
    rule_pos.append(data["rule"][test])
    rows.append(test)
test = np.concatenate(rows)
pos, rule = np.concatenate(gate_pos), np.concatenate(rule_pos)
mask = data["mask"][test] > 0
held = mask & (rule != 0)
factor = np.where(held, pos / np.where(rule == 0, 1, rule), np.nan)
print(
    f"\nthe gate's factor on the rule's positions: mean {np.nanmean(factor):.3f}, "
    f"median {np.nanmedian(factor):.3f}, 10th-90th percentile "
    f"{np.nanpercentile(factor, 10):.3f}-{np.nanpercentile(factor, 90):.3f}"
)


def daily(p, cost_scale=1.0):
    ret, msk, size, cost = (data[k][test] for k in ARGS)
    return pn.day_pnl(p, ret, msk, size, cost * cost_scale)


def line(name, r, base):
    diff = r - base
    t = diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    print(
        f"{name:42} {growth:+6.1%} a year, Sharpe {pn.sharpe(r):5.2f}, "
        f"vol {r.std(ddof=1) * math.sqrt(252):5.1%}, against the rule "
        f"{diff.mean() * 252:+6.2%} a year, t {t:+5.2f}"
    )


base = daily(rule)
gate = daily(pos)
print()
line("rule", base, base)
line("gate (index ETFs)", gate, base)
mean_factor = float(np.nanmean(factor))
room = (pn.CAP / data["size"][test])[:, None]
scaled = rule * np.minimum(mean_factor, room)
line(f"rule x {mean_factor:.3f}, its mean factor", daily(scaled), base)
same_vol = rule * (gate.std() / base.std())
line("rule scaled to the gate's volatility", daily(same_vol), base)
print("\ncosts doubled:")
base2 = daily(rule, 2.0)
line("rule", base2, base2)
line("gate", daily(pos, 2.0), base2)
print("\nby year, against the rule:")
years = data["year"][test]
for year in YEARS:
    on = years == year
    diff = gate[on] - base[on]
    t = diff.mean() / diff.std(ddof=1) * math.sqrt(on.sum())
    print(
        f"  {year}: rule Sharpe {pn.sharpe(base[on]):5.2f}, gate {pn.sharpe(gate[on]):5.2f}, "
        f"difference {diff.mean() * 252:+6.2%} a year, t {t:+5.2f}"
    )
print("\nthe factor by decision time (slot 0 is 10:00):")
for k in range(12):
    f = factor[:, k]
    print(f"  {10 + k // 2}:{'30' if k % 2 else '00'}  {np.nanmean(f):.3f}")
x = data["x"][test]
print("\nthe factor against each feature, over the rule's positions:")
for j, name in enumerate(pn.FEATURES):
    if name == "rule":
        continue
    c = np.corrcoef(x[held][:, j], factor[held])[0, 1]
    print(f"  {name:14} correlation {c:+.3f}")
np.savez(
    SELECTION / "gate_index_positions.npz",
    test=test,
    pos=pos,
    rule=rule,
)
