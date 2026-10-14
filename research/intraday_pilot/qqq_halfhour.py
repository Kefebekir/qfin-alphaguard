"""Intraday pilot: let ML make QQQ's half-hourly decisions.

At each HH:00 and HH:30 from 10:00, the momentum rule of qqq_momentum.py
decides long, short or flat. Here a model sees what the rule sees, and more,
at the same moments, and predicts the return to the next decision (to the
close for the last one):

- where the price stands against the noise band's edges and VWAP, in units
  of the band's width at that minute;
- the move since the open and over the last 30 minutes, the opening gap;
- the time of day, the volume so far against its 14-day average at the same
  minute, the day's realised volatility so far against the last 14 days';
- the rule's own position.

Walk-forward by year, each year's model fitted on the years before it only.
Policies, each sized min(2, 2% / 14-day volatility) and charged 0.001 USD a
share plus 0.15 bps a side, as the rule:

- rule: the rule alone;
- ml: the sign of the prediction, when its size is in the top half of the
  training predictions' sizes; flat otherwise;
- meta: the rule's position, kept only while the model expects it to earn
  (meta-labelling, Lopez de Prado 2018); flat otherwise.
"""

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])
LOOKBACK = 14
CAP, TARGET_VOL = 2.0, 0.02
PER_SHARE, PER_VALUE = 0.001, 0.15e-4

opens = {s.day: s.open for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
bars = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(ROOT, "QQQ", y)).select(
                "start", "open", "high", "low", "close", "volume"
            )
            for y in range(2016, 2027)
            if intraday_path(ROOT, "QQQ", y).exists()
        ]
    )
    .with_columns(pl.col("start").dt.date().alias("day"))
    .sort("start")
)
days = []
for (day,), g in bars.group_by("day", maintain_order=True):
    start = opens.get(day)
    if start is None:
        continue
    k = ((g["start"] - start).dt.total_minutes()).to_numpy()
    keep = k >= 0
    days.append(
        {"day": day, "k": k[keep].astype(int)}
        | {
            c: g[c].to_numpy()[keep].astype(float)
            for c in ("open", "high", "low", "close", "volume")
        }
    )

moves, volumes, closes, rows = [], [], [d["close"][-1] for d in days], []
for i, d in enumerate(days):
    n = int(d["k"].max()) + 1
    price = np.full(n, np.nan)
    price[d["k"]] = d["close"]
    price = pl.Series(price).fill_null(strategy="forward").to_numpy()
    nxt_open = np.full(n + 1, np.nan)
    nxt_open[d["k"]] = d["open"]
    vol_by_minute = np.zeros(n)
    vol_by_minute[d["k"]] = d["volume"]
    cum_volume = np.cumsum(vol_by_minute)
    move = np.abs(price / d["open"][0] - 1)
    if i < LOOKBACK + 1:
        moves.append(move)
        volumes.append(cum_volume)
        continue
    past = moves[-LOOKBACK:]
    sigma = np.array([np.mean([p[m] for p in past if m < len(p)]) for m in range(n)])
    usual_volume = np.array(
        [np.mean([v[m] for v in volumes[-LOOKBACK:] if m < len(v)]) for m in range(n)]
    )
    prev_close, o = closes[i - 1], d["open"][0]
    upper = max(o, prev_close) * (1 + sigma)
    lower = min(o, prev_close) * (1 - sigma)
    typical = (d["high"] + d["low"] + d["close"]) / 3
    vwap_k = np.full(n, np.nan)
    vwap_k[d["k"]] = np.cumsum(typical * d["volume"]) / np.maximum(
        np.cumsum(d["volume"]), 1
    )
    vwap = pl.Series(vwap_k).fill_null(strategy="forward").to_numpy()
    daily = np.diff(np.log(closes[i - LOOKBACK - 1 : i]))
    vol14 = daily.std(ddof=1)
    minute_returns = np.diff(np.log(price))
    times = list(range(30, n, 30))
    fills = [nxt_open[t] if not np.isnan(nxt_open[t]) else price[t - 1] for t in times]
    fills.append(price[-1])  # the close
    pos = 0
    for j, t in enumerate(times):
        p = price[t - 1]
        if pos == 0:
            pos = 1 if p > upper[t - 1] else (-1 if p < lower[t - 1] else 0)
        elif pos == 1 and p < max(upper[t - 1], vwap[t - 1]):
            pos = -1 if p < lower[t - 1] else 0
        elif pos == -1 and p > min(lower[t - 1], vwap[t - 1]):
            pos = 1 if p > upper[t - 1] else 0
        width = o * sigma[t - 1]
        rows.append(
            {
                "day": d["day"],
                "t": t,
                "rule": pos,
                "ret": fills[j + 1] / fills[j] - 1,
                "price": o,
                "vol14": vol14,
                # the features, all known at the end of minute t - 1
                "above_upper": (p - upper[t - 1]) / width,
                "below_lower": (p - lower[t - 1]) / width,
                "from_vwap": (p - vwap[t - 1]) / width,
                "from_open": (p / o - 1) / vol14,
                "last_30": (p / price[t - 31] - 1 if t > 30 else p / o - 1) / vol14,
                "gap": (o / prev_close - 1) / vol14,
                "time": t / n,
                "minutes_left": n - t,
                "volume_so_far": math.log(
                    max(cum_volume[t - 1], 1) / max(usual_volume[t - 1], 1)
                ),
                "vol_so_far": float(
                    np.sqrt(np.nansum(minute_returns[: t - 1] ** 2) * n / (t - 1))
                    / vol14
                ),
                "sigma": sigma[t - 1] / vol14,
                "yesterday": math.log(closes[i - 1] / closes[i - 2]) / vol14,
            }
        )
    moves.append(move)
    volumes.append(cum_volume)

FEATURES = [
    "rule",
    "above_upper",
    "below_lower",
    "from_vwap",
    "from_open",
    "last_30",
    "gap",
    "time",
    "minutes_left",
    "volume_so_far",
    "vol_so_far",
    "sigma",
    "yesterday",
]
d = pl.DataFrame(rows)
print("rows with a missing value:", d.height - d.fill_nan(None).drop_nulls().height)
d = d.fill_nan(None).drop_nulls()
size = np.minimum(CAP, TARGET_VOL / d["vol14"].to_numpy())
unit_cost = PER_VALUE + PER_SHARE / d["price"].to_numpy()
X, ret = d.select(FEATURES).to_numpy(), d["ret"].to_numpy()
y = ret / d["vol14"].to_numpy()  # in units of the day's usual volatility
years, day_index = d["day"].dt.year().to_numpy(), d["day"].to_numpy()
rule = d["rule"].to_numpy().astype(float)
positions = {"rule": rule.copy()}
for name in ("ml_gbt", "ml_ridge", "meta_gbt", "meta_ridge"):
    positions[name] = np.zeros(len(d))
tested = years >= 2018
for year in range(2018, 2027):
    train, test = years < year, years == year
    if not test.any():
        continue
    mean, sd = X[train].mean(0), X[train].std(0) + 1e-12
    for name, model in (
        (
            "gbt",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=200,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=0,
            ),
        ),
        ("ridge", Ridge(alpha=10.0)),
    ):
        model.fit((X[train] - mean) / sd, y[train])
        fitted = model.predict((X[train] - mean) / sd)
        pred = model.predict((X[test] - mean) / sd)
        strong = np.abs(pred) >= np.median(np.abs(fitted))
        positions[f"ml_{name}"][test] = np.where(strong, np.sign(pred), 0.0)
        positions[f"meta_{name}"][test] = np.where(
            rule[test] * pred > 0, rule[test], 0.0
        )

frame = pl.DataFrame({"day": day_index, "year": years, "t": d["t"].to_numpy()})
results = {}
print("QQQ, half-hourly decisions, walk-forward 2018-2026, cap 2x, after costs")
for name, pos in positions.items():
    # A position change at a decision trades |change| x size; every day ends flat.
    prev = np.concatenate([[0.0], pos[:-1]])
    prev[np.concatenate([[True], day_index[1:] != day_index[:-1]])] = 0.0
    last = np.concatenate([day_index[1:] != day_index[:-1], [True]])
    traded = np.abs(pos - prev) + np.where(last, np.abs(pos), 0.0)
    pnl = size * (pos * ret - traded * unit_cost)
    r = (
        frame.with_columns(pl.Series("pnl", pnl))
        .filter(pl.col("year") >= 2018)
        .group_by("day")
        .agg(pl.col("pnl").sum())
        .sort("day")["pnl"]
        .to_numpy()
    )
    results[name] = r
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
    hit = np.mean(
        np.sign(pos[tested & (pos != 0)]) == np.sign(ret[tested & (pos != 0)])
    )
    print(
        f"{name:10} {growth:+.1%} a year, Sharpe {sharpe:.2f}, in the market "
        f"{np.mean(pos[tested] != 0):.0%} of half hours, right {hit:.1%} of them"
    )
for name, r in results.items():
    if name == "rule":
        continue
    diff = r - results["rule"]
    t = diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))
    print(f"{name:10} minus rule: {diff.mean() * 252:+.2%} a year, t {t:+.2f}")
