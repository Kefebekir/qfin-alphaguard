"""Intraday pilot: QQQ intraday momentum (Zarattini, Aziz and Barbon, 2024),
re-implemented to check decision 0011, with each day's record for the ML.

The noise area at minute m of the session is the average absolute move from
the open at that minute over the last 14 days. Its band runs from
min(open, previous close) * (1 - sigma_m) to max(open, previous close) *
(1 + sigma_m). At each HH:00 and HH:30 from 10:00, the strategy goes long
above the band and short below it; a long exits below max(upper band, VWAP),
a short above min(lower band, VWAP). Orders fill at the next minute's open,
and everything is closed at the session's close. The position is
min(cap, 2% / the last 14 days' daily volatility) of the account.
Costs: 0.001 USD a share plus 0.15 bps of the value traded, each side.

Writes one row per day: the day's return at each cap, and what was known at
10:00 for the ML (the opening gap, the first 30 minutes, recent volatility).
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LOOKBACK = 14
TARGET_VOL = 0.02
CAPS = (1.0, 2.0, 4.0)
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
        {
            "day": day,
            "k": k[keep].astype(int),
            "open": g["open"].to_numpy()[keep],
            "high": g["high"].to_numpy()[keep],
            "low": g["low"].to_numpy()[keep],
            "close": g["close"].to_numpy()[keep],
            "volume": g["volume"].to_numpy()[keep].astype(float),
        }
    )

moves: list[np.ndarray] = []  # each day's |close_m / open - 1| by minute
closes = [d["close"][-1] for d in days]
rows = []
for i, d in enumerate(days):
    n = int(d["k"].max()) + 1
    price = np.full(n, np.nan)
    price[d["k"]] = d["close"]
    price = pl.Series(price).fill_null(strategy="forward").to_numpy()
    nxt_open = np.full(n + 1, np.nan)
    nxt_open[d["k"]] = d["open"]
    move = np.abs(price / d["open"][0] - 1)
    if i < LOOKBACK + 1:
        moves.append(move)
        continue
    past = moves[-LOOKBACK:]
    sigma = np.array([np.mean([p[m] for p in past if m < len(p)]) for m in range(n)])
    prev_close = closes[i - 1]
    o = d["open"][0]
    upper = max(o, prev_close) * (1 + sigma)
    lower = min(o, prev_close) * (1 - sigma)
    typical = (d["high"] + d["low"] + d["close"]) / 3
    vwap_k = np.full(n, np.nan)
    vwap_k[d["k"]] = np.cumsum(typical * d["volume"]) / np.maximum(
        np.cumsum(d["volume"]), 1
    )
    vwap = pl.Series(vwap_k).fill_null(strategy="forward").to_numpy()
    daily_returns = np.diff(np.log(closes[i - LOOKBACK - 1 : i]))
    vol = daily_returns.std(ddof=1)
    # Positions in units of "1x the account", decided at HH:00 and HH:30.
    pos, path = 0, []
    for t in range(30, n, 30):
        p = price[t - 1]  # the close of the bar that ends at t
        if pos == 0:
            pos = 1 if p > upper[t - 1] else (-1 if p < lower[t - 1] else 0)
        elif pos == 1 and p < max(upper[t - 1], vwap[t - 1]):
            pos = -1 if p < lower[t - 1] else 0
        elif pos == -1 and p > min(lower[t - 1], vwap[t - 1]):
            pos = 1 if p > upper[t - 1] else 0
        path.append((t, pos))
    # Gross return of a 1x position, and the traded value as a share of it.
    ret, traded, held, entry = 0.0, 0.0, 0, None
    for t, want in path + [(n, 0)]:
        fill = (
            nxt_open[t] if t < n and not np.isnan(nxt_open[t]) else price[min(t, n) - 1]
        )
        if want != held:
            if held != 0:
                ret += held * (fill / entry - 1)
            traded += abs(want - held)
            held, entry = want, fill
    first30 = price[29] / o - 1
    rows.append(
        {
            "day": d["day"],
            "gross_1x": ret,
            "traded_1x": traded,
            "price": o,
            "vol14": vol,
            "gap": o / prev_close - 1,
            "first30": first30,
            "range30": (d["high"][d["k"] < 30].max() - d["low"][d["k"] < 30].min()) / o,
            "volume30": d["volume"][d["k"] < 30].sum(),
            "sigma30": sigma[29],
            "trend_day": price[-1] / o - 1,
        }
    )
    moves.append(move)

out = pl.DataFrame(rows)
for cap in CAPS:
    size = np.minimum(cap, TARGET_VOL / out["vol14"].to_numpy())
    cost = (
        out["traded_1x"].to_numpy()
        * size
        * (PER_VALUE + PER_SHARE / out["price"].to_numpy())
    )
    out = out.with_columns(
        pl.Series(f"net_{cap:g}x", out["gross_1x"].to_numpy() * size - cost),
        pl.Series(f"size_{cap:g}x", size),
    )
out.write_parquet(OUT)
for cap in CAPS:
    r = out[f"net_{cap:g}x"].to_numpy()
    growth = np.prod(1 + r) ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252)
    print(f"cap {cap:g}x: {growth:+.1%} a year, Sharpe {sharpe:.2f}, days {len(r)}")
