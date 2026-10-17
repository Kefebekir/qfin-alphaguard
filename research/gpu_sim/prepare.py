"""An ETF's sessions as arrays for the GPU simulator: one row a session, one column a minute.

As research/intraday_pilot/qqq_momentum.py reads them: each session's minutes from its open
(the US sessions of sessions.py) to its last bar. Saved as numpy arrays: close and the
next-fill open by minute (NaN where no bar), typical price times volume and volume (for
VWAP), each session's first open, last close and length in minutes, and `clean`: whether the
session has the 15 closes before it and no split (an open more than 20% from the last close)
or missing session among the sessions its noise band and volatility read, so that they never
mix the two sides of a split.

Usage: prepare.py ROOT OUT_NPZ [TICKER]   (QQQ by default)
"""

import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT, OUT = Path(sys.argv[1]), sys.argv[2]
TICKER = sys.argv[3] if len(sys.argv) > 3 else "QQQ"
WIDTH = 390
calendar = sessions(date(2016, 1, 1), date(2026, 12, 31))
opens = {s.day: s.open for s in calendar}
previous = {b.day: a.day for a, b in pairwise(calendar)}
bars = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(ROOT, TICKER, y)).select(
                "start", "open", "high", "low", "close", "volume"
            )
            for y in range(2016, 2027)
            if intraday_path(ROOT, TICKER, y).exists()
        ]
    )
    .with_columns(pl.col("start").dt.date().alias("day"))
    .sort("start")
)
groups = [
    (day, g) for (day,), g in bars.group_by("day", maintain_order=True) if day in opens
]
n_days = len(groups)
close = np.full((n_days, WIDTH), np.nan)
fill_open = np.full((n_days, WIDTH), np.nan)
pv = np.zeros((n_days, WIDTH))
vol = np.zeros((n_days, WIDTH))
first_open, last_close, length, days = (
    np.empty(n_days),
    np.empty(n_days),
    np.empty(n_days, dtype=np.int64),
    [],
)
for i, (day, g) in enumerate(groups):
    k = (g["start"] - opens[day]).dt.total_minutes().to_numpy()
    keep = (k >= 0) & (k < WIDTH)
    k = k[keep].astype(int)
    o, h, lo, c, v = (
        g[col].to_numpy()[keep] for col in ("open", "high", "low", "close", "volume")
    )
    close[i, k], fill_open[i, k] = c, o
    pv[i, k], vol[i, k] = (h + lo + c) / 3 * v, v
    first_open[i], last_close[i], length[i] = o[0], c[-1], k.max() + 1
    days.append(day)
# A break at session k (a split, or a missing session, between k - 1 and k) spoils the
# previous close at k and the close-to-close return into k. Session i needs its previous close,
# the 14 returns into sessions i - 14 .. i - 1 and the moves of sessions i - 14 .. i - 1: so no
# break in i - 14 .. i, and i >= 15 (15 closes before it).
broken = np.zeros(n_days, dtype=bool)
for i in range(1, n_days):
    split = abs(np.log(first_open[i] / last_close[i - 1])) > np.log(1.2)
    broken[i] = split or previous.get(days[i]) != days[i - 1]
clean = np.array([i >= 15 and not broken[i - 14 : i + 1].any() for i in range(n_days)])
np.savez(
    OUT,
    clean=clean,
    close=close,
    fill_open=fill_open,
    pv=pv,
    vol=vol,
    first_open=first_open,
    last_close=last_close,
    length=length,
    days=np.array([d.isoformat() for d in days]),
)
print(
    f"{TICKER}: {n_days} sessions ({int(clean.sum())} clean), {int(np.isfinite(close).sum()):,} minutes with a bar"
)
