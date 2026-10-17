"""QQQ's sessions as arrays for the GPU simulator: one row a session, one column a minute.

As research/intraday_pilot/qqq_momentum.py reads them: each session's minutes from its open
(the US sessions of sessions.py) to its last bar, a minute without a bar carrying the last
close. Saved as numpy arrays: close and the next-fill open by minute (NaN where no bar),
typical price times volume and volume (for VWAP), each session's first open, last close and
length in minutes.

Usage: prepare_qqq.py ROOT OUT_NPZ
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT, OUT = Path(sys.argv[1]), sys.argv[2]
WIDTH = 390
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
np.savez(
    OUT,
    close=close,
    fill_open=fill_open,
    pv=pv,
    vol=vol,
    first_open=first_open,
    last_close=last_close,
    length=length,
    days=np.array([d.isoformat() for d in days]),
)
print(f"{n_days} sessions, {int(np.isfinite(close).sum()):,} minutes with a bar")
