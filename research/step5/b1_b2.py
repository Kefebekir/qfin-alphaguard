import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

SESS = {s.day: s for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
m = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(Path("data/raw"), "QQQ", y)).select(
                "start", "open", "close"
            )
            for y in range(2016, 2027)
        ]
    )
    .with_columns(pl.col("start").dt.date().alias("day"))
    .sort("start")
)
half, b2 = [], []
for (day,), g in m.group_by("day", maintain_order=True):
    s = SESS.get(day)
    if s is None or (s.close - s.open) != timedelta(minutes=390):
        continue  # full sessions only: 13 half hours
    k = ((g["start"] - s.open).dt.total_minutes()).to_numpy()
    c = g["close"].to_numpy()
    o = g["open"].to_numpy()
    price = np.full(390, np.nan)
    price[k[(k >= 0) & (k < 390)]] = c[(k >= 0) & (k < 390)]
    price = (
        pl.Series(price)
        .fill_nan(None)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )
    first = o[k >= 0][0]
    marks = np.concatenate(
        [[first], price[29::30]]
    )  # the open, then the close of each half hour
    r = marks[1:] / marks[:-1] - 1
    half.append({"day": day, **{f"h{i}": r[i] for i in range(13)}})
    day_ret = price[-1] / first - 1
    last15 = price[-1] / price[374] - 1  # 15:45 to the close
    b2.append(last15 - day_ret / 26)
h = pl.DataFrame(half).sort("day")
X, Y = [], []
for i in range(13):
    col = h[f"h{i}"].to_numpy()
    past = pl.Series(col).shift(1).rolling_mean(20).to_numpy()
    ok = ~np.isnan(past)
    X.append(past[ok])
    Y.append(col[ok])
x, y = np.concatenate(X), np.concatenate(Y)
xc = x - x.mean()
b = xc @ (y - y.mean()) / (xc @ xc)
e = y - y.mean() - b * xc
t = b / math.sqrt(e @ e / (len(y) - 2) / (xc @ xc))
print(
    f"B1: QQQ half-hour return on the mean of the same half hour over the 20 sessions before: slope {b:+.3f}, t {t:+.2f} (n {len(y)}) -> {'PASS' if t >= 2.64 else 'fail'}"
)
v = np.array(b2)
print(
    f"B2: last 15 minutes minus an average 15 minutes: {v.mean() * 1e4:+.2f} bps a day, t {v.mean() / v.std(ddof=1) * math.sqrt(len(v)):+.2f} (n {len(v)}) -> {'PASS' if v.mean() / v.std(ddof=1) * math.sqrt(len(v)) >= 2.64 else 'fail'}"
)
