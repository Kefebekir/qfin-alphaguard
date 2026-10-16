import math
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

SESS = {s.day: s for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
ordered = sorted(SESS)
PREV = {b: a for a, b in pairwise(ordered)}


def ols(y, x):
    x = x - x.mean()
    b = (x @ (y - y.mean())) / (x @ x)
    e = y - y.mean() - b * x
    return b, b / math.sqrt((e @ e) / (len(y) - 2) / (x @ x))


for t in ("QQQ", "SPY"):
    m = (
        pl.concat(
            [
                pl.read_parquet(intraday_path(Path("data/raw"), t, y)).select(
                    "start", "open", "close", "source"
                )
                for y in range(2016, 2027)
            ]
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .sort("start")
    )
    print(
        t,
        "bar sources:",
        m["source"].value_counts().sort("count", descending=True).rows()[:3],
    )
    rows = []
    for (day,), g in m.group_by("day", maintain_order=True):
        s = SESS.get(day)
        if s is None:
            continue
        n = int((s.close - s.open).total_seconds() // 60)
        part = g.filter(pl.col("start") <= s.open + timedelta(minutes=n - 31))
        if not part.height:
            continue
        rows.append(
            {
                "day": day,
                "open": g["open"][0],
                "close": g["close"][-1],
                "c1529": part["close"][-1],
                "last_start": g["start"][-1],
                "close_time": s.close,
                "last_source": g["source"][-1],
            }
        )
    d = (
        pl.DataFrame(rows)
        .sort("day")
        .with_columns(
            pl.col("day").shift(-1).alias("next_day"),
            pl.col("open").shift(-1).alias("next_open"),
        )
    )
    d = d.filter(
        pl.Series(
            [
                PREV.get(b) == a if b is not None else False
                for a, b in zip(d["day"], d["next_day"])
            ]
        )
    )
    d = d.with_columns(
        (pl.col("next_open") / pl.col("close") - 1).alias("overnight"),
        (pl.col("close") / pl.col("c1529") - 1).alias("last30"),
    )
    for y in range(2016, 2027):
        p = d.filter(pl.col("day").dt.year() == y)
        b, tt = ols(p["overnight"].to_numpy(), p["last30"].to_numpy())
        print(
            f"  {y}: slope {b:+.3f}, t {tt:+5.2f}, nights {p.height}, last bar's source {p['last_source'].value_counts().sort('count', descending=True).rows()[:2]}"
        )
