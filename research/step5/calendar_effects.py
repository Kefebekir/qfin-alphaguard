"""Known calendar effects on QQQ and SPY, 2016-2026: do they survive after publication?

From the stored 1-minute bars: each session's open (first minute's open), close
(last minute's close) and the 14:00 price. Returns are split into overnight
(last close to open) and intraday (open to close). Dividends are not added
(about 0.6% a year for QQQ, 1.3% for SPY, all overnight on ex-dates).

- overnight versus intraday (Cliff, Cooper and Gulen 2008; Kelly and Clark 2011;
  Lou, Polk and Skouras 2019);
- the turn of the month: the last session of a month and the first three of the
  next (Lakonishok and Smidt 1988; Etula, Rinne, Suominen and Vaittinen 2020);
- the pre-FOMC drift: from the last close to 14:00 on announcement days (Lucca
  and Moench 2015; Kurov, Wolfe and Gilbert 2021);
- the correlation of each with the noise-area rule's daily result.

Each in 2016-2019 and 2020-2026 (after most of the papers' samples end).
"""

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

SESS = {s.day: s for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}


def daily(ticker: str) -> pl.DataFrame:
    bars = (
        pl.concat(
            [
                pl.read_parquet(intraday_path(Path("data/raw"), ticker, y)).select(
                    "start", "open", "close"
                )
                for y in range(2016, 2027)
            ]
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .sort("start")
    )
    rows = []
    for (day,), g in bars.group_by("day", maintain_order=True):
        s = SESS.get(day)
        if s is None:
            continue
        g = g.filter(pl.col("start") >= s.open)
        if g.height == 0:
            continue
        at_14 = g.filter(
            pl.col("start") <= s.open + timedelta(minutes=269)
        )  # the 13:59 bar
        rows.append(
            {
                "day": day,
                "open": g["open"][0],
                "close": g["close"][-1],
                "p14": at_14["close"][-1] if at_14.height else None,
            }
        )
    d = pl.DataFrame(rows).sort("day")
    return d.with_columns(
        (pl.col("open") / pl.col("close").shift(1) - 1).alias("overnight"),
        (pl.col("close") / pl.col("open") - 1).alias("intraday"),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("total"),
        (pl.col("p14") / pl.col("close").shift(1) - 1).alias("to_14"),
    ).drop_nulls("overnight")


def describe(name: str, r: np.ndarray, per_year: float = 252) -> str:
    n = len(r)
    if n < 3:
        return f"{name}: n {n}"
    m = r.mean()
    t = m / r.std(ddof=1) * math.sqrt(n)
    return (
        f"{name:34} n {n:5}  mean {m * 1e4:+7.2f} bps  t {t:+5.2f}  "
        f"Sharpe {m / r.std(ddof=1) * math.sqrt(per_year):5.2f}"
    )


events = pl.read_csv("data/raw/calendar/events.csv").with_columns(
    pl.col("day").str.to_date()
)
fomc = set(events.filter(pl.col("event") == "fomc")["day"])
COST = 0.3e-4 + 0.0000206 / 2
dec = pl.read_parquet("data/intraday_pilot/policy/decisions.parquet").filter(
    pl.col("ticker") == "QQQ"
)
dec = dec.sort("day", "t").with_columns(
    pl.col("rule").shift(1).over("day").fill_null(0).alias("before")
)
last = pl.col("t") == pl.col("t").max().over("day")
dec = dec.with_columns(
    (
        pl.col("size") * pl.col("rule") * pl.col("ret")
        - pl.col("size")
        * COST
        * (
            (pl.col("rule") - pl.col("before")).abs()
            + pl.when(last).then(pl.col("rule").abs()).otherwise(0)
        )
    ).alias("net")
)
rule = dec.group_by("day").agg(pl.col("net").sum().alias("rule")).sort("day")

for ticker in ("QQQ", "SPY"):
    d = daily(ticker).join(rule, on="day", how="left").sort("day")
    month = d["day"].dt.strftime("%Y-%m")
    d = d.with_columns(month.alias("month"))
    d = d.with_columns(
        pl.col("day").rank("ordinal").over("month").alias("nth"),
        pl.len().over("month").alias("days_in_month"),
    )
    tom = (d["nth"] == d["days_in_month"]) | (d["nth"] <= 3)
    d = d.with_columns(pl.Series("tom", tom))
    print(f"\n=== {ticker} ({d['day'].min()} to {d['day'].max()})")
    for label, part in (
        ("2016-2019", d.filter(pl.col("day").dt.year() <= 2019)),
        ("2020-2026", d.filter(pl.col("day").dt.year() >= 2020)),
    ):
        print(f"-- {label}")
        for col in ("total", "overnight", "intraday"):
            print("  " + describe(col, part[col].to_numpy()))
        on = part.filter(pl.col("tom"))
        off = part.filter(~pl.col("tom"))
        print(
            "  " + describe("turn of the month, close to close", on["total"].to_numpy())
        )
        print("  " + describe("other days", off["total"].to_numpy()))
        f = part.filter(pl.col("day").is_in(list(fomc))).drop_nulls("to_14")
        print("  " + describe("FOMC day, last close to 14:00", f["to_14"].to_numpy()))
        both = part.drop_nulls("rule")
        print(
            f"  correlation with the QQQ rule's day: overnight "
            f"{np.corrcoef(both['overnight'], both['rule'])[0, 1]:+.2f}, intraday "
            f"{np.corrcoef(both['intraday'], both['rule'])[0, 1]:+.2f}"
        )
