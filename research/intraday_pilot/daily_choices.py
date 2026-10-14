"""The QQQ strategy's choices, day by day, as a page to open in a browser.

For every session from FIRST on, the rules of qqq_momentum.py (decision
0011, cap 2x) are replayed on the stored minutes, and every decision at
HH:00 and HH:30 is kept with what it saw: the price, the noise band's edges,
VWAP, the position chosen and why, and the price it filled at. The page lists
the days with their results; a click shows the day's chart and decisions.

The page holds prices from licensed data, so it is written under data/,
which git ignores, and is for personal use only: do not publish it.

Usage: daily_choices.py data/raw FIRST OUT.html [CHECK.parquet]
CHECK is qqq_momentum.py's output; each day's result must match it.
"""

import json
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT, FIRST, OUT = Path(sys.argv[1]), date.fromisoformat(sys.argv[2]), Path(sys.argv[3])
CHECK = Path(sys.argv[4]) if len(sys.argv) > 4 else None
LOOKBACK, TARGET_VOL, CAP = 14, 0.02, 2.0
PER_SHARE, PER_VALUE = 0.001, 0.15e-4
NEW_YORK = ZoneInfo("America/New_York")

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
        {"day": day, "start": start, "k": k[keep].astype(int)}
        | {
            c: g[c].to_numpy()[keep].astype(float)
            for c in ("open", "high", "low", "close", "volume")
        }
    )


def reason(before: int, after: int, p: float, up: float, lo: float, vwap: float) -> str:
    if before == after:
        if after == 0:
            return "inside the noise band: stay flat"
        stop = max(up, vwap) if after == 1 else min(lo, vwap)
        return f"trend holds; the stop is at {stop:.2f}"
    if after == 1:
        return (
            "above the band: buy"
            if before == 0
            else "short stopped, and above the band: buy"
        )
    if after == -1:
        return (
            "below the band: sell short"
            if before == 0
            else "long stopped, and below the band: sell short"
        )
    return (
        "long stopped: price fell below the band's top or VWAP"
        if before == 1
        else ("short stopped: price rose above the band's bottom or VWAP")
    )


moves, closes, out = [], [d["close"][-1] for d in days], []
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
    sigma = np.array(
        [np.mean([p[m] for p in moves[-LOOKBACK:] if m < len(p)]) for m in range(n)]
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
    vol = np.diff(np.log(closes[i - LOOKBACK - 1 : i])).std(ddof=1)
    size = min(CAP, TARGET_VOL / vol)
    moves.append(move)
    if d["day"] < FIRST:
        continue
    pos, decisions = 0, []
    for t in range(30, n, 30):
        p, up, lo, vw = price[t - 1], upper[t - 1], lower[t - 1], vwap[t - 1]
        before = pos
        if pos == 0:
            pos = 1 if p > up else (-1 if p < lo else 0)
        elif pos == 1 and p < max(up, vw):
            pos = -1 if p < lo else 0
        elif pos == -1 and p > min(lo, vw):
            pos = 1 if p > up else 0
        fill = nxt_open[t] if t < n and not np.isnan(nxt_open[t]) else price[t - 1]
        decisions.append(
            {
                "t": t,
                "price": p,
                "upper": up,
                "lower": lo,
                "vwap": vw,
                "position": pos,
                "fill": fill if pos != before else None,
                "why": reason(before, pos, p, up, lo, vw),
            }
        )
    ret, traded, held, entry = 0.0, 0.0, 0, None
    for t, want in [(x["t"], x["position"]) for x in decisions] + [(n, 0)]:
        fill = (
            nxt_open[t] if t < n and not np.isnan(nxt_open[t]) else price[min(t, n) - 1]
        )
        if want != held:
            if held != 0:
                ret += held * (fill / entry - 1)
            traded += abs(want - held)
            held, entry = want, fill
    net = ret * size - traded * size * (PER_VALUE + PER_SHARE / o)
    step = 5  # the chart's points: every 5th minute's close
    out.append(
        {
            "day": d["day"].isoformat(),
            "open_et": d["start"].astimezone(NEW_YORK).strftime("%H:%M"),
            "size": round(size, 3),
            "net": net,
            "gross": ret * size,
            "trades": int(traded),
            "prev_close": prev_close,
            "minutes": list(range(0, n, step)),
            "price": [round(x, 3) for x in price[::step]],
            "upper": [round(x, 3) for x in upper[::step]],
            "lower": [round(x, 3) for x in lower[::step]],
            "vwap": [round(x, 3) for x in vwap[::step]],
            "decisions": [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}
                for x in decisions
            ],
        }
    )

if CHECK is not None:
    expected = dict(pl.read_parquet(CHECK).select("day", "net_2x").iter_rows())
    worst = max(abs(x["net"] - expected[date.fromisoformat(x["day"])]) for x in out)
    print(f"largest difference from {CHECK.name}: {worst:.2e}")
    assert worst < 1e-12

OUT.parent.mkdir(parents=True, exist_ok=True)
page = Path(__file__).with_name("daily_choices.html").read_text()
OUT.write_text(page.replace("/*DATA*/null", json.dumps(out, separators=(",", ":"))))
r = np.array([x["net"] for x in out])
print(
    f"{len(out)} days from {out[0]['day']} to {out[-1]['day']}: "
    f"{np.prod(1 + r) - 1:+.1%} in all, {np.mean(r > 0):.0%} of days won, "
    f"{np.mean([x['trades'] > 0 for x in out]):.0%} of days traded; written to {OUT}"
)
