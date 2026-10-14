"""The day book: the QQQ strategy's choices, day by day, from its decision log.

The engine runs momentum.py's NoiseAreaMomentum on the stored QQQ minutes
from 30,000 USD, at Alpaca's fees and QQQ's measured half spread (0.3 bps,
engine_check.py), with a stand-in for Guard that sends every order: Guard's
limits for this strategy are Phase 2's step 4, and nothing here trades. Its
decision log is stored under LOG (decision_log.py) and read back into a page:
the days with their results after costs, and for each day its chart (price,
noise band, VWAP, fills) and every decision with the reason the strategy
wrote for it.

The band on the chart is the strategy's own noise_band at every minute, from
the history the strategy keeps.

The page holds prices from licensed data, so it is written under data/,
which git ignores, and is for personal use only: do not publish it.

Usage: daily_choices.py ROOT FIRST OUT.html LOG
"""

import json
import math
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import SessionClose, run_backtest
from qfin_alphaguard.data.intraday import intraday_path, minute_feed
from qfin_alphaguard.decision_log import read_decision_log, write_decision_log
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.momentum import NoiseAreaMomentum, noise_band
from qfin_alphaguard.sessions import sessions

ROOT, FIRST, OUT = Path(sys.argv[1]), date.fromisoformat(sys.argv[2]), Path(sys.argv[3])
LOG = Path(sys.argv[4])
CASH = 30_000.0
NEW_YORK = ZoneInfo("America/New_York")
POSITION = {"long": 1, "flat": 0, "short": -1}


class SendsEverything:
    """Not Guard: lets every intent through, for this replay only."""

    def check(self, intent, view):
        return GuardDecision(intent)


bars = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(ROOT, "QQQ", year)).select(
                "start", "open", "high", "low", "close", "volume"
            )
            for year in range(2016, 2027)
            if intraday_path(ROOT, "QQQ", year).exists()
        ]
    )
    .with_columns(pl.col("start").dt.date().alias("day"))
    .sort("start")
)
first, last = bars["day"].min(), bars["day"].max()
result = run_backtest(
    minute_feed(
        ROOT, {year: ["QQQ"] for year in range(first.year, last.year + 1)}, first, last
    ),
    NoiseAreaMomentum(),
    NextBarBroker(CostModel(half_spread_bps=0.3)),
    SendsEverything(),
    CASH,
)
write_decision_log(result.log, LOG)
notes, fills = read_decision_log(LOG)  # the page reads the stored log
closes = [record for record in result.log if isinstance(record, SessionClose)]
value_before = dict(
    zip([c.day for c in closes], [CASH] + [c.value for c in closes[:-1]], strict=True)
)
value_after = {c.day: c.value for c in closes}
sessions_by_day = {s.day: s for s in sessions(first, last)}


def minutes_since_open(times: pl.Series, open_at) -> list[int]:
    return [int((t - open_at).total_seconds() // 60) for t in times]


def number(value) -> float | None:
    """A float for the page, or None where JSON has no NaN."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 3)


notes = notes.with_columns(pl.col("time").dt.date().alias("day"))
fills = fills.with_columns(pl.col("time").dt.date().alias("day"))
notes_by_day = {day: frame for (day,), frame in notes.group_by("day")}
fills_by_day = {day: frame for (day,), frame in fills.group_by("day")}

# A tracker with the strategy's own memory, to draw its band at every minute.
tracker = NoiseAreaMomentum()
days = []
for (day,), g in bars.group_by("day", maintain_order=True):
    session = sessions_by_day[day]
    rows = list(g.iter_rows(named=True))
    day_open = rows[0]["open"]
    ready = len(tracker.moves) == tracker.moves.maxlen
    n = int((rows[-1]["start"] - session.open).total_seconds() // 60) + 1
    if day >= FIRST and ready:
        prev_close = tracker.closes[-1]
        band = [
            noise_band(list(tracker.moves), m, day_open, prev_close) for m in range(n)
        ]
        typical = (g["high"] + g["low"] + g["close"]) / 3
        vwap = (typical * g["volume"]).cum_sum() / g["volume"].cum_sum()
        minute_of = minutes_since_open(g["start"], session.open)
        price = dict(zip(minute_of, g["close"].to_list(), strict=True))
        vwap_at = dict(zip(minute_of, vwap.to_list(), strict=True))
        step, last_price, last_vwap, series = 5, math.nan, math.nan, []
        for m in range(n):
            last_price = price.get(m, last_price)
            last_vwap = vwap_at.get(m, last_vwap)
            if m % step == 0:
                series.append((m, last_price, band[m][0], band[m][1], last_vwap))
        day_notes = notes_by_day.get(day, notes.clear())
        day_fills = fills_by_day.get(day, fills.clear())
        fill_minutes = minutes_since_open(day_fills["time"], session.open)
        fill_prices = day_fills["price"].to_list()
        held, marks = 0.0, []  # each fill, and the side it leaves the position on
        for minute, row in zip(
            fill_minutes, day_fills.iter_rows(named=True), strict=True
        ):
            held = round(
                held + (row["quantity"] if row["side"] == "buy" else -row["quantity"]),
                9,
            )
            after = 1 if held > 0 else (-1 if held < 0 else 0)
            marks.append({"t": minute, "price": number(row["price"]), "after": after})
        decisions = []
        for row in day_notes.iter_rows(named=True):
            minute = int((row["time"] - session.open).total_seconds() // 60)
            fill = next(
                (
                    p
                    for t, p in zip(fill_minutes, fill_prices, strict=True)
                    if t == minute
                ),
                None,
            )
            decisions.append(
                {
                    "t": minute,
                    "price": number(row["price"]),
                    "upper": number(row.get("upper")),
                    "lower": number(row.get("lower")),
                    "vwap": number(row.get("vwap")),
                    "position": POSITION[row["choice"]],
                    "fill": number(fill),
                    "why": row["reason"],
                }
            )
        entry = next(
            (r for r in day_notes.iter_rows(named=True) if r.get("shares")), None
        )
        size = entry["shares"] * entry["price"] / entry["account"] if entry else 0.0
        days.append(
            {
                "day": day.isoformat(),
                "open_et": session.open.astimezone(NEW_YORK).strftime("%H:%M"),
                "net": value_after[day] / value_before[day] - 1,
                "fees": round(float(day_fills["commission"].sum()), 2),
                "trades": day_fills.height,
                "size": round(abs(size), 2),
                "prev_close": number(prev_close),
                "minutes": [m for m, *_ in series],
                "price": [number(v) for _, v, *_ in series],
                "lower": [number(v) for _, _, v, *_ in series],
                "upper": [number(v) for *_, v, _ in series],
                "vwap": [number(v) for *_, v in series],
                "decisions": decisions,
                "fills": marks,
            }
        )
    tracker.prices, tracker.day_open, tracker.flow, tracker.volume = {}, None, 0.0, 0
    view = SimpleNamespace(session=session)
    for row in rows:
        tracker.record(SimpleNamespace(**row), view)
    tracker.on_session_end(view)

OUT.parent.mkdir(parents=True, exist_ok=True)
page = Path(__file__).with_name("daily_choices.html").read_text()
OUT.write_text(page.replace("/*DATA*/null", json.dumps(days, separators=(",", ":"))))
r = np.array([d["net"] for d in days])
print(
    f"{len(days)} days from {days[0]['day']} to {days[-1]['day']}: "
    f"{np.prod(1 + r) - 1:+.1%} in all after costs, Sharpe "
    f"{r.mean() / r.std(ddof=1) * math.sqrt(252):.2f}, {np.mean(r > 0):.0%} of days won, "
    f"{np.mean([d['trades'] > 0 for d in days]):.0%} of days traded; "
    f"{notes.height} notes and {fills.height} fills in {LOG}; written to {OUT}"
)
