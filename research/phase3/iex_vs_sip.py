"""Phase 3's first question: can the rule decide on Alpaca's free live feed?

Alpaca's free plan streams IEX's trades only (about 1-3% of QQQ's volume); the research and
the backtest used the consolidated tape (SIP). Decision 0008 planned to trade on the free feed
and compare it with the tape every evening. Alpaca keeps IEX's minute bars back to 2021, so
the question can be answered on history first:

- SIP: the rule decides on SIP's minutes and fills at SIP's prices (the backtest);
- IEX: the rule decides on IEX's minutes (its noise band, VWAP and volatility all from them)
  and fills at SIP's prices: the live system on the free feed, whose orders fill in the
  whole market.

Both from 30,000 USD at 1x per unit of size, flat at the last minute's open, 0.4 bps a unit
traded. Reported: how often the two hold different positions at a decision, how many days
differ at all, and what each earns. Check first: with SIP's minutes for both, the code must
give session_rule.py's days exactly.

IEX's minutes are downloaded once to data/raw/intraday_iex/QQQ/<year>.parquet.

Usage: iex_vs_sip.py ROOT   (with ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY set)
"""

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "futures"))
from session_rule import CAP, LOOKBACK, TARGET_VOL, Session, _filled, simulate

from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.intraday import intraday_path, regular_session
from qfin_alphaguard.momentum import daily_volatility, next_position, noise_band
from qfin_alphaguard.sessions import sessions

ROOT = Path(sys.argv[1])
FIRST, LAST = date(2021, 1, 4), date(2026, 10, 9)
COST = 0.4e-4


def iex_minutes(year: int) -> pl.DataFrame:
    path = ROOT / "intraday_iex" / "QQQ" / f"{year}.parquet"
    if not path.exists():
        client = AlpacaClient(*api_keys())
        start, end = max(date(year, 1, 1), FIRST), min(date(year, 12, 31), LAST)
        bars = regular_session(client.minute_bars("QQQ", start, end, feed="iex"))
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.write_parquet(path)
    return pl.read_parquet(path)


def sip_minutes(year: int) -> pl.DataFrame:
    return pl.read_parquet(intraday_path(ROOT, "QQQ", year))


calendar = {s.day: s for s in sessions(FIRST, LAST)}


def as_sessions(minutes: pl.DataFrame) -> dict:
    out = {}
    frame = minutes.with_columns(pl.col("start").dt.date().alias("day")).sort("start")
    for (day,), g in frame.group_by("day", maintain_order=True):
        s = calendar.get(day)
        if s is None:
            continue
        minute = (g["start"] - s.open).dt.total_minutes().to_numpy().astype(int)
        keep = (minute >= 0) & (minute < int((s.close - s.open).total_seconds() // 60))
        if not keep.any():
            continue
        length = int((s.close - s.open).total_seconds() // 60)
        out[day] = Session(
            day=day,
            minute=minute[keep],
            open=g["open"].to_numpy()[keep],
            high=g["high"].to_numpy()[keep],
            low=g["low"].to_numpy()[keep],
            close=g["close"].to_numpy()[keep],
            volume=g["volume"].to_numpy()[keep].astype(float),
            length=length,
            decisions=list(range(30, length, 30)),
        )
    return out


def run(decide: list[Session], fill: dict) -> list[dict]:
    """session_rule.simulate's rule on `decide`'s minutes, filled at `fill`'s prices."""
    moves, ends, returns, out = [], [], [], []
    for i, s in enumerate(decide):
        n = s.length
        price = _filled(n, s.minute, s.close)
        day_open = float(s.open[0])
        end = float(price[~np.isnan(price)][-1])
        if i >= LOOKBACK + 1 and len(returns) >= LOOKBACK:
            typical = (s.high + s.low + s.close) / 3
            vwap = _filled(
                n,
                s.minute,
                np.cumsum(typical * s.volume) / np.maximum(np.cumsum(s.volume), 1),
            )
            levels = np.exp(np.concatenate([[0.0], np.cumsum(returns[-LOOKBACK:])]))
            size = min(CAP, TARGET_VOL / daily_volatility(list(levels)))
            held, path = 0, []
            for t in s.decisions:
                lower, upper = noise_band(moves[-LOOKBACK:], t - 1, day_open, ends[-1])
                held = next_position(held, price[t - 1], lower, upper, vwap[t - 1])
                path.append((t, held))
            f = fill[s.day]
            f_price = _filled(f.length, f.minute, f.close)
            f_open = np.full(f.length, np.nan)
            f_open[f.minute] = f.open
            gross, traded, holding, entry = 0.0, 0.0, 0, None
            for t, want in [*path, (n - 1, 0)]:
                fill_price = f_open[t] if not math.isnan(f_open[t]) else f_price[t - 1]
                if want != holding:
                    if holding != 0:
                        gross += holding * (fill_price / entry - 1)
                    traded += abs(want - holding)
                    holding, entry = want, fill_price
            out.append(
                {
                    "day": s.day,
                    "path": [p for _, p in path],
                    "gross": gross,
                    "traded": traded,
                    "size": size,
                    "net": size * (gross - traded * COST),
                }
            )
        if i > 0:
            returns.append(math.log(end / ends[-1]))
        moves.append(list(np.abs(price / day_open - 1)))
        ends.append(end)
    return out


years = range(FIRST.year, LAST.year + 1)
sip = as_sessions(pl.concat([sip_minutes(y) for y in years]))
iex = as_sessions(pl.concat([iex_minutes(y) for y in years]))
days = sorted(sip.keys() & iex.keys())
print(
    f"{len(days)} sessions in both, {FIRST} to {LAST}; IEX minutes with a trade: "
    f"{sum(len(iex[d].minute) for d in days) / sum(sip[d].length for d in days):.1%} of the session's"
)

# The check: SIP for both must give session_rule.simulate's days exactly.
ours = run([sip[d] for d in days], sip)
theirs = simulate([sip[d] for d in days], exit_at="last_open")
assert len(ours) == len(theirs)
worst = max(abs(a["gross"] - b.gross) for a, b in zip(ours, theirs, strict=True))
print(f"check against session_rule.simulate: largest difference {worst:.1e}")
assert worst < 1e-12

live = run([iex[d] for d in days], sip)
s, i = pl.DataFrame(ours), pl.DataFrame(live)
same_points = sum(
    sum(x == y for x, y in zip(a, b, strict=True))
    for a, b in zip(s["path"], i["path"], strict=True)
)
points = sum(len(a) for a in s["path"])
same_days = sum(
    a == b for a, b in zip(s["path"].to_list(), i["path"].to_list(), strict=True)
)
print(
    f"decisions with the same position: {same_points / points:.1%} of {points:,}; "
    f"days with every decision the same: {same_days / len(s):.1%} of {len(s)}"
)
for name, frame in (
    ("SIP decides (the backtest)", s),
    ("IEX decides, fills at SIP", i),
):
    r = frame["net"].to_numpy()
    print(
        f"{name}: {r.mean() * 252:+.1%} a year, Sharpe {r.mean() / r.std(ddof=1) * math.sqrt(252):.2f}"
    )
d = (i["net"] - s["net"]).to_numpy()
print(
    f"IEX less SIP: {d.mean() * 1e4:+.2f} bps a day (t {d.mean() / d.std(ddof=1) * math.sqrt(len(d)):+.2f}); "
    f"correlation of the daily results {np.corrcoef(s['net'], i['net'])[0, 1]:.3f}"
)
for year in years:
    k = [x.year == year for x in s["day"]]
    a, b = (
        s.filter(pl.Series(k))["net"].to_numpy(),
        i.filter(pl.Series(k))["net"].to_numpy(),
    )
    if len(a):
        print(f"   {year}: SIP {a.mean() * 252:+.1%}, IEX {b.mean() * 252:+.1%} a year")
