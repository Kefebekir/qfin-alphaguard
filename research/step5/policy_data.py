"""Intraday pilot: the data for learned trading policies, from every stored ETF.

For each of the 16 stored ETFs (QQQ, SPY, IWM, DIA, TLT, GLD, HYG and the
nine sector SPDRs) and each session from its 16th, this writes the moments
at which the noise-area rule decides (HH:00 and HH:30 from 10:00), as
qqq_halfhour.py does for QQQ:

- decisions.parquet, one row per decision: the rule's position there, what
  the rule and a model can see at the end of the minute before (in units of
  the band's width or of the 14-day volatility), the day's size
  min(2, 2% / 14-day volatility), and the return from this decision's fill
  to the next one's. A decision fills at the open of its minute, as the
  engine's next-bar broker fills it; the day's last position is closed at
  the open of the session's last minute, as the strategy closes it.
- paths.npz, the day as 5-minute steps, for models that read the day as a
  sequence: each step's return, volume against the same step's 14-day
  average, and the price against the band's edges and VWAP at the step's
  end, all known at that end. A decision at minute t reads steps before t
  only.

Prices are as traded. A session whose open is more than 20% from the last
close (a split) and the 15 after it are left out, so no move or volatility
mixes the two sides of a split. So is a session whose stored day before is
not the session before it (a stock missing a year's minutes), and the 15
after it.

OFFSET (minutes, default 0) shifts the decision grid: 15 decides at 10:15,
10:45 and so on, to measure the rule's timing luck. TICKERS (comma
separated) limits the ETFs.

Usage: policy_data.py ROOT OUT_DIR
"""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TICKERS = [
    "QQQ",
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "HYG",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]
LOOKBACK = 14
CAP, TARGET_VOL = 2.0, 0.02
STEP, STEPS = 5, 78  # 5-minute steps in a full session
OFFSET = int(os.environ.get("OFFSET", "0"))
if os.environ.get("TICKERS"):
    TICKERS = os.environ["TICKERS"].split(",")
ALL_SESSIONS = sessions(date(2016, 1, 1), date(2026, 12, 31))
OPENS = {s.day: s.open for s in ALL_SESSIONS}
PREVIOUS = {b.day: a.day for a, b in pairwise(ALL_SESSIONS)}


def forward(values: np.ndarray) -> np.ndarray:
    """Each missing minute takes the last price before it (NaN, not null)."""
    return pl.Series(values).fill_nan(None).fill_null(strategy="forward").to_numpy()


def build(ticker: str):
    bars = (
        pl.concat(
            [
                pl.read_parquet(intraday_path(ROOT, ticker, y)).select(
                    "start", "open", "high", "low", "close", "volume"
                )
                for y in range(2016, 2027)
                if intraday_path(ROOT, ticker, y).exists()
            ]
        )
        .with_columns(pl.col("start").dt.date().alias("day"))
        .sort("start")
    )
    days = []
    for (day,), g in bars.group_by("day", maintain_order=True):
        start = OPENS.get(day)
        if start is None:
            continue
        k = (g["start"] - start).dt.total_minutes().to_numpy().astype(int)
        keep = k >= 0
        days.append(
            {"day": day, "k": k[keep]}
            | {
                c: g[c].to_numpy()[keep].astype(float)
                for c in ("open", "high", "low", "close", "volume")
            }
        )
    moves, volumes, closes = [], [], [d["close"][-1] for d in days]
    rows, paths, path_days, skip_until = [], [], [], -1
    for i, d in enumerate(days):
        n = int(d["k"].max()) + 1
        price = np.full(n, np.nan)
        price[d["k"]] = d["close"]
        price = forward(price)
        opens = np.full(n, np.nan)
        opens[d["k"]] = d["open"]
        volume = np.zeros(n)
        volume[d["k"]] = d["volume"]
        cum_volume = np.cumsum(volume)
        o = d["open"][0]
        move = np.abs(price / o - 1)
        split = i > 0 and abs(math.log(o / closes[i - 1])) > math.log(1.2)
        gap = i > 0 and PREVIOUS.get(d["day"]) != days[i - 1]["day"]
        if split or gap:
            skip_until = i + LOOKBACK + 1  # start the history again
            moves, volumes = [], []
        if i < LOOKBACK + 1 or i <= skip_until or len(moves) < LOOKBACK:
            moves.append(move)
            volumes.append(cum_volume)
            continue
        past = moves[-LOOKBACK:]
        sigma = np.array(
            [np.mean([p[m] for p in past if m < len(p)]) for m in range(n)]
        )
        usual = np.array(
            [
                np.mean([v[m] for v in volumes[-LOOKBACK:] if m < len(v)])
                for m in range(n)
            ]
        )
        prev_close = closes[i - 1]
        upper = max(o, prev_close) * (1 + sigma)
        lower = min(o, prev_close) * (1 - sigma)
        width = o * sigma
        typical = (d["high"] + d["low"] + d["close"]) / 3
        vwap = np.full(n, np.nan)
        vwap[d["k"]] = np.cumsum(typical * d["volume"]) / np.maximum(
            np.cumsum(d["volume"]), 1
        )
        vwap = forward(vwap)
        daily = np.diff(np.log(closes[i - LOOKBACK - 1 : i]))
        vol14 = float(daily.std(ddof=1))
        log_price = np.log(price)
        times = list(range(30 + OFFSET, n - 1, 30))
        fills = [opens[t] if not np.isnan(opens[t]) else price[t - 1] for t in times]
        last = n - 1
        fills.append(opens[last] if not np.isnan(opens[last]) else price[last - 1])
        size = min(CAP, TARGET_VOL / vol14)
        pos = 0
        for j, t in enumerate(times):
            p = price[t - 1]
            if pos == 0:
                pos = 1 if p > upper[t - 1] else (-1 if p < lower[t - 1] else 0)
            elif pos == 1 and p < max(upper[t - 1], vwap[t - 1]):
                pos = -1 if p < lower[t - 1] else 0
            elif pos == -1 and p > min(lower[t - 1], vwap[t - 1]):
                pos = 1 if p > upper[t - 1] else 0
            w = width[t - 1]
            rows.append(
                {
                    "ticker": ticker,
                    "day": d["day"],
                    "t": t,
                    "n": n,
                    "rule": pos,
                    "ret": fills[j + 1] / fills[j] - 1,
                    "size": size,
                    "price": o,
                    "vol14": vol14,
                    "above_upper": (p - upper[t - 1]) / w,
                    "below_lower": (p - lower[t - 1]) / w,
                    "from_vwap": (p - vwap[t - 1]) / w,
                    "from_open": (p / o - 1) / vol14,
                    "last_30": (p / price[t - 31] - 1 if t > 30 else p / o - 1) / vol14,
                    "gap": (o / prev_close - 1) / vol14,
                    "time": t / 390,
                    "minutes_left": (n - t) / 390,
                    "volume_so_far": math.log(
                        max(cum_volume[t - 1], 1) / max(usual[t - 1], 1)
                    ),
                    "vol_so_far": float(
                        np.sqrt(np.sum(np.diff(log_price[:t]) ** 2) * n / (t - 1))
                        / vol14
                    ),
                    "sigma": sigma[t - 1] / vol14,
                    "yesterday": math.log(closes[i - 1] / closes[i - 2]) / vol14,
                }
            )
        # The day in 5-minute steps; step s ends at minute 5s + 4.
        path = np.zeros((STEPS, 5), dtype=np.float32)
        unit = vol14 / math.sqrt(STEPS)
        for s in range(n // STEP):
            end = STEP * s + STEP - 1
            before = log_price[end - STEP] if s else math.log(o)
            spent = cum_volume[end] - (cum_volume[end - STEP] if s else 0.0)
            usual_spent = usual[end] - (usual[end - STEP] if s else 0.0)
            path[s] = (
                (log_price[end] - before) / unit,
                math.log(max(spent, 1) / max(usual_spent, 1)),
                (price[end] - upper[end]) / width[end],
                (price[end] - lower[end]) / width[end],
                (price[end] - vwap[end]) / width[end],
            )
        paths.append(path)
        path_days.append(d["day"])
        moves.append(move)
        volumes.append(cum_volume)
    frame = pl.DataFrame(rows)
    return ticker, frame, np.stack(paths), np.array(path_days)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        built = list(pool.map(build, TICKERS))
    frames, arrays = [], {}
    for ticker, frame, paths, path_days in built:
        floats = frame.select(pl.col(pl.Float64))
        bad = [c for c in floats.columns if floats[c].is_nan().any()]
        print(
            f"{ticker}: {frame['day'].n_unique()} days, {frame.height} decisions, "
            f"from {frame['day'].min()} to {frame['day'].max()}, "
            f"rule in the market {np.mean(frame['rule'].to_numpy() != 0):.0%}; "
            f"columns with NaN: {bad}, paths with NaN: {int(np.isnan(paths).sum())}"
        )
        frames.append(frame)
        arrays[f"{ticker}_paths"] = paths
        arrays[f"{ticker}_days"] = np.array([d.isoformat() for d in path_days])
    pl.concat(frames).write_parquet(OUT / "decisions.parquet")
    np.savez_compressed(OUT / "paths.npz", **arrays)
    print(f"written to {OUT}")
