"""The QQQ rule on any market's session, with momentum.py's own rule functions (F1).

A session is its minutes since the open (a minute with no trade carries the last price), its
length from the exchange's calendar, the minutes at which the rule decides, and the contract
it trades. For each session from the 16th:

- the noise band at the decision's last minute, from the 14 sessions before
  (momentum.noise_band), around the session's first open and the previous session's last
  price;
- VWAP of the session so far: the typical price (high + low + close) / 3, weighted by volume;
- the decision (momentum.next_position), filled at the next minute's open, or the last
  price if that minute has no trade;
- the position closed at the end: at the last minute's open (the engine, and F1) or at the
  last price (research/intraday_pilot/qqq_momentum.py, for the check on QQQ);
- the size min(2, 2% / momentum.daily_volatility of the last 15 session-end prices), with a
  session whose contract differs from the session before's not traded and its return
  left out of later volatility windows.

Returns are of a position worth 1x the account; `traded` is the account's worth traded, so
costs are traded * size * cost per unit traded.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qfin_alphaguard.momentum import daily_volatility, next_position, noise_band

LOOKBACK = 14
TARGET_VOL = 0.02
CAP = 2.0


@dataclass(frozen=True)
class Session:
    day: object  # the session's date
    minute: np.ndarray  # each bar's minutes since the open, increasing, >= 0
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    length: int  # minutes from the open to the close, from the calendar
    decisions: Sequence[int]  # minutes since the open at which the rule decides
    contract: object = None  # the contract traded; None for one instrument throughout


@dataclass(frozen=True)
class Day:
    day: object
    gross: float  # the day's return of a 1x position, before costs
    traded: float  # the account's worth traded, at 1x
    size: float  # min(2, 2% / volatility); 0 on a session not traded
    price: float  # the session's first open
    traded_today: bool  # whether the rule held a position at any time


def _filled(n: int, minute: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Values by minute 0..n-1, a minute without a bar carrying the last one (NaN before)."""
    out = np.full(n, np.nan)
    inside = minute < n
    out[minute[inside]] = values[inside]
    last = np.nan
    for i in range(n):
        if math.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def simulate(sessions: Sequence[Session], exit_at: str = "last_open") -> list[Day]:
    """Each session's result from the 16th, in order."""
    if exit_at not in ("last_open", "last_price"):
        raise ValueError(f"exit_at must be last_open or last_price, got {exit_at!r}")
    moves: list[list[float]] = []
    ends: list[float] = []  # each session's last price
    returns: list[float] = []  # log returns between sessions of the same contract
    out = []
    for i, s in enumerate(sessions):
        n = s.length
        price = _filled(n, s.minute, s.close)
        opens = np.full(n, np.nan)
        opens[s.minute[s.minute < n]] = s.open[s.minute < n]
        day_open = float(s.open[0])
        move = list(np.abs(price / day_open - 1))
        end = float(price[~np.isnan(price)][-1])
        same_contract = i > 0 and s.contract == sessions[i - 1].contract
        if i >= LOOKBACK + 1 and len(returns) >= LOOKBACK and same_contract:
            prev_close = ends[-1]
            typical = (s.high + s.low + s.close) / 3
            vwap = _filled(
                n,
                s.minute,
                np.cumsum(typical * s.volume) / np.maximum(np.cumsum(s.volume), 1),
            )
            # momentum.daily_volatility of the last 15 prices whose 14 returns these are
            levels = np.exp(np.concatenate([[0.0], np.cumsum(returns[-LOOKBACK:])]))
            size = min(CAP, TARGET_VOL / daily_volatility(list(levels)))
            held, path = 0, []
            for t in s.decisions:
                lower, upper = noise_band(
                    moves[-LOOKBACK:], t - 1, day_open, prev_close
                )
                held = next_position(held, price[t - 1], lower, upper, vwap[t - 1])
                path.append((t, held))
            gross, traded, holding, entry, ever = 0.0, 0.0, 0, None, False
            for t, want in [*path, (n - 1 if exit_at == "last_open" else n, 0)]:
                if t < n and not math.isnan(opens[t]):
                    fill = opens[t]
                else:
                    fill = price[min(t, n) - 1]
                if want != holding:
                    if holding != 0:
                        gross += holding * (fill / entry - 1)
                    traded += abs(want - holding)
                    holding, entry = want, fill
                    ever = ever or want != 0
            out.append(Day(s.day, gross, traded, size, day_open, ever))
        else:
            out.append(Day(s.day, 0.0, 0.0, 0.0, day_open, False))
        if i > 0 and same_contract:
            returns.append(math.log(end / ends[-1]))
        moves.append(move)
        ends.append(end)
    return out[LOOKBACK + 1 :]
