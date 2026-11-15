"""What KUTBIT sees of one instrument's session at each minute (decision 0023).

The same code builds the inputs in training and, later, live, so the model sees in a
session exactly what it was trained on. The input at minute t comes only from:
- the session's bars from the open to bar t, both included;
- what was known before the open: the previous close, the volatility and the volume of the
  LOOKBACK sessions before, the VIX's last close and the headlines since the previous close.

Each input is a window of the last WINDOW minutes, one row a minute and one column a
CHANNEL, and a few SCALARS. Moves are scaled by the instrument's own volatility, so a
quiet utility and a jumpy chip maker are read in the same units.
"""

import math
from dataclasses import dataclass

import numpy as np

WINDOW = 120  # minutes of chart in each input
FULL_SESSION = 390  # minutes from 09:30 to 16:00
LOOKBACK = 20  # sessions before, for the volatility and the volume profile

CHANNELS = (
    "move",  # this minute's log return, in units of a typical minute's move
    "range",  # this minute's log high over low, in the same units
    "volume",  # log volume over the volume usual at this minute of the session
    "vwap",  # log distance of the close from the session's VWAP, in units of a day's move
    "news",  # log(1 + headlines first published this minute naming the instrument)
    "market",  # QQQ's move this minute, in QQQ's units
    "seen",  # 1 if the minute had a trade, 0 if its bar only carries the last close
)
SCALARS = (
    "time",  # minutes since the open, over a full session
    "left",  # minutes left to the close, over a full session
    "gap",  # log open over the previous close, in units of a day's move
    "since_open",  # log close over the open, in units of a day's move
    "place",  # where the close stands in the session's range so far, 0 to 1
    "day_vol",  # log of this session's typical minute move so far over the usual one
    "vix",  # log of the VIX's last close
    "news_before",  # log(1 + headlines between the previous close and the open)
    "etf",  # 1 for an ETF, 0 for a stock
)


@dataclass(frozen=True)
class Day:
    """One instrument's regular session, minute by minute, and what was known before it.

    The arrays hold the session's bars from the open, one a minute; live, they hold the
    minutes so far, so they may be shorter than `minutes`, the session's length. A minute
    without a trade carries the last close as its open, high, low and close, with no volume
    and `seen` False. Prices are as traded that day; `prev_close` and `profile` are on the
    same basis, splits taken into account.
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    seen: np.ndarray
    profile: (
        np.ndarray
    )  # the median volume at each minute over the LOOKBACK sessions before
    news: np.ndarray  # headlines first published in each minute naming the instrument
    minutes: int  # the session's length: 390, or 210 on a half day
    prev_close: float
    vol: float  # std of daily close-to-close log returns over the LOOKBACK sessions before
    vix: float
    news_before: int
    etf: bool

    def __post_init__(self) -> None:
        arrays = (
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.seen,
            self.profile,
            self.news,
        )
        n = len(self.open)
        if not 1 <= n <= self.minutes or any(len(a) != n for a in arrays):
            raise ValueError(
                f"a day needs between 1 and {self.minutes} minutes, all arrays alike"
            )
        prices = np.stack([self.open, self.high, self.low, self.close])
        if not (np.isfinite(prices).all() and (prices > 0).all()):
            raise ValueError("prices must be finite and positive")
        body_high = np.maximum(self.open, self.close)
        body_low = np.minimum(self.open, self.close)
        if (self.high < body_high).any() or (self.low > body_low).any():
            raise ValueError("a bar's open and close must lie within its low and high")
        if (np.asarray(self.volume) < 0).any() or (np.asarray(self.profile) < 0).any():
            raise ValueError("volumes cannot be negative")
        if (np.asarray(self.news) < 0).any() or self.news_before < 0:
            raise ValueError("headline counts cannot be negative")
        for name in ("prev_close", "vol", "vix"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive, got {value!r}")


def channels(day: Day, market: Day) -> np.ndarray:
    """Every minute's CHANNELS, [minutes so far, len(CHANNELS)]; row m uses bars 0..m."""
    if len(market.open) != len(day.open):
        raise ValueError("the market's day must cover the same minutes")
    minute = day.vol / math.sqrt(FULL_SESSION)
    vwap = np.cumsum((day.high + day.low + day.close) / 3 * day.volume) / np.maximum(
        np.cumsum(day.volume), 1e-12
    )
    # before the first trade, the VWAP is the bar's typical price
    traded = np.cumsum(day.volume) > 0
    vwap = np.where(traded, vwap, (day.high + day.low + day.close) / 3)
    columns = [
        moves(day) / minute,
        np.log(day.high / day.low) / minute,
        np.log1p(day.volume) - np.log1p(day.profile),
        np.log(day.close / vwap) / day.vol,
        np.log1p(day.news),
        moves(market) / (market.vol / math.sqrt(FULL_SESSION)),
        day.seen.astype(float),
    ]
    return np.stack(columns, axis=1)


def moves(day: Day) -> np.ndarray:
    """Each minute's log return: the first minute's from its open, the others' from the
    close before."""
    previous = np.concatenate([[day.open[0]], day.close[:-1]])
    return np.log(day.close / previous)


def scalars(day: Day) -> np.ndarray:
    """Every minute's SCALARS, [minutes so far, len(SCALARS)]; row m uses bars 0..m."""
    n = len(day.open)
    m = np.arange(n)
    minute = day.vol / math.sqrt(FULL_SESSION)
    high, low = np.maximum.accumulate(day.high), np.minimum.accumulate(day.low)
    span = high - low
    place = np.where(span > 0, (day.close - low) / np.where(span > 0, span, 1), 0.5)
    typical = np.sqrt(np.cumsum(moves(day) ** 2) / (m + 1)) / minute
    columns = [
        (m + 1) / FULL_SESSION,
        (day.minutes - 1 - m) / FULL_SESSION,
        np.full(n, math.log(day.open[0] / day.prev_close) / day.vol),
        np.log(day.close / day.open[0]) / day.vol,
        place,
        np.log(np.maximum(typical, 1e-3)),
        np.full(n, math.log(day.vix)),
        np.full(n, math.log1p(day.news_before)),
        np.full(n, float(day.etf)),
    ]
    return np.stack(columns, axis=1)


def inputs(day: Day, market: Day) -> tuple[np.ndarray, np.ndarray]:
    """Every minute's input: windows [minutes, WINDOW, len(CHANNELS)] and scalars
    [minutes, len(SCALARS)], as float32.

    Row t is what KUTBIT sees at the close of bar t: the last WINDOW minutes up to t, with
    the minutes before the open as zeros, which `seen` marks as unseen too.
    """
    rows = channels(day, market)
    padded = np.concatenate([np.zeros((WINDOW - 1, rows.shape[1])), rows])
    windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=0)
    return (
        np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32),
        scalars(day).astype(np.float32),
    )
