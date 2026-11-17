"""What happened after each minute of a session: the labels KUTBIT learns from (decision 0023).

A decision at the close of bar t is filled at the open of bar t + 1, never earlier. For each
horizon h, the position is held to the close of bar t + h, or to the session's last bar if
that comes first, since KUTBIT is flat at every close. Three things are recorded for a long
position, each relative to the entry price:
- `ret`: the return at the exit;
- `worst`: the lowest low along the way, from the entry bar to the exit bar;
- `best`: the highest high along the way.
A short position's return is -ret, its worst moment is `best` and its best is `worst`. At 10x
the worst moment along the way, not the exit, is what closes an account.

The session's last bar has no labels, since nothing can be filled after it.

`targets` adds what the forecaster learns from them:
- `apart`: the return with the market's part taken out, ret - beta * QQQ's ret over the same
  bars: a long position hedged with QQQ at the beta known before the open;
- `scale`: the usual size of a move over the bars held, from the instrument's usual move at
  each of those minutes (features.Day.swing), also known before the open;
- `z`: apart over scale. The open and midday, a quiet stock and a jumpy one, are then in the
  same units, so the forecaster's work is the direction and what today adds to the usual.
"""

import numpy as np

HORIZONS = (1, 5, 15, 30, 60)  # minutes


def labels(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, dict[str, np.ndarray]]:
    """For each horizon, arrays over the session's minutes: ret, worst, best and cut (True
    where the close came before the horizon). The last minute's are NaN and cut."""
    n = len(open_)
    t = np.arange(n - 1)
    out = {}
    for h in horizons:
        ret, worst, best = (np.full(n, np.nan) for _ in range(3))
        cut = np.ones(n, dtype=bool)
        if n > 1:
            entry = open_[1:]
            # bars t + 1 .. t + h; past the last bar, padding that never wins a min or max
            lows = np.concatenate([low[1:], np.full(h - 1, np.inf)])
            highs = np.concatenate([high[1:], np.full(h - 1, -np.inf)])
            window = np.lib.stride_tricks.sliding_window_view
            stop = np.minimum(t + h, n - 1)
            ret[:-1] = close[stop] / entry - 1
            worst[:-1] = window(lows, h).min(axis=1) / entry - 1
            best[:-1] = window(highs, h).max(axis=1) / entry - 1
            cut[:-1] = t + h > n - 1
        out[h] = {"ret": ret, "worst": worst, "best": best, "cut": cut}
    return out


def targets(day, market, horizons: tuple[int, ...] = HORIZONS):
    """For each horizon, the labels of `day` (a features.Day) with apart, scale and z added;
    `market` is QQQ's Day of the same session."""
    if len(market.open) != len(day.open):
        raise ValueError("the market's day must cover the same minutes")
    own = labels(day.open, day.high, day.low, day.close, horizons)
    of_market = labels(market.open, market.high, market.low, market.close, horizons)
    n = len(day.open)
    t = np.arange(n - 1)
    squares = np.concatenate([[0.0], np.cumsum(np.asarray(day.swing) ** 2)])
    for h in horizons:
        stop = np.minimum(t + h, n - 1)
        scale = np.full(n, np.nan)
        scale[:-1] = np.sqrt(squares[stop + 1] - squares[t + 1])  # bars t + 1 .. stop
        apart = own[h]["ret"] - day.beta * of_market[h]["ret"]
        own[h].update(apart=apart, scale=scale, z=apart / scale)
    return own
