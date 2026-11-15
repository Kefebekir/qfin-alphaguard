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
"""

import numpy as np

HORIZONS = (5, 15, 30, 60)  # minutes


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
