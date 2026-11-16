"""What KUTBIT sees at each minute (src/qfin_alphaguard/kutbit/features.py)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from qfin_alphaguard.kutbit import features as kf
from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW, Day, inputs

MINUTE_FIELDS = ("open", "high", "low", "close", "volume", "seen", "profile", "news")


def day(close, open_=None, **given) -> Day:
    """A session from its closes; each open is the close before unless given."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    open_ = np.asarray(
        open_ if open_ is not None else np.concatenate([[close[0]], close[:-1]]),
        dtype=float,
    )
    fields = {
        "open": open_,
        "high": np.maximum(open_, close),
        "low": np.minimum(open_, close),
        "close": close,
        "volume": np.full(n, 1000.0),
        "seen": np.ones(n, dtype=bool),
        "profile": np.full(n, 1000.0),
        "news": np.zeros(n),
        "minutes": 390,
        "prev_close": 100.0,
        "vol": 0.02,
        "vix": 20.0,
        "news_before": 0,
        "etf": False,
    }
    fields.update(
        {k: np.asarray(v) if k in MINUTE_FIELDS else v for k, v in given.items()}
    )
    return Day(**fields)


def random_day(rng, n=390) -> Day:
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 2e-4, n))
    return day(
        close,
        open_,
        high=np.maximum(open_, close) * (1 + rng.uniform(0, 5e-4, n)),
        low=np.minimum(open_, close) * (1 - rng.uniform(0, 5e-4, n)),
        volume=rng.integers(0, 5000, n).astype(float),
        profile=rng.uniform(500, 3000, n),
        news=rng.poisson(0.05, n).astype(float),
        prev_close=99.0,
        vix=18.0,
        news_before=2,
    )


@pytest.mark.parametrize(
    "broken",
    [
        {"high": [100.0, 100.0, 101.0]},  # a high below its close
        {"volume": [1.0, -1.0, 1.0]},
        {"news": [0.0, 0.0]},  # one minute short
        {"minutes": 2},  # more minutes than the session has
        {"prev_close": 0.0},
        {"vol": float("nan")},
        {"vix": -1.0},
        {"news_before": -1},
    ],
)
def test_a_day_refuses_what_cannot_be(broken):
    with pytest.raises(ValueError):
        day([100.0, 101.0, 102.0], **broken)


def test_each_channel_by_hand():
    d = day(
        [100.0, 101.0, 99.0],
        open_=[100.0, 100.5, 101.0],
        volume=[1000.0, 3000.0, 0.0],
        profile=[1000.0, 1000.0, 1000.0],
        news=[0.0, 2.0, 0.0],
        seen=[True, True, False],
    )
    market = day([400.0, 404.0, 402.0], vol=0.01)  # in its own units, not the stock's
    rows = kf.channels(d, market)
    unit = 0.02 / math.sqrt(390)
    column = {name: rows[:, i] for i, name in enumerate(CHANNELS)}
    moves = np.log([100 / 100, 101 / 100, 99 / 101])
    np.testing.assert_allclose(column["move"], moves / unit)
    np.testing.assert_allclose(
        column["market"], np.log([1, 404 / 400, 402 / 404]) / (0.01 / math.sqrt(390))
    )
    np.testing.assert_allclose(column["range"], np.log(d.high / d.low) / unit)
    np.testing.assert_allclose(
        column["volume"], np.log1p([1000, 3000, 0]) - np.log1p(1000)
    )
    typical = (d.high + d.low + d.close) / 3
    vwap = np.cumsum(typical * d.volume) / np.cumsum(d.volume)
    np.testing.assert_allclose(column["vwap"], np.log(d.close / vwap) / 0.02)
    np.testing.assert_allclose(column["news"], np.log1p([0, 2, 0]))
    assert column["seen"].tolist() == [1.0, 1.0, 0.0]


def test_before_the_first_trade_the_vwap_is_the_bar_s_typical_price():
    d = day([100.0, 102.0], volume=[0.0, 0.0])
    assert kf.channels(d, d)[:, CHANNELS.index("vwap")] == pytest.approx(
        np.log(d.close / ((d.high + d.low + d.close) / 3)) / 0.02
    )


def test_each_scalar_by_hand():
    d = day(
        [100.0, 102.0, 101.0],
        open_=[101.0, 100.0, 102.0],
        minutes=210,  # a half day
        prev_close=100.0,
        vix=25.0,
        news_before=3,
        etf=True,
    )
    s = {name: kf.scalars(d)[:, i] for i, name in enumerate(SCALARS)}
    np.testing.assert_allclose(s["time"], np.array([1, 2, 3]) / 390)
    np.testing.assert_allclose(s["left"], np.array([209, 208, 207]) / 390)
    np.testing.assert_allclose(s["gap"], math.log(101 / 100) / 0.02)
    np.testing.assert_allclose(s["since_open"], np.log(d.close / 101) / 0.02)
    # low so far 100, 100, 100; high so far 101, 102, 102
    np.testing.assert_allclose(s["place"], [0.0, 1.0, 0.5])
    np.testing.assert_allclose(s["vix"], math.log(25))
    np.testing.assert_allclose(s["news_before"], math.log1p(3))
    np.testing.assert_allclose(s["etf"], 1.0)
    unit = 0.02 / math.sqrt(390)
    moves = np.log([100 / 101, 102 / 100, 101 / 102])
    typical = np.sqrt(np.cumsum(moves**2) / [1, 2, 3]) / unit
    np.testing.assert_allclose(s["day_vol"], np.log(typical))


def test_the_window_holds_the_last_minutes_with_zeros_before_the_open():
    d = day(100 + np.arange(150) * 0.1)
    windows, scalars = inputs(d, d)
    assert windows.shape == (150, WINDOW, len(CHANNELS)) and windows.dtype == np.float32
    assert scalars.shape == (150, len(SCALARS)) and scalars.dtype == np.float32
    rows = kf.channels(d, d)
    np.testing.assert_allclose(windows[0, -1], rows[0], rtol=1e-6)
    assert not windows[0, :-1].any()
    np.testing.assert_allclose(windows[149], rows[30:150], rtol=1e-6)
    seen = CHANNELS.index("seen")
    assert windows[5, : WINDOW - 6, seen].sum() == 0 and windows[5, -6:, seen].all()


def test_each_minute_sees_only_the_bars_up_to_it():
    rng = np.random.default_rng(7)
    d, market = random_day(rng), random_day(rng)
    windows, scalars = inputs(d, market)
    for t in (0, 1, 59, 119, 120, 200, 389):
        # as the live engine will hold the session at the close of bar t
        cut = {f: getattr(d, f)[: t + 1] for f in MINUTE_FIELDS}
        cut_market = {f: getattr(market, f)[: t + 1] for f in MINUTE_FIELDS}
        w, s = inputs(replace(d, **cut), replace(market, **cut_market))
        assert np.array_equal(w[t], windows[t]) and np.array_equal(s[t], scalars[t])
        # whatever comes after bar t changes nothing at t
        later = {f: getattr(d, f).copy() for f in MINUTE_FIELDS}
        for f in ("open", "high", "low", "close"):
            later[f][t + 1 :] *= 1.5
        later["volume"][t + 1 :] = 1e9
        later["news"][t + 1 :] = 50
        w, s = inputs(replace(d, **later), market)
        assert np.array_equal(w[t], windows[t]) and np.array_equal(s[t], scalars[t])


def test_the_market_must_cover_the_same_minutes():
    with pytest.raises(ValueError, match="same minutes"):
        kf.channels(day([100.0, 101.0, 102.0]), day([100.0, 101.0]))
