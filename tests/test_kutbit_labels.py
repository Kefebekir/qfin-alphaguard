"""What happened after each minute (research/kutbit/labels.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "kutbit_labels", Path(__file__).parents[1] / "research" / "kutbit" / "labels.py"
)
kl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kl)


def by_hand(open_, high, low, close, h):
    """The labels one minute at a time, as the module's docstring says them."""
    n = len(open_)
    ret, worst, best, cut = [], [], [], []
    for t in range(n - 1):
        entry, stop = open_[t + 1], min(t + h, n - 1)
        ret.append(close[stop] / entry - 1)
        worst.append(min(low[t + 1 : stop + 1]) / entry - 1)
        best.append(max(high[t + 1 : stop + 1]) / entry - 1)
        cut.append(t + h > n - 1)
    return ret, worst, best, cut


def test_a_position_enters_at_the_next_open_and_records_its_way():
    open_ = np.array([100.0, 101.0, 102.0, 100.0, 103.0])
    high = np.array([101.0, 103.0, 104.0, 101.0, 105.0])
    low = np.array([99.0, 100.0, 99.5, 98.0, 102.0])
    close = np.array([101.0, 102.0, 100.0, 103.0, 104.0])
    two = kl.labels(open_, high, low, close, horizons=(2,))[2]
    # decided at the close of bar 0: in at bar 1's open, 101; out at bar 2's close, 100
    assert two["ret"][0] == pytest.approx(100 / 101 - 1)
    assert two["worst"][0] == pytest.approx(99.5 / 101 - 1)
    assert two["best"][0] == pytest.approx(104 / 101 - 1)
    # decided at bar 3: in at bar 4's open, 103; the session ends at bar 4
    assert two["ret"][3] == pytest.approx(104 / 103 - 1) and two["cut"][3]
    assert np.isnan(two["ret"][4]) and two["cut"][4]


def test_every_minute_and_horizon_as_by_hand():
    rng = np.random.default_rng(3)
    n = 130
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = close * np.exp(rng.normal(0, 3e-4, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 1e-3, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 1e-3, n))
    found = kl.labels(open_, high, low, close)
    assert set(found) == set(kl.HORIZONS) == {5, 15, 30, 60}
    for h in kl.HORIZONS:
        ret, worst, best, cut = by_hand(open_, high, low, close, h)
        np.testing.assert_allclose(found[h]["ret"][:-1], ret)
        np.testing.assert_allclose(found[h]["worst"][:-1], worst)
        np.testing.assert_allclose(found[h]["best"][:-1], best)
        assert found[h]["cut"][:-1].tolist() == cut


def test_a_one_minute_session_has_nothing_to_learn():
    one = np.array([100.0])
    found = kl.labels(one, one, one, one)
    assert all(np.isnan(found[h]["ret"]).all() and found[h]["cut"].all() for h in found)
