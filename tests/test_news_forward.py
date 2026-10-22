"""N5's frozen challenger (research/news_reader/forward.py)."""

import importlib.util
import json
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


forward = _load("forward")
efe_model = _load("efe_model")


def training(rows=400, width=32, seed=0):
    g = np.random.default_rng(seed)
    x = np.hstack([g.normal(size=(rows, width)), np.log1p(g.integers(1, 6, (rows, 1)))])
    y = x[:, 0] * 0.3 - x[:, -1] * 0.2 + g.normal(0, 0.1, rows)
    return x.astype(np.float32), y


def test_the_arrays_give_the_fitted_pipeline_s_forecasts():
    x, y = training()
    model = efe_model.make(8, 10.0)().fit(x, y)
    arrays = forward.freeze_arrays(model)
    fresh, _ = training(rows=50, seed=1)
    assert forward.disagreement(arrays, model, fresh) < forward.TOLERANCE


class FakeHarness:
    N4_LEVERS = ("target", "side", "weights", "sectors")

    def __init__(self, root):
        self.VARIANTS = root
        (root / "N4_chosen.json").write_text(
            json.dumps(
                {
                    "components": 8,
                    "penalty": 10.0,
                    "features": "base-last",
                    "target": "scaled",
                    "side": "tenth",
                    "weights": "equal",
                    "sectors": "none",
                    "variant": 2,
                }
            )
        )
        self.asked = None

    def next_day_returns(self):
        return "returns"

    def n4_year(self, year, returns, features, levers):
        self.asked = (year, returns, features, levers)
        x, y = training()
        days = pl.DataFrame({"day": [date(2016, 1, 4)] * 399 + [date(2026, 10, 9)]})
        return types.SimpleNamespace(train_x=x, train_y=y, train=days)


def test_freezing_fits_the_choice_on_every_year_writes_once_and_round_trips(tmp_path):
    fake = FakeHarness(tmp_path)
    weights = tmp_path / "models" / "frozen.npz"
    digest = forward.freeze(fake, efe_model, weights)
    assert fake.asked == (
        2027,
        "returns",
        "base-last",
        {"target": "scaled", "side": "tenth", "weights": "equal", "sectors": "none"},
    )
    assert digest == forward.sha256(weights) and len(digest) == 64
    stored = forward.load(weights)
    assert str(stored["trained_from"]) == "2016-01-04"
    assert str(stored["trained_to"]) == "2026-10-09"
    assert json.loads(str(stored["settings"]))["components"] == 8
    x, y = training()
    model = efe_model.make(8, 10.0)().fit(x, y)
    assert forward.disagreement(stored, model, x) < forward.TOLERANCE
    with pytest.raises(SystemExit, match="frozen once"):
        forward.freeze(fake, efe_model, weights)
    assert forward.sha256(weights) == digest


def test_the_freeze_measures_the_arrays_against_the_model_and_catches_a_wrong_one():
    x, y = training()
    model = efe_model.make(8, 10.0)().fit(x, y)
    arrays = forward.freeze_arrays(model)
    assert forward.disagreement(arrays, model, x) < forward.TOLERANCE
    wrong = dict(arrays, scale_scale=arrays["scale_scale"] * 1.01)
    assert forward.disagreement(wrong, model, x) > 100 * forward.TOLERANCE


def test_wrong_arrays_are_never_written(tmp_path, monkeypatch):
    fake = FakeHarness(tmp_path)
    real = forward.freeze_arrays

    def off_by_a_little(model):
        arrays = real(model)
        return dict(arrays, ridge_intercept=arrays["ridge_intercept"] + 0.01)

    monkeypatch.setattr(forward, "freeze_arrays", off_by_a_little)
    weights = tmp_path / "frozen.npz"
    with pytest.raises(SystemExit, match="nothing written"):
        forward.freeze(fake, efe_model, weights)
    assert not weights.exists()


def test_one_row_off_is_enough_to_disagree():
    x, y = training()
    model = efe_model.make(8, 10.0)().fit(x, y)
    arrays = forward.freeze_arrays(model)
    true = model.predict(x)

    class OneRowOff:
        def predict(self, rows):
            out = true.copy()
            out[7] += (
                1e-3 * true.std()
            )  # lost in a mean over 400 rows, not in the largest
            return out

    assert forward.disagreement(arrays, OneRowOff(), x) > forward.TOLERANCE
