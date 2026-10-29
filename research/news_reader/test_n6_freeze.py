"""N6's freeze (n6_freeze.py). They need PyTorch:
data/llmenv/bin/python -m pytest research/news_reader/test_n6_freeze.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
import n6_freeze as nf
import n6_model as nm
from test_n6_model import world

CHOSEN = {
    "variant": 3,
    "name": "v3",
    "settings": {
        "dim": 8,
        "attention": 4,
        "context": 8,
        "hidden": 8,
        "dropout": 0.0,
        "loss": "mse",
        "aux": 0.1,
        "lr": 0.003,
        "decay": 0.01,
        "epochs": 2,
        "days_per_batch": 8,
        "start": "pca",
        "inputs": "news",
    },
    "seeds": [0, 1],
}


def test_the_frozen_models_are_written_once_and_give_the_same_forecasts(tmp_path):
    a, store = world(days=40, signal=1.0, first=date(2026, 9, 1))
    weights = tmp_path / "models" / "n6.npz"
    digest = nf.freeze(a, store, CHOSEN, "e" * 64, weights)
    assert digest == nf.sha256(weights) and len(digest) == 64
    with np.load(weights, allow_pickle=False) as kept:  # plain arrays, no pickle
        arrays = dict(kept)
    assert json.loads(str(arrays["settings"])) == CHOSEN["settings"]
    assert arrays["seeds"].tolist() == [0, 1] and str(arrays["eyes_sha256"]) == "e" * 64
    assert str(arrays["trained_from"]) == "2026-09-01"
    assert str(arrays["trained_to"]) <= "2026-10-09"
    models = nf.models_from(arrays)
    assert len(models) == 2 and not hasattr(models[0], "context")
    rows = np.flatnonzero(a.count > 0)[:50]
    x = nm.take(nm.inputs(a, np.ones(len(a.day), bool), "cpu"), torch.as_tensor(rows))
    out = nf.forecast(models, store, x)
    assert np.isfinite(out).all() and np.std(out) > 0
    one = nf.forecast(models[:1], store, x)
    assert not np.allclose(out, one)  # the mean of both seeds, not the first alone
    with pytest.raises(SystemExit, match="frozen once"):
        nf.freeze(a, store, CHOSEN, "e" * 64, weights)


def test_the_frozen_models_learn_the_headlines_as_the_fitted_ones_did(tmp_path):
    a, store = world(days=200, signal=1.0, first=date(2026, 1, 1))
    weights = tmp_path / "n6.npz"
    nf.freeze(
        a,
        store,
        CHOSEN | {"settings": CHOSEN["settings"] | {"epochs": 8}},
        "e" * 64,
        weights,
    )
    with np.load(weights) as kept:
        models = nf.models_from(dict(kept))
    rows = np.flatnonzero(a.count > 0)
    x = nm.take(nm.inputs(a, np.ones(len(a.day), bool), "cpu"), torch.as_tensor(rows))
    assert np.corrcoef(nf.forecast(models, store, x), a.y[rows])[0, 1] > 0.8


def test_nothing_is_frozen_without_a_second_look_above_0(tmp_path, monkeypatch):
    monkeypatch.setattr(nf, "SECOND_LOOK", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="second look comes first"):
        nf.main()
    lock = tmp_path / "second_look.json"
    lock.write_text(json.dumps({"report": {"passes": False}, "chosen": CHOSEN}))
    monkeypatch.setattr(nf, "SECOND_LOOK", lock)
    with pytest.raises(SystemExit, match="not above 0"):
        nf.main()
    lock.write_text(
        json.dumps({"report": {"passes": True}, "chosen": CHOSEN | {"variant": 4}})
    )
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps(CHOSEN))
    monkeypatch.setattr(nf, "CHOSEN", chosen)
    with pytest.raises(SystemExit, match="not the choice the second look ran"):
        nf.main()
