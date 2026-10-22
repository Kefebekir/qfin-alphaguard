"""N6's eyes (research/news_reader/n6_eyes.py) and the alignment it freezes (align_check.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


align_check = _load("align_check")
n6_eyes = _load("n6_eyes")

NAMES = ("old", "new")


@pytest.fixture
def two(monkeypatch):
    """Two checkpoints of width 3: 'new' the reference, 'old' mapped to it."""
    monkeypatch.setattr(n6_eyes, "CHECKPOINTS", NAMES)
    return {
        "mean_old": np.array([1.0, 2.0, 3.0], np.float32),
        "spread_old": np.array([2.0, 4.0, 1.0], np.float32),
        "map_old": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], np.float32),
        "penalty_old": np.array(100.0, np.float32),
        "mean_new": np.zeros(3, np.float32),
        "spread_new": np.full(3, 2.0, np.float32),
    }


def test_a_reading_is_scaled_then_mapped_unless_it_is_the_reference(two):
    x = np.array([[3.0, 6.0, 4.0]])
    np.testing.assert_allclose(n6_eyes.aligned(two, "old", x), [[1.0, 1.0, -1.0]])
    np.testing.assert_allclose(n6_eyes.aligned(two, "new", x), [[1.5, 3.0, 2.0]])
    assert n6_eyes.aligned(two, "new", x.astype(np.float16)).dtype == np.float32


def test_the_arrays_are_kept_once_and_never_replaced(two, tmp_path):
    path = tmp_path / "models" / "eyes.npz"
    n6_eyes.keep(two, path)
    n6_eyes.keep(two, path)  # the same arrays: nothing to do
    with np.load(path) as kept:
        assert set(kept.files) == set(two)
    changed = two | {"map_old": two["map_old"] * 1.0001}
    with pytest.raises(SystemExit, match="frozen once"):
        n6_eyes.keep(changed, path)
    with pytest.raises(SystemExit, match="frozen once"):
        n6_eyes.keep({k: v for k, v in two.items() if k != "penalty_old"}, path)


def test_the_store_holds_every_checkpoint_s_headlines_in_order(two, tmp_path):
    source, g = tmp_path / "pit", np.random.default_rng(0)
    source.mkdir()
    reads = {"old": g.normal(size=(7, 3)), "new": g.normal(size=(5, 3))}
    for k, name in enumerate(NAMES):
        np.save(source / f"{name}.ids.npy", np.arange(len(reads[name])) + 100 * k)
        np.save(source / f"{name}.last.npy", reads[name].astype(np.float16))
    n6_eyes.build(two, source, tmp_path / "small", chunk=3)
    n6_eyes.build(two, source, tmp_path / "whole", chunk=100)
    store = np.load(tmp_path / "small" / "headlines.npy")
    assert store.dtype == np.float16 and store.shape == (12, 3)
    np.testing.assert_array_equal(store, np.load(tmp_path / "whole" / "headlines.npy"))
    expected = np.vstack(
        [n6_eyes.aligned(two, n, reads[n].astype(np.float16)) for n in NAMES]
    )
    np.testing.assert_allclose(store, expected, rtol=1e-3, atol=1e-3)
    assert np.load(tmp_path / "small" / "ids.npy").tolist() == [
        *range(7),
        *range(100, 105),
    ]
    assert np.load(tmp_path / "small" / "checkpoint.npy").tolist() == [0] * 7 + [1] * 5
    assert not (tmp_path / "small" / "headlines.part.npy").exists()


def test_the_judged_results_are_read_from_align_check_s_table(tmp_path, monkeypatch):
    monkeypatch.setattr(n6_eyes, "CHECKPOINTS", ("201511", "201612", "202412"))
    path = tmp_path / "results.txt"
    path.write_text(
        "checkpoint  none cos none kNN none CKA    A cos    A kNN    A CKA    B cos"
        "    B kNN    B CKA  penalty\n"
        "201511 0.574 0.584 0.944 0.572 0.589 0.942 0.868 0.647 0.974 1e+03\n"
        "201612 0.576 0.583 0.925 0.576 0.591 0.936 0.875 0.653 0.975 1e+03\n"
        "B less A in neighbours agreed, mean of nine: +0.062\n"
    )
    assert n6_eyes.judged_results(path) == {
        "201511": (0.868, 0.647),
        "201612": (0.875, 0.653),
    }


def test_align_check_s_ridge_finds_a_known_map_and_its_scale_standardises():
    g = np.random.default_rng(0)
    z = g.normal(size=(2_000, 20))
    w = g.normal(size=(20, 20))
    found, penalty, scores = align_check.ridge_map(z, z @ w, g)
    assert np.abs(found - w).max() < 0.05  # the least penalty, 10, shrinks it a little
    assert penalty == min(align_check.PENALTIES) and set(scores) == set(
        align_check.PENALTIES
    )
    x = g.normal(3.0, 5.0, size=(1_000, 4))
    scaled = align_check.Scale(x)(x)
    np.testing.assert_allclose(scaled.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(0), 1)


def test_the_frozen_arrays_hold_each_scale_and_each_older_map_with_its_penalty(
    monkeypatch,
):
    monkeypatch.setattr(n6_eyes, "CHECKPOINTS", NAMES)
    g = np.random.default_rng(1)
    scale = {name: align_check.Scale(g.normal(size=(50, 3))) for name in NAMES}
    w = g.normal(size=(3, 3))
    arrays = n6_eyes.freeze(scale, {"old": (w, 1000.0)})
    assert set(arrays) == {
        "mean_old",
        "spread_old",
        "map_old",
        "penalty_old",
        "mean_new",
        "spread_new",
    }
    assert all(a.dtype == np.float32 for a in arrays.values())
    np.testing.assert_allclose(arrays["mean_new"], scale["new"].mean, rtol=1e-6)
    np.testing.assert_allclose(arrays["spread_old"], scale["old"].spread, rtol=1e-6)
    np.testing.assert_allclose(arrays["map_old"], w, rtol=1e-6)
    assert arrays["penalty_old"] == 1000.0
