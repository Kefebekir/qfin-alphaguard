"""Evolution strategies and the holdout lock, on made-up markets (data/llmenv, the GPU)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import es
import gym
from test_gym import write_market

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def trending_market(path: Path) -> gym.Market:
    """Days that keep the direction of their first hour: a pattern a policy can learn."""
    rng = np.random.default_rng(3)
    sessions = 80
    sign = rng.choice([-1.0, 1.0], sessions)
    minute = np.arange(390)[None, :]
    drift = sign[:, None] * 0.01 * minute / 390
    noise = np.cumsum(rng.normal(0, 0.0003, (sessions, 390)), axis=1)
    level = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, sessions)))[:, None]
    write_market(path / "A.npz", level * np.exp(drift + noise), np.full(sessions, 390))
    return gym.load(str(path), ["A"], "cuda", torch.float32)


def linear(obs, params, market):
    """exposure = 2 tanh(w . obs + b): 11 parameters."""
    return 2 * torch.tanh(
        torch.einsum("vnf,vf->vn", obs, params[:, :10]) + params[:, 10:11]
    )


def test_evolution_learns_a_pattern_that_is_there(tmp_path):
    market = trending_market(tmp_path)
    theta, history = es.evolve(
        market, linear, 11, generations=60, population=64, sigma=0.1, lr=0.05
    )
    learned = es.sharpe(gym.rollout(market, linear, theta[None, :]))[0].item()
    assert history[-1] > history[0]
    assert learned > 3  # following the first hour on days that keep its direction


def test_the_same_seed_trains_the_same_model(tmp_path):
    market = trending_market(tmp_path)
    a, _ = es.evolve(market, linear, 11, generations=5, population=16, seed=7)
    b, _ = es.evolve(market, linear, 11, generations=5, population=16, seed=7)
    c, _ = es.evolve(market, linear, 11, generations=5, population=16, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_the_noise_comes_in_pairs(tmp_path):
    market = trending_market(tmp_path)
    with pytest.raises(ValueError, match="pairs"):
        es.evolve(market, linear, 11, generations=1, population=15)


def test_the_holdout_needs_its_settings_committed_and_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "VARIANTS", tmp_path / "variants")
    monkeypatch.setattr(es, "LOCKS", tmp_path)
    net = {"hidden": 16, "see_rule": True}
    with pytest.raises(RuntimeError, match="commit"):
        es.run_holdout(linear, 11, {"generations": 10}, net)
    (tmp_path / "variants").mkdir()
    chosen = {"settings": {"generations": 20}, "network": net}
    (tmp_path / "variants" / "G1_chosen.json").write_text(json.dumps(chosen))
    with pytest.raises(RuntimeError, match="differ"):
        es.run_holdout(linear, 11, {"generations": 10}, net)
    with pytest.raises(RuntimeError, match="differ"):
        es.run_holdout(linear, 11, {"generations": 20}, net | {"see_rule": False})
    (tmp_path / "holdout_G1.json").write_text("{}")
    with pytest.raises(RuntimeError, match="has been run"):
        es.run_holdout(linear, 11, {"generations": 20}, net)
