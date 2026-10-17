"""The GPU environment's guarantees, on small made-up markets (data/llmenv: python -m pytest).

These run on the GPU with PyTorch, which the project's CI does not have; run them before
every change to sim.py or gym.py:

    data/llmenv/bin/python -m pytest research/gpu_sim
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gym

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def write_market(path: Path, close: np.ndarray, length: np.ndarray) -> None:
    """A made-up ETF: `close` [sessions, 390] (NaN past each session's length)."""
    sessions = close.shape[0]
    fill_open = np.roll(close, 1, axis=1)
    fill_open[:, 0] = close[:, 0]
    minute = np.arange(390)[None, :]
    inside = minute < length[:, None]
    close = np.where(inside, close, np.nan)
    fill_open = np.where(inside, fill_open, np.nan)
    volume = np.where(inside, 1000.0, 0.0)
    np.savez(
        path,
        close=close,
        fill_open=fill_open,
        pv=np.nan_to_num(close) * volume,
        vol=volume,
        first_open=fill_open[:, 0],
        last_close=np.array([close[i, length[i] - 1] for i in range(sessions)]),
        length=length,
        days=np.array(
            [
                (date(2020, 1, 1) + timedelta(days=i)).isoformat()
                for i in range(sessions)
            ]
        ),
        clean=np.arange(sessions) >= 15,
    )


def random_walk(sessions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0007, (sessions, 390))
    days = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, sessions)))
    return days[:, None] * np.exp(np.cumsum(steps, axis=1))


def test_an_observation_uses_only_the_minutes_before_its_decision(tmp_path):
    close = random_walk(40, 0)
    write_market(tmp_path / "A.npz", close, np.full(40, 390))
    before = gym.load(str(tmp_path), ["A"], "cuda", torch.float64)
    changed = close.copy()
    changed[30, 200:] *= 1.5  # the last session but nine, from minute 200 on
    write_market(tmp_path / "A.npz", changed, np.full(40, 390))
    after = gym.load(str(tmp_path), ["A"], "cuda", torch.float64)
    row = 30 - 15  # the session's row among the clean ones
    decisions_seeing_200 = [j for j in range(12) if 30 * (j + 1) - 1 >= 200]
    for j in range(12):
        same = torch.equal(before.obs[row, j], after.obs[row, j])
        assert same == (j not in decisions_seeing_200), j
    assert torch.equal(before.obs[:row], after.obs[:row])  # earlier sessions untouched


def test_a_half_day_has_decisions_only_before_its_close(tmp_path):
    length = np.full(40, 390)
    length[20] = 210
    write_market(tmp_path / "A.npz", random_walk(40, 1), length)
    market = gym.load(str(tmp_path), ["A"], "cuda", torch.float64)
    assert market.decide[20 - 15].tolist() == [True] * 6 + [False] * 6
    assert market.decide[21 - 15].tolist() == [True] * 12


def test_holding_all_day_on_a_flat_market_costs_a_round_trip(tmp_path):
    write_market(tmp_path / "A.npz", np.full((40, 390), 100.0), np.full(40, 390))
    market = gym.load(str(tmp_path), ["A"], "cuda", torch.float64)
    one = torch.zeros(1, 1, device="cuda", dtype=torch.float64)
    returns = gym.rollout(
        market, lambda obs, p, m: torch.ones_like(obs[..., 0]), one, 0.0001
    )
    assert torch.allclose(
        returns, torch.full_like(returns, -0.0002)
    )  # in and out, 1 bp each


def test_exposure_is_held_between_decisions_and_capped(tmp_path):
    close = np.full((40, 390), 100.0)
    close[:, 60:] = 101.0  # +1% after minute 60
    write_market(tmp_path / "A.npz", close, np.full(40, 390))
    market = gym.load(str(tmp_path), ["A"], "cuda", torch.float64)
    one = torch.zeros(1, 1, device="cuda", dtype=torch.float64)
    returns = gym.rollout(
        market, lambda obs, p, m: torch.full_like(obs[..., 0], 5.0), one, 0.0
    )
    # bought at 10:00's fill (minute 30, 100), capped at 2x, held to the close (101)
    assert torch.allclose(returns, torch.full_like(returns, 2 * 0.01))


def test_the_rule_reproduces_the_research():
    data = Path("data/gpu_sim/QQQ.npz")
    reference = Path("data/intraday_pilot/qqq_days.parquet")
    if not (data.exists() and reference.exists()):
        pytest.skip("needs the stored QQQ data")
    import polars as pl

    market = gym.load("data/gpu_sim", ["QQQ"], "cuda", torch.float64)
    one = torch.zeros(1, 1, device="cuda", dtype=torch.float64)
    traded: list = []
    gross = (
        gym.rollout(market, gym.rule_policy, one, 0.0, turnover=traded)[0].cpu().numpy()
    )
    ref = pl.read_parquet(reference)
    days = {d.isoformat(): i for i, d in enumerate(ref["day"].to_list())}
    rows = [days[d] for d in market.day]
    want = ref["gross_1x"].to_numpy()[rows] * ref["size_2x"].to_numpy()[rows]
    assert np.abs(gross - want).max() < 1e-12
    want_traded = ref["traded_1x"].to_numpy()[rows] * ref["size_2x"].to_numpy()[rows]
    assert np.abs(traded[0][0].cpu().numpy() - want_traded).max() < 1e-9


def test_no_trade_after_a_half_day_closes(tmp_path):
    close = np.full((40, 390), 100.0)
    close[:, 150:] = 101.0
    length = np.full(40, 390)
    length[20] = 210  # closes at 13:00
    write_market(tmp_path / "A.npz", close, length)
    market = gym.load(str(tmp_path), ["A"], "cuda", torch.float64)
    one = torch.zeros(1, 1, device="cuda", dtype=torch.float64)

    def turn_after_noon(obs, params, market):  # long before 13:15, short after
        return torch.where(obs[..., 6] < 0.5, 1.0, -1.0).to(obs.dtype)

    traded: list = []
    returns = gym.rollout(market, turn_after_noon, one, 0.0001, turnover=traded)
    row = 20 - 15
    # the half day: long at 10:00 (100) to its close (101), one round trip of cost
    assert returns[0, row].item() == pytest.approx(0.01 - 0.0002)
    assert traded[0][0, row].item() == pytest.approx(2.0)
