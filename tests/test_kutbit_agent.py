"""Efe's agent for KUTBIT's first variant (research/kutbit/agent.py), step by step.

They check the agent Efe writes from research/kutbit/AGENT_GUIDE.md. They need PyTorch, so they
run in data/llmenv, and they wait until agent.py exists; in the project's environment and in CI
they are skipped. Each step of the guide has its tests:

    PYTHONPATH=src data/llmenv/bin/python -m pytest tests/test_kutbit_agent.py -k step2
"""

import importlib.util
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = Path(__file__).parents[1] / "research" / "kutbit"
if not (HERE / "agent.py").exists():
    pytest.skip(
        "Efe has not written research/kutbit/agent.py yet", allow_module_level=True
    )
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("efe_agent", HERE / "agent.py")
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)
import gym
import harness as hs


def synthetic(seed=1, edge=0.01, odd=True):
    """Events of 2016-2021, ten stocks, forty a month. Big moves (over 6 usual moves) reverse
    by `edge` over every hold, smaller ones go on by half as much; stock 0, when `odd`, goes on
    after big moves too. QQQ swings five times as much when the event's QQQ number is large."""
    rng = np.random.default_rng(seed)
    day, minute, ticker = [], [], []
    for year in range(2016, 2022):
        for month in range(1, 13):
            for d in range(4):
                for e in range(10):
                    day.append(date(year, month, 1 + 6 * d).toordinal())
                    minute.append(int(rng.integers(0, 380)))
                    ticker.append(e)
    n = len(day)
    state = rng.normal(0, 1, (n, len(gym.STATE))).astype(np.float32)
    state[:, 0] = 4 + rng.exponential(2.5, n)
    state[:, gym.STATE.index("beta")] = 1.0
    big = state[:, 0] > 6
    ticker = np.array(ticker, dtype=np.int16)
    pattern = np.where(big, -edge, edge / 2)
    if odd:
        pattern = np.where(big & (ticker == 0), edge, pattern)
    jumpy = np.abs(state[:, 4]) > 1
    qqq = rng.normal(0, 1, (n, 3)) * np.where(jumpy, 0.01, 0.002)[:, None]
    apart = rng.normal(0, 0.003, (n, 3)) + pattern[:, None]
    minute = np.array(minute, dtype=np.int16)
    return {
        "state": state,
        "own": apart + qqq,
        "qqq": qqq,
        "beta": np.ones(n),
        "kind": rng.integers(0, 8, n).astype(np.uint8),
        "day": np.array(day, dtype=np.int32),
        "minute": minute,
        "known": np.minimum(minute + 30, 389).astype(np.int16),
        "ticker": ticker,
        "chart": np.zeros((n, 2 * gym.CHART + 1), np.float32),
    }


def history(data, until=None):
    """What fit is given: every event's view with its outcomes, up to `until`."""
    keep = np.arange(len(data["day"]) if until is None else until)
    values, permitted = gym.utility(gym.returns(data)), gym.allowed(data)
    return hs.view(data, keep, permitted, values)


def fitted(data, **change):
    a = agent.make({**agent.SETTINGS[0], **change})
    a.fit(history(data))
    return a


# --- step 1: the frame ---------------------------------------------------------------------


def test_step1_the_settings_and_make():
    assert len(agent.SETTINGS) == 4
    for setting in agent.SETTINGS:
        assert set(setting) == {"hidden", "pull", "epochs", "rate", "seed"}
    a = agent.make(agent.SETTINGS[0])
    assert a.weights is None and a.counts.sum() == 0 and (a.trust == 0).all()
    assert (
        agent.SIZE.shape == agent.SIDE.shape == agent.HOLD.shape == (len(gym.ACTIONS),)
    )
    j = gym.index_of(gym.Action(-1.0, 2.0, 30, 0.5))
    assert (agent.SIDE[j], agent.SIZE[j], agent.HEDGE[j]) == (-1.0, 2.0, 0.5)
    assert agent.HOLD[j] == gym.HOLDS.index(30) and agent.SIZE[0] == 0


# --- step 2: the network -------------------------------------------------------------------


def test_step2_a_layer_and_the_network():
    a = agent.make(agent.SETTINGS[0])
    generator = torch.Generator().manual_seed(0)
    weight, bias = a.layer(16, 64, generator)
    assert weight.shape == (16, 64) and bias.shape == (64,)
    assert weight.requires_grad and bias.requires_grad and (bias == 0).all()
    assert weight.std().item() == pytest.approx(math.sqrt(2 / 16), rel=0.15)
    a.weights = [
        *a.layer(16, 8, generator),
        *a.layer(8, 8, generator),
        *a.layer(8, 9, generator),
    ]
    out = a.network(torch.zeros(5, 16))
    assert out.shape == (5, len(gym.HOLDS), 3)


def test_step2_the_inputs_are_in_units_of_their_spread():
    a = agent.make(agent.SETTINGS[0])
    a.mean, a.spread = torch.full((16,), 2.0), torch.full((16,), 4.0)
    x = a.inputs(np.array([[6.0] * 16, [100.0] * 16, [-100.0] * 16], dtype=np.float32))
    assert x[0, 0].item() == pytest.approx(1.0)
    assert x[1].max().item() == 5 and x[2].min().item() == -5


# --- step 3: the loss, and what it is taken on -------------------------------------------


def test_step3_the_loss_is_the_student_t_less_its_constant():
    x = torch.tensor([-2.0, 0.1, 0.5, 3.0])
    mean, log_scale = torch.tensor(0.3), torch.tensor(math.log(0.7))
    found = agent.Agent.loss(x, mean, log_scale)
    nu = agent.NU
    constant = (
        math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    )
    exact = -torch.distributions.StudentT(nu, mean, log_scale.exp()).log_prob(x).mean()
    assert found.item() - constant == pytest.approx(exact.item(), rel=1e-5)


def test_step3_the_loss_is_least_at_the_true_mean_and_scale():
    rng = np.random.default_rng(2)
    x = torch.as_tensor(
        rng.standard_t(agent.NU, 20000) * 0.5 + 1.0, dtype=torch.float32
    )
    at = agent.Agent.loss
    best = at(x, torch.tensor(1.0), torch.tensor(math.log(0.5)))
    for mean, scale in ((0.8, 0.5), (1.2, 0.5), (1.0, 0.35), (1.0, 0.7)):
        assert at(x, torch.tensor(mean), torch.tensor(math.log(scale))) > best


def test_step3_the_targets_are_the_move_apart_from_qqq_and_qqq_s_in_percent():
    a = agent.make(agent.SETTINGS[0])
    known = {
        "own": np.array([[0.01, 0.02, 0.03]]),
        "qqq": np.array([[0.004, -0.002, 0.0]]),
        "beta": np.array([1.5]),
    }
    apart, qqq = a.targets(known)
    np.testing.assert_allclose(apart.numpy(), [[0.4, 2.3, 3.0]], rtol=1e-5)
    np.testing.assert_allclose(qqq.numpy(), [[0.4, -0.2, 0.0]], rtol=1e-5)


# --- step 4: every minute ----------------------------------------------------------------


def by_hand(data):
    """An agent whose network and trust are set by hand, as step 5's learning would set them."""
    a = agent.make(agent.SETTINGS[0])
    generator = torch.Generator().manual_seed(3)
    state = torch.as_tensor(data["state"], dtype=torch.float32)
    a.mean, a.spread = state.mean(0), state.std(0)
    a.weights = [
        *a.layer(16, 8, generator),
        *a.layer(8, 8, generator),
        *a.layer(8, 3 * len(gym.HOLDS), generator),
    ]
    a.trust = torch.tensor([0.5, 0.25, 1.0])
    return a


def test_step4_each_outcome_adds_its_miss_to_its_stock():
    data = synthetic()
    a = by_hand(data)
    picked = np.flatnonzero(data["ticker"] == 3)[:7]
    known = hs.view(data, picked, gym.allowed(data), gym.utility(gym.returns(data)))
    a.update(known)
    assert a.counts[3] == 7 and a.counts.sum() == 7
    with torch.no_grad():
        edge = a.trust * a.network(a.inputs(known["state"]))[..., 0]
    apart = (known["own"] - known["beta"][:, None] * known["qqq"]) * 100
    np.testing.assert_allclose(
        a.sums[3], (apart - edge.numpy()).sum(0), rtol=1e-4, atol=1e-6
    )
    expected = a.sums[3] / (7 + a.pull)
    np.testing.assert_allclose(
        a.correction(np.array([3]))[0].numpy(), expected, rtol=1e-5
    )
    assert (a.correction(np.array([5])) == 0).all()  # a stock without events: nothing


def test_step4_nothing_is_counted_before_the_network_has_learned():
    data = synthetic()
    a = agent.make(agent.SETTINGS[0])
    a.update(
        hs.view(data, np.arange(9), gym.allowed(data), gym.utility(gym.returns(data)))
    )
    assert a.counts.sum() == 0


# --- step 5: learning ----------------------------------------------------------------------


def test_step5_it_trusts_an_edge_that_came_true_and_not_noise():
    loud = fitted(synthetic())
    assert (loud.trust > 0.5).all()
    quiet = fitted(synthetic(seed=2, edge=0.0, odd=False))
    assert (quiet.trust < 0.5).all() and quiet.trust.mean() < 0.3


def test_step5_the_trust_is_the_slope_kept_between_none_and_all():
    data = synthetic()
    a = fitted(data, epochs=5)
    with torch.no_grad():
        edge = a.network(a.inputs(data["state"]))[..., 0]
    for slope, trust in ((-1.0, 0.0), (0.5, 0.5), (3.0, 1.0)):
        later = {
            "state": data["state"],
            "own": (slope * edge / 100).double().numpy(),
            "qqq": np.zeros_like(data["qqq"]),
            "beta": data["beta"],
        }
        found = a.measure_trust(later).numpy()
        np.testing.assert_allclose(found, trust, atol=1e-4)


def test_step5_it_tells_a_jumpy_qqq_from_a_calm_one():
    data = synthetic()
    a = fitted(data)
    with torch.no_grad():
        scale = a.network(a.inputs(data["state"]))[..., 2].exp().mean(1).numpy()
    jumpy = np.abs(data["state"][:, 4]) > 1
    assert scale[jumpy].mean() > 2 * scale[~jumpy].mean()


def test_step5_too_few_events_teach_nothing():
    data = synthetic()
    a = agent.make(agent.SETTINGS[0])
    a.fit(history(data, until=100))
    assert a.weights is None


def test_step5_the_same_seed_learns_the_same():
    data = synthetic()
    one, two = fitted(data, epochs=5), fitted(data, epochs=5)
    for w1, w2 in zip(one.weights, two.weights, strict=True):
        assert torch.equal(w1, w2)


def test_step5_fit_counts_every_stock_again_from_its_events():
    data = synthetic()
    a = fitted(data, epochs=5)
    a.fit(history(data))  # a second month's fit counts afresh, not on top
    assert a.counts[:10].sum() == len(data["day"])
    assert (a.counts[:10] == np.bincount(data["ticker"], minlength=10)).all()


# --- step 6: deciding ----------------------------------------------------------------------


def test_step6_every_action_s_worth_by_hand():
    data = synthetic()
    a = fitted(data, epochs=5)
    idx = np.arange(20)
    permitted = gym.allowed(data)
    events = hs.view(data, idx, permitted)
    worth = a.forecast(events)
    with torch.no_grad():
        out = a.network(a.inputs(events["state"])).double()
    edge = (
        a.trust.double() * out[..., 0] + a.correction(events["ticker"]).double()
    ) / 100
    var = agent.NU / (agent.NU - 2) * torch.exp(2 * out[..., 1:]) / 1e4
    for i in (0, 7, 19):
        for action in (gym.Action(1.0, 1.0, 5, 0.0), gym.Action(-1.0, 2.0, 30, 0.5)):
            j = gym.index_of(action)
            h = gym.HOLDS.index(action.hold)
            mean = action.size * action.side * edge[i, h]
            cost = action.size * gym.COST + action.hedge * action.size * gym.COST_QQQ
            left = 1 - action.hedge
            variance = action.size**2 * (var[i, h, 0] + left**2 * var[i, h, 1])
            hand = float(mean - cost - gym.GAMMA / 2 * variance)
            assert worth[i, j] == pytest.approx(hand, rel=1e-4, abs=1e-9)
    assert (worth[:, 0] == 0).all()
    assert np.isneginf(worth[~permitted[idx]]).all()


def test_step6_an_agent_that_learned_nothing_lets_every_move_be():
    data = synthetic()
    a = agent.make(agent.SETTINGS[0])
    a.fit(history(data, until=100))
    events = hs.view(data, np.arange(5), gym.allowed(data))
    assert list(a.act(events)) == [0] * 5


def test_step6_it_takes_the_best_action_or_lets_the_move_be():
    data = synthetic()
    a = fitted(data, epochs=5)
    events = hs.view(data, np.arange(40), gym.allowed(data))
    worth = a.forecast(events)
    chosen = a.act(events)
    best = worth.argmax(axis=1)
    expected = np.where(worth[np.arange(40), best] > 0, best, 0)
    assert (chosen == expected).all()
    assert gym.allowed(data)[np.arange(40), chosen].all()


# --- the whole agent, through the harness ---------------------------------------------------


def test_the_agent_finds_a_planted_edge_and_waits_on_noise():
    data = synthetic()
    found = hs.measure(agent.make, agent.SETTINGS[0], data, log=lambda s: None)
    assert found["gate"]["passed"]
    noise = synthetic(seed=2, edge=0.0, odd=False)
    quiet = hs.placebo(agent.make, agent.SETTINGS[:2], noise, log=lambda s: None)
    assert quiet["passed"] and quiet["trades"] < 50


def test_the_agent_hedges_where_qqq_swings_and_learns_each_stock():
    data = synthetic()
    found = hs.measure(agent.make, agent.SETTINGS[0], data, log=lambda s: None)
    idx = found["idx"]
    acts = [gym.ACTIONS[c] for c in found["choice"][idx]]
    traded = np.array([x.side != 0 for x in acts])
    hedge = np.array([x.hedge for x in acts])
    jumpy = np.abs(data["state"][idx, 4]) > 1
    assert hedge[traded & jumpy].mean() > hedge[traded & ~jumpy].mean() + 0.2
    side = np.array([x.side for x in acts])
    big = data["state"][idx, 0] > 6
    late = data["day"][idx] >= date(2021, 1, 1).toordinal()
    mine = big & traded & late & (data["ticker"][idx] == 0)
    others = big & traded & late & (data["ticker"][idx] != 0)
    assert np.mean(side[mine] > 0) > 0.5 > np.mean(side[others] > 0)


def test_the_agent_has_every_method_the_harness_calls():
    a = agent.make(agent.SETTINGS[0])
    for method in ("fit", "update", "forecast", "act"):
        assert callable(getattr(a, method))


def test_it_is_written_from_scratch():
    source = (HERE / "agent.py").read_text()
    for ready_made in (
        "torch.nn",
        "torch.optim",
        "torch.distributions",
        "nn.Module",
        "sklearn",
    ):
        assert ready_made not in source
