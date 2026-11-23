"""Efe's network for KUTBIT's second variant (research/kutbit/net.py), step by step.

They need PyTorch, so they run in data/llmenv; in the project's environment and in CI they are
skipped. Each step of research/kutbit/NETWORK_GUIDE.md has its tests:

    PYTHONPATH=src data/llmenv/bin/python -m pytest tests/test_kutbit_net.py -k step3
"""

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))


def loaded(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


net = loaded("efe_net", HERE / "net.py")
import net_data

NAMES = ("linear", "bias", "w1", "b1", "w2", "b2", "w3")


class Table:
    """Rows as the harness gives them: their count, and by position their inputs and targets.
    It keeps the positions it was asked for."""

    def __init__(self, x, z):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
        self.count, self.inputs = self.x.shape
        self.horizons = self.z.shape[1]
        self.device = self.x.device
        self.asked = []

    def batch(self, index):
        self.asked.append(index.clone())
        return self.x[index], self.z[index]


def drawn(seed: int, inputs: int = 971, horizons: int = 5, width: int = 128) -> dict:
    return net.init(inputs, horizons, width, torch.Generator().manual_seed(seed), "cpu")


# --- step 1: the constants ------------------------------------------------------------------


def test_step1_the_constants_are_decision_0027_s():
    assert net.BATCH == net_data.BATCH == 4_096
    assert net.RATE == 0.001 and net.BETAS == (0.9, 0.999) and net.EPSILON == 1e-8
    assert set(net.DECAYED) == {"linear", "w1", "w2", "w3"}


def test_step1_pytorch_is_a_tool_for_tensors_and_derivatives_only():
    source = (HERE / "net.py").read_text()
    ready_made = (
        r"torch\.(nn|optim)\.|import torch\.(nn|optim)|from torch import (nn|optim)"
    )
    assert not re.search(ready_made, source)


# --- step 2: the parameters -----------------------------------------------------------------


def test_step2_the_parameters_start_where_decision_0027_says():
    params = drawn(1)
    assert {k: tuple(v.shape) for k, v in params.items()} == {
        "linear": (971, 5),
        "bias": (5,),
        "w1": (971, 128),
        "b1": (128,),
        "w2": (128, 128),
        "b2": (128,),
        "w3": (128, 5),
    }
    for name in ("linear", "bias", "b1", "b2", "w3"):
        assert (params[name] == 0).all(), name
    for name, fan_in in (("w1", 971), ("w2", 128)):  # He's scale
        assert params[name].std().item() == pytest.approx(
            math.sqrt(2 / fan_in), rel=0.05
        )
        assert abs(params[name].mean().item()) < 0.01
    assert all(p.requires_grad and p.is_leaf for p in params.values())
    assert set(net.DECAYED) | {"bias", "b1", "b2"} == set(params)


def test_step2_a_seed_draws_the_same_weights_and_another_seed_others():
    first, again, other = drawn(1), drawn(1), drawn(2)
    assert all(torch.equal(first[k], again[k]) for k in NAMES)
    assert not torch.equal(first["w1"], other["w1"])


# --- step 3: the forecast -------------------------------------------------------------------


def small(**given) -> dict:
    """Parameters of 3 inputs, 2 hidden units and 2 horizons, as given, zeros otherwise."""
    shapes = {
        "linear": (3, 2),
        "bias": (2,),
        "w1": (3, 2),
        "b1": (2,),
        "w2": (2, 2),
        "b2": (2,),
        "w3": (2, 2),
    }
    return {
        k: torch.tensor(given[k], dtype=torch.float64)
        if k in given
        else torch.zeros(s, dtype=torch.float64)
        for k, s in shapes.items()
    }


def test_step3_a_new_network_forecasts_nothing():
    x = torch.randn(50, 971)
    assert (net.forward(drawn(3), x) == 0).all()


def test_step3_the_forecast_by_hand():
    params = small(
        linear=[[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]],
        bias=[0.1, -0.1],
        w1=[[1.0, -1.0], [0.0, 1.0], [1.0, 0.0]],
        b1=[0.0, 0.5],
        w2=[[1.0, 0.0], [1.0, -1.0]],
        b2=[-0.5, 0.0],
        w3=[[2.0, 0.0], [0.0, 3.0]],
    )
    x = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.0, 4.0]], dtype=torch.float64)
    p = {k: v.numpy() for k, v in params.items()}
    xs = x.numpy()
    hidden = np.maximum(xs @ p["w1"] + p["b1"], 0)
    hidden = np.maximum(hidden @ p["w2"] + p["b2"], 0)
    expected = xs @ p["linear"] + p["bias"] + hidden @ p["w3"]
    assert net.forward(params, x).numpy() == pytest.approx(expected)
    # the linear path alone, and the hidden path alone
    alone = dict(params, w3=torch.zeros(2, 2, dtype=torch.float64))
    assert net.forward(alone, x).numpy() == pytest.approx(xs @ p["linear"] + p["bias"])
    hidden_only = dict(params, linear=torch.zeros(3, 2, dtype=torch.float64))
    assert net.forward(hidden_only, x).numpy() == pytest.approx(
        p["bias"] + hidden @ p["w3"]
    )


def test_step3_a_hidden_unit_below_zero_adds_nothing():
    params = small(
        w1=[[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        w2=[[1.0, 0.0], [0.0, 0.0]],
        w3=[[5.0, 5.0], [0.0, 0.0]],
    )
    positive = torch.tensor(
        [[2.0, 0.0, 0.0]], dtype=torch.float64
    )  # w1 gives -2: cut to 0
    assert (net.forward(params, positive) == 0).all()
    negative = -positive  # w1 gives +2, passed on
    assert net.forward(params, negative).tolist() == [[10.0, 10.0]]


# --- step 4: the loss -----------------------------------------------------------------------


def test_step4_the_loss_is_the_mean_square_error_over_rows_and_horizons():
    forecast = torch.tensor([[1.0, 2.0], [0.0, -1.0], [3.0, 3.0]])
    z = torch.tensor([[0.0, 2.0], [1.0, 1.0], [3.0, 0.0]])
    assert net.loss(forecast, z).item() == pytest.approx((1 + 0 + 1 + 4 + 0 + 9) / 6)


def test_step4_the_derivatives_match_small_differences():
    generator = torch.Generator().manual_seed(4)
    params = {
        k: v.detach().double().requires_grad_()
        for k, v in net.init(6, 3, 4, generator, "cpu").items()
    }
    with torch.no_grad():  # away from zero, so the hidden layers are in play
        for p in params.values():
            p.add_(0.3 * torch.randn(p.shape, generator=generator, dtype=torch.float64))
    x = torch.randn(10, 6, generator=generator, dtype=torch.float64)
    z = torch.randn(10, 3, generator=generator, dtype=torch.float64)

    def of(*values):
        return net.loss(net.forward(dict(zip(NAMES, values)), x), z)

    assert torch.autograd.gradcheck(of, tuple(params[k] for k in NAMES))


# --- step 5: Adam, by hand ------------------------------------------------------------------


def two_parameters() -> dict:
    return {
        "w1": torch.tensor([1.0, -2.0], requires_grad=True),  # a weight: decayed
        "b1": torch.tensor([0.5, 0.5], requires_grad=True),  # a bias: not decayed
    }


def test_step5_two_steps_of_adam_by_hand():
    params, decay = two_parameters(), 0.1
    memory = net.adam(params)
    grads = [
        {"w1": torch.tensor([0.2, -4.0]), "b1": torch.tensor([1e-3, 0.0])},
        {"w1": torch.tensor([-0.1, 2.0]), "b1": torch.tensor([3.0, -1.0])},
    ]
    p = {k: v.detach().numpy().copy() for k, v in params.items()}
    m = {k: np.zeros(2) for k in p}
    v = {k: np.zeros(2) for k in p}
    b1, b2, rate = 0.9, 0.999, 0.001
    for t, g in enumerate(grads, start=1):
        for name, tensor in params.items():
            tensor.grad = g[name].clone()
        net.step(params, memory, decay)
        for name in ("w1", "b1"):
            gn = g[name].numpy()
            m[name] = b1 * m[name] + (1 - b1) * gn
            v[name] = b2 * v[name] + (1 - b2) * gn * gn
            m_hat, v_hat = m[name] / (1 - b1**t), v[name] / (1 - b2**t)
            if name == "w1":  # a weight shrinks first; a bias does not
                p[name] = p[name] * (1 - rate * decay)
            p[name] = p[name] - rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        assert memory["step"] == t
        for name, tensor in params.items():
            assert tensor.detach().numpy() == pytest.approx(p[name], rel=1e-5, abs=1e-7)
            assert tensor.grad is None  # cleared for the next backward pass


def test_step5_a_first_step_moves_each_number_by_the_rate_whatever_its_gradient():
    params = two_parameters()
    memory = net.adam(params)
    params["w1"].grad = torch.tensor([1e-4, -500.0])
    params["b1"].grad = torch.tensor([7.0, -0.02])
    before = {k: v.detach().clone() for k, v in params.items()}
    net.step(params, memory, decay=0.0)
    for name in params:
        moved = (params[name].detach() - before[name]).abs()
        assert moved.numpy() == pytest.approx([net.RATE, net.RATE], rel=1e-3)


def test_step5_without_a_gradient_only_the_weights_shrink():
    params = two_parameters()
    memory = net.adam(params)
    for p in params.values():
        p.grad = torch.zeros(2)
    net.step(params, memory, decay=0.1)
    assert params["w1"].detach().tolist() == pytest.approx(
        [1.0 * (1 - 1e-4), -2.0 * (1 - 1e-4)]
    )
    assert params["b1"].detach().tolist() == [0.5, 0.5]


# --- step 6: learning -----------------------------------------------------------------------


def signal_table(seed: int, rows: int = 4_096, inputs: int = 8, kind: str = "linear"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, inputs))
    if kind == "linear":
        truth = x @ rng.normal(0, 0.5, size=(inputs, 2))
    else:  # |x0| is invisible to a straight line, and two ReLUs make it
        truth = np.repeat(np.abs(x[:, :1]) - math.sqrt(2 / math.pi), 2, axis=1)
    z = truth + rng.normal(0, 0.1, size=truth.shape)
    return Table(x, z), truth


def test_step6_each_epoch_sees_every_row_once_in_a_new_order(monkeypatch):
    monkeypatch.setattr(net, "BATCH", 100)
    table, _ = signal_table(5, rows=1_050)
    seen = []
    net.learn(table, 16, 0.01, 7, 3, lambda epoch, params: seen.append(epoch))
    assert seen == [1, 2, 3]
    batches = [len(i) for i in table.asked]
    assert batches == ([100] * 10 + [50]) * 3
    epochs = [torch.cat(table.asked[e * 11 : (e + 1) * 11]) for e in range(3)]
    for order in epochs:
        assert torch.equal(order.sort().values, torch.arange(1_050))
    assert not torch.equal(epochs[0], epochs[1])


def test_step6_a_seed_gives_the_same_network_and_another_seed_another(monkeypatch):
    monkeypatch.setattr(net, "BATCH", 256)

    def learned(seed):
        table, _ = signal_table(6, rows=1_024)
        return net.learn(table, 16, 0.01, seed, 2, lambda e, p: None)

    first, again, other = learned(1), learned(1), learned(2)
    assert all(torch.equal(first[k], again[k]) for k in NAMES)
    assert not torch.equal(first["w1"], other["w1"])


def r_squared(found: np.ndarray, truth: np.ndarray) -> float:
    return 1 - ((found - truth) ** 2).mean() / truth.var()


def test_step6_it_learns_a_straight_line(monkeypatch):
    monkeypatch.setattr(net, "BATCH", 256)
    monkeypatch.setattr(net, "RATE", 0.01)
    table, truth = signal_table(8)
    params = net.learn(table, 32, 0.0, 1, 30, lambda e, p: None)
    assert r_squared(net.forecast(params, table.x).numpy(), truth) > 0.95


def test_step6_its_hidden_layers_learn_what_a_straight_line_cannot(monkeypatch):
    monkeypatch.setattr(net, "BATCH", 256)
    monkeypatch.setattr(net, "RATE", 0.01)
    table, truth = signal_table(9, kind="curve")
    x = table.x.numpy().astype(np.float64)
    design = np.c_[x, np.ones(len(x))]
    line = design @ np.linalg.lstsq(design, truth, rcond=None)[0]
    assert r_squared(line, truth) < 0.05  # the best straight line sees nothing
    params = net.learn(table, 32, 0.0, 1, 30, lambda e, p: None)
    assert r_squared(net.forecast(params, table.x).numpy(), truth) > 0.8


def test_step6_weight_decay_pulls_the_weights_in(monkeypatch):
    monkeypatch.setattr(net, "BATCH", 256)
    monkeypatch.setattr(net, "RATE", 0.01)

    def size(decay):
        table, _ = signal_table(10, rows=2_048)
        params = net.learn(table, 16, decay, 1, 10, lambda e, p: None)
        return params["linear"].norm().item(), params["bias"].abs().sum().item()

    free, pulled = size(0.0), size(20.0)
    assert pulled[0] < 0.5 * free[0]


# --- step 7: the forecast, once learned -----------------------------------------------------


def test_step7_the_forecast_keeps_no_gradient():
    params = drawn(11, inputs=6, horizons=3, width=4)
    with torch.no_grad():
        params["linear"].add_(1.0)
    x = torch.randn(9, 6)
    found = net.forecast(params, x)
    assert found.shape == (9, 3) and not found.requires_grad
    assert torch.equal(found, net.forward(params, x).detach())


# --- with the harness -----------------------------------------------------------------------


def test_the_harness_runs_the_network_from_start_to_end(tmp_path, monkeypatch):
    harness_tests = loaded(
        "kutbit_net_harness_tests", Path(__file__).parent / "test_kutbit_net_harness.py"
    )
    nh = harness_tests.nh
    from threadpoolctl import threadpool_limits

    with threadpool_limits(4):  # the ridge's small solves crawl on many threads
        data = harness_tests.store_with_skill_runs(tmp_path)
        found, placebo = run_both(nh, data, monkeypatch)
    assert all(found[w]["step"] > 0 for w in (128, 512))
    assert placebo["choice"]["setting"] in (1, 2, 3, 4)
    assert set(placebo["results"]) == {
        "network",
        "ridge",
        "shuffled",
        "habit",
        "reversal",
    }
    assert math.isfinite(placebo["results"]["network"]["15"]["all"]["daily"]["mean"])


def run_both(nh, data, monkeypatch):
    """The harness's smoke run and its placebo, of two epochs and two seeds."""
    found = nh.smoke(net, data, log=lambda *_: None)
    monkeypatch.setattr(nh, "EPOCHS", 2)
    monkeypatch.setattr(nh, "SEEDS", (1, 2))
    return found, nh.placebo(net, data, log=lambda *_: None)
