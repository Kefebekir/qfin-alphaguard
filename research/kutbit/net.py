"""Efe's network for KUTBIT's second variant (decision 0027), design A.

For each horizon of 1, 5, 15, 30 and 60 minutes, it forecasts the skill test's `z`, a stock's
or an ETF's move apart from QQQ in its usual moves, from the same 971 inputs the ridge was
given, standardised. NETWORK_GUIDE.md explains it step by step.

- A linear path from the 971 inputs to the five horizons, as the ridge forecasts. Beside it,
  two hidden layers with ReLU, whose output is added to the linear path's: they can add only
  what a straight line misses, such as a pattern that holds when the day is wild but not when
  it is calm.
- The linear path and the hidden path's last layer start at zero. So the network starts by
  forecasting nothing, and its hidden layers add only what they find.
- It learns by the mean square error of the five horizons' `z`, with Adam written here: a
  rate of 0.001, betas of 0.9 and 0.999, and a weight decay on every weight but the biases,
  decoupled from the step. Batches are of BATCH rows, in a new random order each epoch, all
  drawn from the seed.

PyTorch is a tool for tensors and derivatives only (decision 0024, section 7): no `torch.nn`,
no `torch.optim`. The harness (net_harness.py) calls `learn` and `forecast`.
"""

import math

import torch

# --- step 1: the constants of decision 0027 -----------------------------------------------

BATCH = 4_096  # rows in each step of learning
RATE = 0.001  # how far each step of Adam goes
BETAS = (0.9, 0.999)  # how long Adam remembers a gradient, and its square
EPSILON = 1e-8  # keeps Adam from dividing by nothing
DECAYED = ("linear", "w1", "w2", "w3")  # the weights; the biases are not decayed


# --- step 2: the parameters -----------------------------------------------------------------


def init(inputs: int, horizons: int, width: int, generator, device) -> dict:
    """The network's parameters. The hidden layers' weights are drawn at He's scale, √(2 /
    their inputs), so a signal neither fades nor grows as it passes through a ReLU; the
    biases start at zero; the linear path and the hidden path's last layer start at zero."""

    def he(n_in: int, n_out: int) -> torch.Tensor:
        drawn = torch.randn(n_in, n_out, generator=generator, device=device)
        return drawn * math.sqrt(2 / n_in)

    params = {
        "linear": torch.zeros(inputs, horizons, device=device),
        "bias": torch.zeros(horizons, device=device),
        "w1": he(inputs, width),
        "b1": torch.zeros(width, device=device),
        "w2": he(width, width),
        "b2": torch.zeros(width, device=device),
        "w3": torch.zeros(width, horizons, device=device),
    }
    return {name: p.requires_grad_() for name, p in params.items()}


# --- step 3: the forecast -------------------------------------------------------------------


def forward(params: dict, x: torch.Tensor) -> torch.Tensor:
    """The forecast of each row of `x` for each horizon: the linear path, plus what the two
    hidden layers add."""
    hidden = torch.relu(x @ params["w1"] + params["b1"])
    hidden = torch.relu(hidden @ params["w2"] + params["b2"])
    return x @ params["linear"] + params["bias"] + hidden @ params["w3"]


# --- step 4: the loss -----------------------------------------------------------------------


def loss(forecast: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The mean square error over the rows and the five horizons."""
    return ((forecast - z) ** 2).mean()


# --- step 5: Adam, by hand ------------------------------------------------------------------


def adam(params: dict) -> dict:
    """Adam's memory: its step count, and for each parameter the running mean of its
    gradient (m) and of its gradient's square (v)."""
    return {
        "step": 0,
        "m": {name: torch.zeros_like(p) for name, p in params.items()},
        "v": {name: torch.zeros_like(p) for name, p in params.items()},
    }


def step(params: dict, memory: dict, decay: float) -> None:
    """One step of Adam with decoupled weight decay, from the gradients the last backward pass
    left; it clears them. A weight first shrinks by RATE x decay, then moves by RATE times its
    gradient's mean over the root of its square's mean, both corrected for starting at zero."""
    memory["step"] += 1
    t = memory["step"]
    b1, b2 = BETAS
    with torch.no_grad():
        for name, p in params.items():
            g = p.grad
            m, v = memory["m"][name], memory["v"][name]
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            if name in DECAYED:
                p.mul_(1 - RATE * decay)
            p.sub_(RATE * m_hat / (v_hat.sqrt() + EPSILON))
            p.grad = None


# --- step 6: learning -----------------------------------------------------------------------


def learn(rows, width: int, decay: float, seed: int, epochs: int, after_epoch) -> dict:
    """Learns the network on `rows` for `epochs` epochs, everything random drawn from `seed`:
    first the weights, then each epoch's order of the rows, in batches of BATCH. After each
    epoch, counted from 1, it calls after_epoch(epoch, params)."""
    generator = torch.Generator(device=rows.device).manual_seed(seed)
    params = init(rows.inputs, rows.horizons, width, generator, rows.device)
    memory = adam(params)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(rows.count, generator=generator, device=rows.device)
        for start in range(0, rows.count, BATCH):
            x, z = rows.batch(order[start : start + BATCH])
            loss(forward(params, x), z).backward()
            step(params, memory, decay)
        after_epoch(epoch, params)
    return params


# --- step 7: the forecast, once learned -----------------------------------------------------


def forecast(params: dict, x: torch.Tensor) -> torch.Tensor:
    """The forecasts of `x`, [rows, horizons], without keeping what a gradient would need."""
    with torch.no_grad():
        return forward(params, x)
