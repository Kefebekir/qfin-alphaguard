"""Efe's agent for KUTBIT's first variant (decision 0026), design A.

At an extreme minute it forecasts, for each hold, the stock's move apart from QQQ (its mean, the
edge, and its spread) and how much QQQ itself swings; it never forecasts QQQ's direction. From
these it values every action as the Kelly criterion does at Efe's risk appetite, the expected
return less gamma/2 times the variance, after costs, and takes the best one, or lets the move
be.

- A shared network learns afresh at each month's start, on every event before the month.
- It trusts its own edge only as far as it proved right: each month it learns on the earlier
  four fifths of the events before the month, and on the latest fifth, which it has not
  learned from, it measures how far the edge it forecast came true. That share, from none to
  all, is its trust, and the edge it acts on is the forecast times its trust.
- Each stock has its own correction of the edge, learned from every outcome as soon as it is
  known and pulled toward the shared network's view by `pull` events' worth of weight.

PyTorch is a tool for tensors and derivatives only: the layers, the loss and the learning loop
are written here.
"""

import math

import gym
import numpy as np
import torch

SETTINGS = [
    {"hidden": hidden, "pull": pull, "epochs": 100, "rate": 1e-3, "seed": 1}
    for hidden in (32, 64)
    for pull in (20, 100)
]
NU = 4.0
PERCENT = 100.0
BATCH = 256
CHECK = 0.2
STOCKS = 1000


SIDE = torch.tensor([a.side for a in gym.ACTIONS])
SIZE = torch.tensor([a.size for a in gym.ACTIONS])
HEDGE = torch.tensor([a.hedge for a in gym.ACTIONS])
HOLD = torch.tensor([gym.HOLDS.index(a.hold) if a.side else 0 for a in gym.ACTIONS])


def make(settings: dict) -> "Agent":
    return Agent(**settings)


class Agent:
    def __init__(self, hidden: int, pull: float, epochs: int, rate: float, seed: int):
        self.hidden, self.pull, self.epochs = hidden, pull, epochs
        self.rate, self.seed = rate, seed
        self.weights = None
        self.trust = torch.zeros(len(gym.HOLDS))
        self.sums = np.zeros((STOCKS, len(gym.HOLDS)))
        self.counts = np.zeros(STOCKS)

    # --- step 2: the network ----------------------------------------------------------

    def layer(self, n_in: int, n_out: int, generator) -> list:
        """A layer's weights, drawn so that a signal neither fades nor grows through it, and
        its biases at zero."""
        weight = torch.randn(n_in, n_out, generator=generator) * math.sqrt(2 / n_in)
        return [weight.requires_grad_(), torch.zeros(n_out, requires_grad=True)]

    def network(self, x: torch.Tensor) -> torch.Tensor:
        """For each event and hold: the edge, the log of its spread, and the log of QQQ's."""
        w1, b1, w2, b2, w3, b3 = self.weights
        h = torch.relu(x @ w1 + b1)
        h = torch.relu(h @ w2 + b2)
        return (h @ w3 + b3).reshape(len(x), len(gym.HOLDS), 3)

    def inputs(self, state: np.ndarray) -> torch.Tensor:
        """The 16 numbers in units of their spread over the events learned from."""
        x = (torch.as_tensor(state, dtype=torch.float32) - self.mean) / self.spread
        return x.clamp(-5, 5)

    # --- step 3: the loss and what it is taken on -------------------------------------

    @staticmethod
    def loss(x: torch.Tensor, mean, log_scale: torch.Tensor) -> torch.Tensor:
        """How unlikely the moves x are under a Student t of this mean and scale: its
        negative log density, less the parts that do not depend on them."""
        z = (x - mean) / log_scale.exp()
        return (log_scale + (NU + 1) / 2 * torch.log1p(z * z / NU)).mean()

    def targets(self, known: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The stock's move apart from QQQ and QQQ's move, over each hold, in percent."""
        own = torch.as_tensor(known["own"], dtype=torch.float32) * PERCENT
        qqq = torch.as_tensor(known["qqq"], dtype=torch.float32) * PERCENT
        beta = torch.as_tensor(known["beta"], dtype=torch.float32)[:, None]
        return own - beta * qqq, qqq

    # --- step 4: every minute ---------------------------------------------------------

    def update(self, known: dict) -> None:
        """Each outcome as soon as it is known: how far the stock's move apart from QQQ was
        from the shared network's edge, added to that stock's count."""
        if self.weights is None or not len(known["state"]):
            return
        with torch.no_grad():
            edge = self.trust * self.network(self.inputs(known["state"]))[..., 0]
        apart, _ = self.targets(known)
        miss = (apart - edge).numpy()
        np.add.at(self.sums, known["ticker"].astype(int), miss)
        np.add.at(self.counts, known["ticker"].astype(int), 1)

    def correction(self, tickers: np.ndarray) -> torch.Tensor:
        """Each stock's own edge beside the shared one: its mean miss, pulled toward nothing
        by `pull` events' worth of weight."""
        t = tickers.astype(int)
        found = self.sums[t] / (self.counts[t] + self.pull)[:, None]
        return torch.as_tensor(found, dtype=torch.float32)

    # --- step 5: learning -------------------------------------------------------------

    def fit(self, history: dict) -> None:
        """At the month's start: the shared network learns afresh on the earlier events before
        the month, its trust is measured on the latest ones, and every stock's correction is
        counted again from its events."""
        self.sums[:], self.counts[:] = 0, 0
        n = len(history["state"])
        split = int(n * (1 - CHECK))
        if split < BATCH or n - split < BATCH:
            self.weights = None
            return
        generator = torch.Generator().manual_seed(self.seed)
        state = torch.as_tensor(history["state"][:split], dtype=torch.float32)
        self.mean, self.spread = state.mean(0), state.std(0).clamp_min(1e-6)
        x = self.inputs(history["state"][:split])
        apart, qqq = self.targets({k: v[:split] for k, v in history.items()})
        self.weights = [
            *self.layer(x.shape[1], self.hidden, generator),
            *self.layer(self.hidden, self.hidden, generator),
            *self.layer(self.hidden, 3 * len(gym.HOLDS), generator),
        ]
        with torch.no_grad():  # the spreads start at the moves' usual size
            start = self.weights[5].view(len(gym.HOLDS), 3)
            start[:, 1] = apart.std(0).log()
            start[:, 2] = qqq.std(0).log()
        # Adam's running means of the gradients and of their squares
        first = [torch.zeros_like(w) for w in self.weights]
        second = [torch.zeros_like(w) for w in self.weights]
        step = 0
        for _ in range(self.epochs):
            order = torch.randperm(len(x), generator=generator)
            for begin in range(0, len(x), BATCH):
                i = order[begin : begin + BATCH]
                out = self.network(x[i])
                total = self.loss(apart[i], out[..., 0], out[..., 1])
                total = total + self.loss(qqq[i], 0.0, out[..., 2])
                gradients = torch.autograd.grad(total, self.weights)
                step += 1
                moments = zip(self.weights, gradients, first, second, strict=True)
                with torch.no_grad():
                    for w, g, m, v in moments:
                        m.mul_(0.9).add_(0.1 * g)
                        v.mul_(0.999).add_(0.001 * g * g)
                        m_hat = m / (1 - 0.9**step)
                        v_hat = v / (1 - 0.999**step)
                        w.sub_(self.rate * m_hat / (v_hat.sqrt() + 1e-8))
        self.trust = self.measure_trust({k: v[split:] for k, v in history.items()})
        self.update(history)

    def measure_trust(self, later: dict) -> torch.Tensor:
        """For each hold, how far the forecast edge came true on events not learned from:
        the slope of what followed on what was forecast, from none to all."""
        with torch.no_grad():
            edge = self.network(self.inputs(later["state"]))[..., 0]
        apart, _ = self.targets(later)
        slope = (edge * apart).sum(0) / (edge * edge).sum(0).clamp_min(1e-12)
        return slope.clamp(0.0, 1.0)

    # --- step 6: deciding -------------------------------------------------------------

    def forecast(self, events: dict) -> np.ndarray:
        """Every action's worth at each event: the expected return less gamma/2 times its
        variance, after costs; -inf where Efe's limits forbid it."""
        if self.weights is None:  # nothing learned yet: only letting the move be
            worth = np.full(events["allowed"].shape, -np.inf)
            worth[:, 0] = 0.0
            return worth
        with torch.no_grad():
            out = self.network(self.inputs(events["state"]))
        edge = (self.trust * out[..., 0] + self.correction(events["ticker"])) / PERCENT
        spread = NU / (NU - 2) / PERCENT**2
        var_apart = spread * (2 * out[..., 1]).exp()
        var_qqq = spread * (2 * out[..., 2]).exp()
        beta = torch.as_tensor(events["beta"], dtype=torch.float32)[:, None]
        mean = SIZE * SIDE * edge[:, HOLD]
        cost = SIZE * gym.COST + HEDGE * beta.abs() * SIZE * gym.COST_QQQ
        left = (1 - HEDGE) * beta
        variance = SIZE**2 * (var_apart[:, HOLD] + left**2 * var_qqq[:, HOLD])
        worth = (mean - cost - gym.GAMMA / 2 * variance).double().numpy()
        worth[:, 0] = 0.0
        return np.where(events["allowed"], worth, -np.inf)

    def act(self, events: dict) -> np.ndarray:
        """The best action at each event. Letting the move be is worth nothing, so it is the
        choice whenever no trade is worth more."""
        return self.forecast(events).argmax(axis=1)
