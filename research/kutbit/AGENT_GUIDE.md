# Writing KUTBIT's first agent: Efe's guide

This is design A of decision 0026's agent, which Efe chose on 20 November 2026. Everything
around it is built and tested: the event gym (`gym.py`), the harness that feeds the agent
minute by minute and measures it (`harness.py`), the bars and the placebo. What you write is
one file, `research/kutbit/agent.py`, in six steps. Each step has its tests, and a step is
done when they pass.

## The idea in four lines

1. At an extreme minute the network forecasts, for each hold of 5, 15 and 30 minutes, the
   stock's move apart from QQQ (its mean, the edge, and its spread) and how much QQQ itself
   will swing. It never forecasts QQQ's direction.
2. Every action's worth follows from these, as the Kelly criterion values a bet: the expected
   return, less γ/2 times its variance, less the costs. The agent takes the best action, or
   lets the move be when none is worth anything. A large size comes from a strong edge, and a
   hedge from a QQQ that is expected to swing.
3. It trusts its own edge only as far as it came true. Each month it learns on the earlier
   four fifths of the events before the month, and on the latest fifth it measures how much
   of its forecast edge came true: none to all. It acts on its forecast times that trust.
4. Each stock has its own correction of the edge, learned from every outcome as soon as it
   is known, and pulled toward the shared view until the stock has had enough events.

PyTorch is a tool for tensors and derivatives only (decision 0024, section 7). The layers,
the loss and the learning loop, Adam included, are written here; no `torch.nn`, no
`torch.optim`.

## Where it runs

PyTorch lives in `data/llmenv`, beside the project's environment, as it did for N6. Run every
test from the repository's root:

```bash
PYTHONPATH=src data/llmenv/bin/python -m pytest tests/test_kutbit_agent.py -k step1
```

In the project's environment and in CI the agent's tests are skipped, since PyTorch is not
there.

## Step 1: the frame

The settings chosen among on 2019, the constants, every action's columns, and the agent's
memory. `SETTINGS` are the four candidates: a hidden layer of 32 or 64, and a stock's own
correction pulled toward the shared view by 20 or 100 events' worth of weight.

```python
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
NU = 4.0  # the Student t's degrees of freedom: the market's moves have fat tails
PERCENT = 100.0  # moves are learned in percent, so the numbers stay near one
BATCH = 256  # events in each step of learning
CHECK = 0.2  # the latest share of the events before the month, kept for the trust
STOCKS = 1000  # room for every stock's correction

# every action's side, size, hedge and hold, as columns the decision can use at once
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
```

The methods of the next steps go inside `class Agent`, indented as shown. Check:
`-k step1`.

## Step 2: the network

Three layers: 16 inputs to `hidden`, `hidden` to `hidden`, and `hidden` to 9 outputs, three
for each hold: the edge, the log of its spread, the log of QQQ's. A layer's weights are drawn
with a spread of √(2 / inputs), so a signal neither fades nor grows as it passes through the
ReLUs. The 16 numbers are put in units of their spread over the events learned from, and
clamped at ±5 so a freak value cannot throw the network.

<!-- fmt: off -->
```python
    # --- step 2: the network -------------------------------------------------------------

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
```
<!-- fmt: on -->

Check: `-k step2`.

## Step 3: the loss

The network learns by making what followed as likely as it can under its forecast. Minute
moves have fat tails (0024 measured a kurtosis of 7 to 20), so the forecast is a Student t
with 4 degrees of freedom, not a normal: one crash minute does not tear the forecast apart.
The loss is the t's negative log density, less the parts that do not depend on the mean or
the scale.

<!-- fmt: off -->
```python
    # --- step 3: the loss ----------------------------------------------------------------

    @staticmethod
    def loss(x: torch.Tensor, mean, log_scale: torch.Tensor) -> torch.Tensor:
        """How unlikely the moves x are under a Student t of this mean and scale: its
        negative log density, less the parts that do not depend on them."""
        z = (x - mean) / log_scale.exp()
        return (log_scale + (NU + 1) / 2 * torch.log1p(z * z / NU)).mean()
```
<!-- fmt: on -->

Check: `-k step3`. One test compares it with PyTorch's own Student t, used only there.

## Step 4: learning

What the network learns from: the stock's move apart from QQQ (own less β times QQQ's) and
QQQ's move, in percent. At each month's start, `fit`:

1. keeps the latest fifth of the events before the month aside;
2. draws the network afresh from the seed, and starts its spreads at the moves' usual size;
3. learns on the earlier four fifths for `epochs` passes, in steps of `BATCH` events, with
   Adam written out: each weight moves by its gradient's running mean over the root of its
   running square, both corrected for starting at zero;
4. measures its trust on the fifth it kept aside: the slope of what followed on what it
   forecast, kept between none and all;
5. counts every stock's correction again from all the events before the month.

QQQ's loss has a mean of zero: the network learns how much QQQ swings, never where it goes.

<!-- fmt: off -->
```python
    # --- step 4: learning ----------------------------------------------------------------

    def targets(self, known: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The stock's move apart from QQQ and QQQ's move, over each hold, in percent."""
        own = torch.as_tensor(known["own"], dtype=torch.float32) * PERCENT
        qqq = torch.as_tensor(known["qqq"], dtype=torch.float32) * PERCENT
        beta = torch.as_tensor(known["beta"], dtype=torch.float32)[:, None]
        return own - beta * qqq, qqq

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
```
<!-- fmt: on -->

The events come in time order, so the fifth kept aside is the latest. Check: `-k step4`. Its
tests learn for real on small made-up events and take about a minute.

## Step 5: every minute

Each outcome, as soon as it is known, tells the agent how far that stock's move apart from
QQQ was from the shared edge it would have acted on. That miss is added to the stock's sum,
and the stock's correction is its sum over its count plus `pull`: with few events it is
nearly nothing, with many it is the stock's own mean miss.

<!-- fmt: off -->
```python
    # --- step 5: every minute ------------------------------------------------------------

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
```
<!-- fmt: on -->

Check: `-k step5`.

## Step 6: deciding

For each action at each event, with the edge `e` (the trusted forecast plus the stock's
correction), its variance `σ²` and QQQ's `σq²` for the action's hold:

- expected return: size × side × e;
- variance: size² × (σ² + ((1 − hedge) × β)² × σq²), since the hedge takes out its share of
  QQQ's part;
- costs: size × 5 bps for the stock, and hedge × β × size × 0.5 bps for QQQ;
- worth: expected return − costs − γ/2 × variance, with γ = 2.

A Student t of scale s has a variance of s² × ν / (ν − 2). An action Efe's limits forbid is
worth −∞; letting the move be is worth 0, so the agent trades only when an action is worth
more than nothing.

<!-- fmt: off -->
```python
    # --- step 6: deciding ------------------------------------------------------------------

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
```
<!-- fmt: on -->

Check: every test, `tests/test_kutbit_agent.py` whole. The last ones run the agent through
the harness on made-up events: it must find a planted edge, wait on noise, hedge where QQQ
swings and learn a stock that behaves unlike the others. The whole file takes about two
minutes.

## Then the runs, in this order

```bash
PYTHONPATH=src data/llmenv/bin/python research/kutbit/gym.py build
PYTHONPATH=src OMP_NUM_THREADS=8 data/llmenv/bin/python research/kutbit/harness.py placebo
PYTHONPATH=src OMP_NUM_THREADS=8 data/llmenv/bin/python research/kutbit/harness.py choose
PYTHONPATH=src OMP_NUM_THREADS=8 data/llmenv/bin/python research/kutbit/harness.py measure
```

A month's learning takes 4 to 10 seconds, so the placebo and the choice take about 5 to 10
minutes each, and the measurement, which also trains the agent on dealt outcomes, about as
long. `measure` refuses until the placebo has passed and the setting is chosen. Read
`data/kutbit/agent/choice.html` before the measurement, and `agent.html` after it.

## What to look at on the pages

- **Its worth against the bars:** the certainty equivalent in bps of the account an event,
  at 2, 5 and 10 bps a round trip. To pass, the agent must beat every bar.
- **Where it hedged:** the hedge it chose when QQQ was quiet and when QQQ moved. A good agent
  hedges little when QQQ is quiet. Above 2x, Efe's M of 2x makes it hedge at least half.
- **Its forecasts against what followed:** if its top tenths earned what it forecast, its
  trust is earned.
- **Where it left the best simple policy:** what its own reading of each event earned beyond
  the simple rule.
