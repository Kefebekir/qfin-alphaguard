"""N6's model: Efe's network on N6's frozen eyes (decisions page of 22 October 2026).

A member-day's headlines, each a row of N6's eyes (4,096 numbers in 202412's scaled reading):
- a learned projection to `dim` numbers, which N3 and N4 took from PCA instead; it starts
  from the training headlines' first `dim` principal components, each scaled to unit spread
  (start="pca"), or at random (start="random");
- with each headline's log(1 + age), edit and two-tag flags, attention pools them into one
  vector: a learned score a headline, softmaxed over the member-day's headlines;
- the member's and the market's context through a small network, the weekday and the sector
  embedded; with inputs="news" (amendment 7) the context is left out;
- a head on the pooled news, log(1 + count) and the context gives the forecast; a second head
  the size of the move (|target|), weighted by `aux` in the loss.

Losses: "mse" on N4's scaled target, or "sharpe": minus the long-short Sharpe ratio over the
batch's days, each day's forecasts made weights (less their mean, a gross of 2) on its raw next
returns. Training: Adam with weight decay, batches of whole days in a random order, a fixed
number of epochs. The headline store sits on the GPU in float16; only the batch's headlines go
through the projection.

Runs in data/llmenv (PyTorch); its tests: data/llmenv/bin/python -m pytest
research/news_reader/test_n6_model.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from n6_harness import NO_SECTOR, Arrays, market_scaled

WIDTH = 4096
MAX_PARAMETERS = 1_100_000  # decision 6, amendment 8: up to dim 256
LOSSES = ("mse", "sharpe")
STARTS = ("pca", "random")
INPUTS = ("all", "news")  # amendment 7: "news" leaves the context out
PCA_ROWS = 50_000  # training headlines the starting components are taken from


@dataclass(frozen=True)
class Settings:
    """Efe's levers; the defaults are the decisions page's recommendations."""

    dim: int = 32  # the projection's size
    attention: int = 16
    context: int = 32
    hidden: int = 32
    dropout: float = 0.1
    loss: str = "mse"
    aux: float = 0.1
    lr: float = 1e-3
    decay: float = 1e-4
    epochs: int = 10
    days_per_batch: int = 8
    start: str = "pca"
    inputs: str = "all"

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss is one of {LOSSES}, got {self.loss!r}")
        if self.start not in STARTS:
            raise ValueError(f"start is one of {STARTS}, got {self.start!r}")
        if self.inputs not in INPUTS:
            raise ValueError(f"inputs is one of {INPUTS}, got {self.inputs!r}")
        for name in (
            "dim",
            "attention",
            "context",
            "hidden",
            "epochs",
            "days_per_batch",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 <= self.dropout < 1 or self.aux < 0 or self.lr <= 0 or self.decay < 0:
            raise ValueError("dropout in [0, 1), aux and decay at least 0, lr above 0")


class N6(nn.Module):
    def __init__(self, s: Settings, members: int, markets: int, events: int):
        super().__init__()
        self.project = nn.Linear(WIDTH, s.dim)
        self.score = nn.Sequential(
            nn.Linear(s.dim + 3, s.attention), nn.Tanh(), nn.Linear(s.attention, 1)
        )
        self.uses_context = s.inputs == "all"
        if self.uses_context:  # made in the same order as before amendment 7
            self.weekday = nn.Embedding(5, 2)
            self.sector = nn.Embedding(NO_SECTOR + 1, 4)
            self.context = nn.Sequential(
                nn.Linear(members + markets + events + 2 + 4, s.context),
                nn.GELU(),
                nn.Dropout(s.dropout),
                nn.Linear(s.context, s.context),
                nn.GELU(),
            )
        fused = s.dim + 3 + 1 + (s.context if self.uses_context else 0)
        self.head = nn.Sequential(
            nn.Dropout(s.dropout),
            nn.Linear(fused, s.hidden),
            nn.GELU(),
            nn.Linear(s.hidden, 1),
        )
        self.size = nn.Linear(fused, 1)

    def pool(self, store, heads, info) -> torch.Tensor:
        """Each member-day's headlines pooled by attention; zeros where it has none."""
        mask = heads >= 0
        where = mask.nonzero(as_tuple=True)
        projected = self.project(store[heads[where].long()].float())
        h = torch.zeros(*heads.shape, projected.shape[1], device=heads.device)
        h[where] = projected
        h = torch.cat([h, info], dim=-1)
        score = self.score(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(score, dim=1) * mask
        w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
        return (w.unsqueeze(-1) * h).sum(1)

    def forward(self, store, x: dict) -> tuple[torch.Tensor, torch.Tensor]:
        news = self.pool(store, x["heads"], x["info"])
        parts = [news, x["count"].unsqueeze(1)]
        if self.uses_context:
            parts.append(
                self.context(
                    torch.cat(
                        [
                            x["member"],
                            x["market"],
                            x["events"],
                            self.weekday(x["weekday"]),
                            self.sector(x["sector"]),
                        ],
                        dim=1,
                    )
                )
            )
        z = torch.cat(parts, dim=1)
        return self.head(z).squeeze(1), self.size(z).squeeze(1)


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inputs(a: Arrays, trained: np.ndarray, device) -> dict[str, torch.Tensor]:
    """Every member-day's inputs on `device`, the market scaled on the training days."""

    def t(v, dtype=None):
        return torch.as_tensor(v, device=device, dtype=dtype)

    return {
        "heads": t(a.heads),
        "info": t(a.head_info),
        "member": t(a.member),
        "market": t(market_scaled(a.market, a.day, trained)),
        "events": t(a.events),
        "weekday": t(a.weekday),
        "sector": t(a.sector),
        "count": t(a.count),
        "y": t(np.nan_to_num(a.y), torch.float32),
        "ret": t(np.nan_to_num(a.days["next_return"].to_numpy()), torch.float32),
        "day": t(a.day),
    }


def take(x: dict, rows: torch.Tensor) -> dict:
    return {k: v[rows] for k, v in x.items()}


def sharpe_loss(forecast, ret, day) -> torch.Tensor:
    """Minus the long-short Sharpe ratio over the days in the batch: each day's forecasts
    less their mean, scaled to a gross of 2, as weights on its raw next returns."""
    days, at = torch.unique(day, return_inverse=True)
    n = len(days)
    mean = torch.zeros(n, device=forecast.device).index_add(0, at, forecast)
    mean = mean / torch.bincount(at, minlength=n)
    centred = forecast - mean[at]
    gross = torch.zeros(n, device=forecast.device).index_add(0, at, centred.abs())
    w = 2 * centred / gross[at].clamp_min(1e-9)
    daily = torch.zeros(n, device=forecast.device).index_add(0, at, w * ret)
    return -daily.mean() / (daily.std() + 1e-6)


def loss_of(s: Settings, forecast, size, batch) -> torch.Tensor:
    main = (
        ((forecast - batch["y"]) ** 2).mean()
        if s.loss == "mse"
        else sharpe_loss(forecast, batch["ret"], batch["day"])
    )
    return main + s.aux * ((size - batch["y"].abs()) ** 2).mean()


def batches(day: np.ndarray, rows: np.ndarray, per: int, rng=None) -> list[np.ndarray]:
    """The rows in groups of `per` whole days; days in a random order when rng is given."""
    chosen = np.flatnonzero(rows)
    days = np.unique(day[chosen])
    if rng is not None:
        days = rng.permutation(days)
    groups = []
    for k in range(0, len(days), per):
        groups.append(chosen[np.isin(day[chosen], days[k : k + per])])
    return groups


@torch.no_grad()
def pca_start(store, rows: np.ndarray, dim: int, seed: int) -> tuple:
    """The projection onto the first `dim` principal components of the headlines in
    `rows` (at most PCA_ROWS of them), each scaled to unit spread: (weight, bias)."""
    rows = np.unique(rows)
    if len(rows) > PCA_ROWS:
        rows = np.sort(
            np.random.default_rng(seed).choice(rows, PCA_ROWS, replace=False)
        )
    x = store[torch.as_tensor(rows, device=store.device).long()].double()
    mean = x.mean(0)
    x = x - mean
    values, vectors = torch.linalg.eigh(x.T @ x / (len(rows) - 1))
    top = vectors[:, -dim:].flip(1).T  # [dim, WIDTH], the largest first
    spread = values[-dim:].flip(0).clamp_min(1e-12).sqrt()
    weight = top / spread[:, None]
    return weight.float(), (-weight @ mean).float()


def fit(a: Arrays, trained: np.ndarray, s: Settings, seed: int, store, x=None) -> N6:
    """A model trained on the `trained` member-days, from `seed`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = store.device
    x = x if x is not None else inputs(a, trained, device)
    model = N6(s, a.member.shape[1], a.market.shape[1], a.events.shape[1]).to(device)
    if parameters(model) > MAX_PARAMETERS:
        raise ValueError(
            f"{parameters(model):,} parameters, more than {MAX_PARAMETERS:,} "
            "(decision 6, amendment 8)"
        )
    rows = a.heads[trained]
    rows = rows[rows >= 0]
    if (
        s.start == "pca" and len(np.unique(rows)) > s.dim
    ):  # the context-alone placebo has none
        weight, bias = pca_start(store, rows, s.dim, seed)
        model.project.weight.data.copy_(weight)
        model.project.bias.data.copy_(bias)
    optimiser = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=s.decay)
    model.train()
    for _ in range(s.epochs):
        for rows in batches(a.day, trained, s.days_per_batch, rng):
            batch = take(x, torch.as_tensor(rows, device=device))
            forecast, size = model(store, batch)
            loss = loss_of(s, forecast, size, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


@torch.no_grad()
def predict(model: N6, a: Arrays, rows: np.ndarray, store, x: dict) -> np.ndarray:
    """The model's forecasts for the `rows` member-days, in their order."""
    model.eval()
    out = np.empty(len(a.day), dtype=np.float64)
    for group in batches(a.day, rows, 64):
        forecast, _ = model(store, take(x, torch.as_tensor(group, device=store.device)))
        out[group] = forecast.double().cpu().numpy()
    return out[rows]


def ensemble(a: Arrays, trained, validated, s: Settings, seeds, store) -> np.ndarray:
    """The mean forecast of one model a seed, each trained on the same days."""
    x = inputs(a, trained, store.device)
    forecasts = [
        predict(fit(a, trained, s, seed, store, x), a, validated, store, x)
        for seed in seeds
    ]
    return np.mean(forecasts, axis=0)
