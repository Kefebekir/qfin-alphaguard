"""Evolution strategies on the GPU gym, and the walk-forward score against the rule.

Efe's part is the network: a function policy(obs, params, market) -> exposure, and the number
of its parameters. This file trains it and scores it.

Training is OpenAI's evolution strategies (Salimans et al. 2017): each generation tries
`population` copies of the parameters, each nudged by noise in pairs (theta + sigma e and
theta - sigma e), runs them all on the training sessions at once, ranks them by fitness, and
moves theta towards the noise of the better ones (with Adam). Fitness is the annual Sharpe
ratio of a copy's returns over every training session, after costs.

Scoring is walk-forward, as in the news reader: for each test year, a model trained from
scratch on the 16 ETFs' sessions of the years before, tested on QQQ's sessions of that year,
sized to the rule's risk (the scale from QQQ's training sessions), and compared with the
rule day by day. Development: 2018-2021, as often as Efe likes. Holdout: 2022 to 9 October
2026, once, with the chosen settings committed in variants/G1_chosen.json; the result goes
to data/gpu_sim/holdout_G1.json and a second run is refused.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gym

TICKERS = ["QQQ", "SPY", "IWM", "DIA", "TLT", "GLD", "HYG", "XLB"]
TICKERS += ["XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEVELOPMENT, HOLDOUT = (2018, 2019, 2020, 2021), (2022, 2023, 2024, 2025, 2026)
LAST_DAY = "2026-10-09"
THRESHOLD = 2.0
VARIANTS = Path("research/gpu_sim/variants")
LOCKS = Path("data/gpu_sim")


def sharpe(returns: torch.Tensor) -> torch.Tensor:
    """Annual Sharpe ratio along the last axis; 0 for a copy that never moves."""
    std = returns.std(-1)
    return torch.where(std > 0, returns.mean(-1) / std * math.sqrt(252), 0.0)


def evolve(
    market: gym.Market,
    policy: gym.Policy,
    n_params: int,
    generations: int = 300,
    population: int = 256,
    sigma: float = 0.05,
    lr: float = 0.02,
    seed: int = 0,
    cost_per_unit: float = 0.4e-4,
) -> tuple[torch.Tensor, list[float]]:
    """Trained parameters, and the mean fitness of each generation."""
    if population % 2:
        raise ValueError("population must be even: the noise comes in pairs")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    theta = torch.zeros(n_params, device="cuda")
    m, v = torch.zeros_like(theta), torch.zeros_like(theta)
    history = []
    for step in range(1, generations + 1):
        half = torch.randn(
            population // 2, n_params, device="cuda", generator=generator
        )
        noise = torch.cat([half, -half])
        fitness = sharpe(
            gym.rollout(market, policy, theta + sigma * noise, cost_per_unit)
        )
        ranks = torch.empty_like(fitness)
        ranks[fitness.argsort()] = torch.arange(
            population, device="cuda", dtype=fitness.dtype
        )
        shaped = ranks / (population - 1) - 0.5  # centred ranks, from -0.5 to 0.5
        grad = (shaped[:, None] * noise).sum(0) / (population * sigma)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        theta = theta + lr * (m / (1 - 0.9**step)) / (
            torch.sqrt(v / (1 - 0.999**step)) + 1e-8
        )
        history.append(float(fitness.mean()))
    return theta, history


def walk_forward(
    policy: gym.Policy,
    n_params: int,
    years: tuple[int, ...],
    settings: dict,
    npz_dir: str = "data/gpu_sim",
    record: str | None = None,
) -> dict:
    """Train on the 16 ETFs before each year, score on QQQ that year against the rule.

    With `record`, a JSON file of the run's details for report.py: each year's mean fitness
    at every generation, and its test days with the network's and the rule's daily returns.
    """
    market = gym.load(npz_dir, TICKERS)
    year = np.array([int(d[:4]) for d in market.day])
    qqq = market.ticker == "QQQ"
    per_year, diffs, details = {}, [], {}
    for test_year in years:
        train = market.select((year < test_year) & (year >= 2016))
        test = market.select(qqq & (year == test_year) & (market.day <= LAST_DAY))
        theta, history = evolve(train, policy, n_params, **settings)
        params = theta[None, :]
        one = torch.zeros(1, 1, device="cuda")
        cost = settings.get("cost_per_unit", 0.4e-4)
        qqq_train = market.select(qqq & (year < test_year) & (year >= 2016))
        model_train = gym.rollout(qqq_train, policy, params, cost)[0]
        rule_train = gym.rollout(qqq_train, gym.rule_policy, one, cost)[0]
        scale = (
            (rule_train.std() / model_train.std()).item()
            if model_train.std() > 0
            else 0.0
        )
        model = gym.rollout(test, policy, params, cost)[0] * scale
        rule = gym.rollout(test, gym.rule_policy, one, cost)[0]
        diff = (model - rule).cpu().numpy()
        diffs.append(diff)
        details[test_year] = {
            "fitness": history,
            "days": list(test.day),
            "model": model.cpu().numpy().tolist(),
            "rule": rule.cpu().numpy().tolist(),
        }
        per_year[test_year] = {
            "days": len(diff),
            "mean_bps": float(diff.mean() * 1e4),
            "t": float(diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))),
            "model_sharpe": float(sharpe(model[None])[0]),
            "rule_sharpe": float(sharpe(rule[None])[0]),
            "scale": scale,
            "last_fitness": history[-1],
        }
    d = np.concatenate(diffs)
    t = float(d.mean() / d.std(ddof=1) * math.sqrt(len(d)))
    result = {
        "days": len(d),
        "mean_bps": float(d.mean() * 1e4),
        "t": t,
        "passes": t >= THRESHOLD,
        "years": per_year,
        "settings": settings,
    }
    if record is not None:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        Path(record).write_text(json.dumps(result | {"details": details}))
    return result


def run_holdout(
    policy: gym.Policy, n_params: int, settings: dict, network: dict
) -> dict:
    """The holdout, once: the chosen settings must be committed, and no earlier run.

    `network` is the network's own settings (e.g. {"hidden": 16, "see_rule": True}), and
    `settings` evolve's; both must equal those committed in variants/G1_chosen.json.
    """
    chosen, lock = VARIANTS / "G1_chosen.json", LOCKS / "holdout_G1.json"
    if lock.exists():
        raise RuntimeError(f"G1's holdout has been run: {lock}")
    if not chosen.exists():
        raise RuntimeError(f"write and commit {chosen} with the chosen settings first")
    committed = json.loads(chosen.read_text())
    if committed.get("settings") != settings or committed.get("network") != network:
        raise RuntimeError(f"the settings differ from those committed in {chosen}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "research/gpu_sim"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "commit the network, the settings and the harness before the holdout"
        )
    result = walk_forward(policy, n_params, HOLDOUT, settings) | {
        "chosen": json.loads(chosen.read_text()),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
    }
    lock.write_text(json.dumps(result, indent=2))
    return result
