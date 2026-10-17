"""A batched environment for learned intraday policies on the GPU (step A of the engine plan).

Every clean session of every ETF in a set runs side by side. At each of the rule's decisions,
every 30 minutes from the open (10:00 to 15:30 on a full day), a policy sees what was known
at the end of the minute before and chooses an exposure in [-2, 2], a multiple of the account,
long or short. The exposure is filled at the next minute's open (or the last price if that
minute has no bar), held to the next decision, and closed at the session's last price, as in
the rule's research and the engine. A day's return is the sum of the returns realised each
time the exposure changes, less the cost of each unit of exposure traded.

The observation, ten numbers at each decision, all known at the minute before it:

    0  (price - band top) / (open * sigma)       above the noise band, in band widths
    1  (price - band bottom) / (open * sigma)
    2  (price - VWAP) / (open * sigma)
    3  the move since the open / the 14-day volatility
    4  the last 30 minutes' move / the 14-day volatility
    5  the gap from the previous close / the 14-day volatility
    6  the time: minutes since the open / 390
    7  log(the 14-day volatility / 1%)
    8  the rule's position at this decision: -1, 0 or 1 (it follows its own path)
    9  the exposure the policy holds, before this decision

The rule itself is one policy: its position times min(2, 2% / the 14-day volatility). Run with
qqq_momentum.py's costs it must give that script's daily results exactly (check_rule).

Usage: gym.py NPZ_DIR QQQ_DAYS_PARQUET   (data/llmenv/bin/python): runs check_rule
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sim import DECISIONS, prepare

CAP, TARGET_VOL = 2.0, 0.02
FEATURES = 10


@dataclass
class Market:
    """Clean sessions of a set of ETFs, side by side: what the decisions need."""

    obs: torch.Tensor  # [sessions, decisions, 9]: features 0-8
    fill: torch.Tensor  # [sessions, decisions]: each decision's fill price
    decide: torch.Tensor  # [sessions, decisions]: whether the decision exists that day
    end: torch.Tensor  # [sessions]: the last price, where everything is closed
    rule_size: torch.Tensor  # [sessions]: min(2, 2% / the 14-day volatility)
    ticker: np.ndarray  # [sessions]: which ETF
    day: np.ndarray  # [sessions]: the session's date, ISO

    @property
    def sessions(self) -> int:
        return self.fill.shape[0]

    def select(self, keep: np.ndarray) -> "Market":
        """The sessions where `keep` is true, e.g. a range of years or one ETF."""
        k = torch.as_tensor(keep, device=self.fill.device)
        return Market(
            self.obs[k],
            self.fill[k],
            self.decide[k],
            self.end[k],
            self.rule_size[k],
            self.ticker[keep],
            self.day[keep],
        )


def rule_path(s: dict) -> torch.Tensor:
    """The rule's position at each decision of each session, [sessions, decisions]."""
    upper = s["hi"] * (1 + s["sigma"])
    lower = s["lo"] * (1 - s["sigma"])
    pos = torch.zeros(
        s["price"].shape[0], device=s["price"].device, dtype=s["price"].dtype
    )
    path = []
    for j in range(DECISIONS):
        p, up, lo, vw = s["price"][:, j], upper[:, j], lower[:, j], s["vwap"][:, j]
        flat_next = torch.where(p > up, 1.0, torch.where(p < lo, -1.0, 0.0))
        long_next = torch.where(
            p < torch.maximum(up, vw), torch.where(p < lo, -1.0, 0.0), 1.0
        )
        short_next = torch.where(
            p > torch.minimum(lo, vw), torch.where(p > up, 1.0, 0.0), -1.0
        )
        want = torch.where(
            pos == 0, flat_next, torch.where(pos == 1, long_next, short_next)
        )
        pos = torch.where(s["decide"][:, j], want, pos)
        path.append(pos)
    return torch.stack(path, 1)


def load(
    npz_dir: str, tickers: list[str], device: str = "cuda", dtype=torch.float32
) -> Market:
    parts = []
    for name in tickers:
        z = np.load(Path(npz_dir) / f"{name}.npz")
        s = prepare(str(Path(npz_dir) / f"{name}.npz"), device, dtype)
        first = torch.tensor(z["first_open"], device=device, dtype=dtype)[:, None]
        prev_close = torch.roll(
            torch.tensor(z["last_close"], device=device, dtype=dtype), 1
        )
        vol14 = s["vol14"][:, None]
        width = first * s["sigma"]
        steps = torch.arange(1, DECISIONS + 1, device=device) * 30
        # the price 30 minutes before each decision's minute: the decision before's price,
        # and the open for the first
        before30 = torch.cat([first, s["price"][:, :-1]], 1)
        obs = torch.stack(
            [
                (s["price"] - s["hi"] * (1 + s["sigma"])) / width,
                (s["price"] - s["lo"] * (1 - s["sigma"])) / width,
                (s["price"] - s["vwap"]) / width,
                (s["price"] / first - 1) / vol14,
                (s["price"] / before30 - 1) / vol14,
                (first / prev_close[:, None] - 1).expand_as(s["price"]) / vol14,
                (steps / 390.0).expand_as(s["price"]).to(dtype),
                torch.log(vol14 / 0.01).expand_as(s["price"]),
                rule_path(s),
            ],
            2,
        )
        keep = torch.as_tensor(z["clean"], device=device)
        parts.append(
            (
                obs[keep],
                s["fill"][keep],
                s["decide"][keep],
                s["end"][keep],
                torch.clamp(TARGET_VOL / s["vol14"][keep], max=CAP),
                np.full(int(keep.sum()), name),
                z["days"][z["clean"]],
            )
        )
    obs, fill, decide, end, size, ticker, day = zip(*parts, strict=True)
    market = Market(
        torch.cat(obs),
        torch.cat(fill),
        torch.cat(decide),
        torch.cat(end),
        torch.cat(size),
        np.concatenate(ticker),
        np.concatenate(day),
    )
    market.obs = torch.nan_to_num(market.obs, nan=0.0, posinf=0.0, neginf=0.0)
    return market


Policy = Callable[[torch.Tensor, torch.Tensor, Market], torch.Tensor]


def rollout(
    market: Market,
    policy: Policy,
    params: torch.Tensor,
    cost_per_unit: float = 0.4e-4,
    cost_per_share: float = 0.0,
    turnover: list | None = None,
) -> torch.Tensor:
    """Each variant's return on each session, [variants, sessions], after costs.

    If `turnover` is a list, the exposure traded on each session, [variants, sessions], is
    appended to it.

    `policy(obs, params, market)` gets obs [variants, sessions, 10] and returns an exposure
    [variants, sessions], clipped to [-2, 2]. A unit of exposure traded costs `cost_per_unit`
    of its value plus `cost_per_share` per share (in the account's currency at the fill
    price); the default, 0.4 bps, is QQQ's half spread of 0.3 bps and about 0.1 bps of fees.
    """
    variants, n = params.shape[0], market.sessions
    zeros = torch.zeros(variants, n, device=market.fill.device, dtype=market.fill.dtype)
    held, entry, total, traded = (
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
    )
    for j in range(DECISIONS):
        obs = torch.cat(
            [market.obs[:, j].expand(variants, n, FEATURES - 1), held[..., None]], 2
        )
        want = torch.clamp(policy(obs, params, market), -CAP, CAP)
        live = market.decide[:, j].expand(variants, n)
        want = torch.where(live, want, held)
        f = market.fill[:, j].expand(variants, n)
        change = want != held
        total = total + torch.where(change & (held != 0), held * (f / entry - 1), 0.0)
        cost = (want - held).abs() * (cost_per_unit + cost_per_share / f)
        total = total - torch.where(change, cost, 0.0)
        traded = traded + torch.where(change, (want - held).abs(), 0.0)
        entry = torch.where(change, f, entry)
        held = torch.where(change, want, held)
    end = market.end.expand(variants, n)
    total = total + torch.where(held != 0, held * (end / entry - 1), 0.0)
    total = total - held.abs() * (cost_per_unit + cost_per_share / end)
    if turnover is not None:
        turnover.append(traded + held.abs())
    return total


def rule_policy(
    obs: torch.Tensor, params: torch.Tensor, market: Market
) -> torch.Tensor:
    """The rule: its position times its size."""
    return obs[..., 8] * market.rule_size


def check_rule(npz_dir: str, reference: str) -> None:
    """The rule as a policy against qqq_momentum.py's stored days: return and turnover.

    Before costs, the rule's day must equal that script's gross return of a 1x position times
    its size at the 2x cap, and the exposure traded its turnover times the size. (Its costs
    price every share at the day's open; rollout prices them at each fill, so the costs
    differ by design.)
    """
    import polars as pl

    market = load(npz_dir, ["QQQ"], "cuda", torch.float64)
    one = torch.zeros(1, 1, device="cuda", dtype=torch.float64)
    traded: list = []
    gross = rollout(market, rule_policy, one, cost_per_unit=0.0, turnover=traded)[0]
    frame = pl.DataFrame(
        {
            "day": market.day,
            "gross": gross.cpu().numpy(),
            "traded": traded[0][0].cpu().numpy(),
        }
    ).with_columns(pl.col("day").str.to_date())
    ref = pl.read_parquet(reference).select("day", "gross_1x", "traded_1x", "size_2x")
    both = frame.join(ref, on="day", how="inner")
    want_gross = both["gross_1x"].to_numpy() * both["size_2x"].to_numpy()
    want_traded = both["traded_1x"].to_numpy() * both["size_2x"].to_numpy()
    d_gross = np.abs(both["gross"].to_numpy() - want_gross)
    d_traded = np.abs(both["traded"].to_numpy() - want_traded)
    print(
        f"the rule as a policy: {both.height} sessions against qqq_momentum.py at 2x; "
        f"largest difference in a day's gross return {d_gross.max():.2e}, in its turnover "
        f"{d_traded.max():.2e}; days differing by more than 1e-9: {(d_gross > 1e-9).sum()} "
        f"and {(d_traded > 1e-9).sum()}"
    )


if __name__ == "__main__":
    check_rule(sys.argv[1], sys.argv[2])
