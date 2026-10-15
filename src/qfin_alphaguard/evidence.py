"""Evidence that a challenger beats the rule: a bet that stays valid however often it is read.

The shadow record (decision 0020) runs each frozen challenger in its own account, with the
rule's costs and Guard, and compares its daily result with the rule's. A t-test on that
record is valid only if it is read once, at a horizon fixed in advance; a record that is
looked at every evening needs a test that allows it. This is one: testing by betting
(Shafer and Vovk; Waudby-Smith and Ramdas 2024).

Each day t:

    d_t = scale * challenger_t - rule_t       the challenger at the rule's risk
    x_t = clip(d_t, -clip, clip) / clip       in [-1, 1]
    W_t = W_(t-1) * (1 + bet_t * x_t)         W_0 = 1

The bet is a fraction in [0, max_bet], set from the days before t only. The null hypothesis
is that the challenger does not beat the rule: every day, given what came before, the mean
of x_t is at most 0. Then W never grows on average (it is a nonnegative supermartingale),
and by Ville's inequality the chance that it ever reaches 1/alpha is at most alpha, however
often it is checked and whenever the record is stopped. A challenger is promoted when W
reaches 1 / alpha. Its alpha is its share of the platform's (challengers.TOTAL_ALPHA), fixed on
its card before its record starts; the shares of every challenger ever given a record add to
at most that total, so the chance of promoting any challenger that does not beat the rule is
at most the total too, in whatever order the challengers were added.

The bet is aGRAPA (Waudby-Smith and Ramdas 2024): the fraction that maximises the expected
growth of W for the mean and variance of x seen so far, mean / (variance + mean^2), never
below 0 (a bet against the challenger cannot promote it) and never above max_bet.

The guarantee is for the clipped difference. A challenger whose losses come rarely and beyond
the clip gains from it: one that earns a little every day and loses 10% about once a year,
nothing on average, is promoted in about half of 12-year records at a clip of 5%, and almost
never at 10% (research/step5/evidence_power.py). A clip far above the usual difference costs
time instead. So each card's clip comes from its training days (training_settings): the
largest difference there, so that no training day is clipped, and a record day beyond it is
a difference the challenger never showed in training. Evidence.clipped lists those days.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PRIOR_VARIANCE = 0.25  # of x before any day is seen: x lies in [-1, 1]


@dataclass(frozen=True)
class Evidence:
    """A challenger's record against the rule, day by day."""

    outcomes: tuple[float, ...]  # x_t, in [-1, 1]
    bets: tuple[float, ...]  # bet_t, set before day t
    wealth: tuple[float, ...]  # W_t after day t
    threshold: float  # 1 / alpha
    promoted_on: int | None  # the first day W reached the threshold, if it did
    clipped: tuple[int, ...]  # the days whose difference was beyond the clip

    @property
    def promoted(self) -> bool:
        return self.promoted_on is not None


@dataclass(frozen=True)
class BettingTest:
    """The test of one challenger. Its settings are fixed when the challenger is frozen."""

    # the rule's volatility over the challenger's, from its training days
    scale: float
    # the largest daily difference counted in full, from its training days
    clip: float
    # the chance of promoting this challenger if it does not beat the rule
    alpha: float = 0.025
    max_bet: float = 0.5  # W can lose at most this share in a day

    def __post_init__(self) -> None:
        for name in ("scale", "clip"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be a finite number above 0, got {value!r}"
                )
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha!r}")
        if not (0 < self.max_bet < 1):
            raise ValueError(f"max_bet must be between 0 and 1, got {self.max_bet!r}")

    @property
    def threshold(self) -> float:
        return 1 / self.alpha

    def outcomes(
        self, challenger: Sequence[float], rule: Sequence[float]
    ) -> np.ndarray:
        """x_t for each day: the clipped difference at equal risk, in [-1, 1]."""
        d = differences(self.scale, challenger, rule)
        return np.clip(d, -self.clip, self.clip) / self.clip

    def run(self, challenger: Sequence[float], rule: Sequence[float]) -> Evidence:
        """The evidence after each day of a record, both lists in day order."""
        x = self.outcomes(challenger, rule)
        d = differences(self.scale, challenger, rule)
        bets, wealth = betting_wealth(x, self.max_bet)
        crossed = np.flatnonzero(wealth >= self.threshold)
        return Evidence(
            outcomes=tuple(x.tolist()),
            bets=tuple(bets.tolist()),
            wealth=tuple(wealth.tolist()),
            threshold=self.threshold,
            promoted_on=int(crossed[0]) if len(crossed) else None,
            clipped=tuple(np.flatnonzero(np.abs(d) > self.clip).tolist()),
        )


def differences(
    scale: float, challenger: Sequence[float], rule: Sequence[float]
) -> np.ndarray:
    """d_t for each day: the challenger at the rule's risk, less the rule."""
    c = np.asarray(challenger, dtype=float)
    r = np.asarray(rule, dtype=float)
    if c.shape != r.shape or c.ndim != 1:
        raise ValueError(f"{len(c)} challenger days but {len(r)} rule days")
    if not (np.isfinite(c).all() and np.isfinite(r).all()):
        raise ValueError("daily results must be finite numbers")
    return scale * c - r


def training_settings(
    challenger: Sequence[float], rule: Sequence[float]
) -> tuple[float, float]:
    """The scale and the clip for a challenger's card, from its training days.

    The scale puts the challenger at the rule's risk: the standard deviation of the rule's
    daily results over the challenger's, to 4 decimals. The clip is the largest difference
    at that scale on those days, rounded up to 0.1%, so that no training day is clipped.
    """
    c = np.asarray(challenger, dtype=float)
    r = np.asarray(rule, dtype=float)
    if len(c) < 2:
        raise ValueError("training needs at least 2 days")
    differences(1.0, c, r)  # the same checks
    if not (c.std(ddof=1) > 0 and r.std(ddof=1) > 0):
        raise ValueError("the challenger and the rule must each have some risk")
    scale = round(float(r.std(ddof=1) / c.std(ddof=1)), 4)
    largest = float(np.abs(differences(scale, c, r)).max())
    if largest == 0:
        raise ValueError("the challenger is the rule at equal risk: nothing to test")
    return scale, round_up(largest)


def round_up(value: float) -> float:
    """value rounded up to 0.1%: the smallest multiple of 0.001 not below it."""
    steps = math.ceil(value * 1000)
    if steps / 1000 < value:  # value * 1000 rounded down, as for just above 0.043
        steps += 1
    return steps / 1000


def betting_wealth(
    x: np.ndarray, max_bet: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """The bets and the wealth after each day, for outcomes x in [-1, 1] along the last axis.

    Earlier axes are separate records (used to check the test by simulation). The bet on
    day t uses x before t only: the mean of those days, and their spread around the running
    mean, each with one prior day of mean 0 and variance PRIOR_VARIANCE.
    """
    x = np.asarray(x, dtype=float)
    days = x.shape[-1]
    bets = np.empty_like(x)
    wealth = np.empty_like(x)
    total = np.zeros(x.shape[:-1])  # the sum of x so far
    # the sum of squared surprises, starting from the prior
    spread = np.full(x.shape[:-1], PRIOR_VARIANCE)
    w = np.ones(x.shape[:-1])
    for t in range(days):
        mean = total / (t + 1)
        variance = spread / (t + 1)
        bet = np.clip(mean / (variance + mean**2), 0.0, max_bet)
        bets[..., t] = bet
        w = w * (1 + bet * x[..., t])
        wealth[..., t] = w
        spread = spread + (x[..., t] - mean) ** 2
        total = total + x[..., t]
    return bets, wealth
