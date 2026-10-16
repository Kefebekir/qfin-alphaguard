"""The shadow record of a challenger against the rule (decision 0020).

Each runs in its own account through the engine (run_backtest) on the same minutes, with the
same broker, costs and Guard, so the difference between their days is the difference between
their decisions. The rule's account is the champion's; the challenger's never reaches a
broker. Their daily results, from the record's first day, go into the challenger's betting
test (evidence.py). In paper trading the same accounts run on live bars (Phase 3).

The test compares them at equal risk. Beside it the record shows the challenger's raw results
too, as it traded (Efe's choice): describe() puts both side by side. Only the test decides.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from qfin_alphaguard.backtest.engine import BacktestResult
from qfin_alphaguard.challengers import ChallengerCard
from qfin_alphaguard.evidence import Evidence

TRADING_DAYS = 252


def daily_returns(result: BacktestResult, cash: float) -> dict[date, float]:
    """Each session's return: its closing value over the one before (over `cash` at first)."""
    returns, before = {}, cash
    for close in result.closes:
        if not before > 0:
            raise ValueError(f"the account was worth {before} before {close.day}")
        returns[close.day] = close.value / before - 1
        before = close.value
    return returns


@dataclass(frozen=True)
class Summary:
    """Daily returns as a year. Shown beside the test, never part of it."""

    days: int
    annual_return: float  # the mean day, times 252
    annual_volatility: float  # the days' standard deviation, times the root of 252
    total_return: float  # compounded over the days

    @property
    def sharpe(self) -> float:
        if not self.annual_volatility > 0:
            return math.nan
        return self.annual_return / self.annual_volatility


def summarise(returns: Sequence[float]) -> Summary:
    """The annual return, volatility and total of daily returns."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        raise ValueError("a summary needs at least 2 days")
    return Summary(
        days=len(r),
        annual_return=float(r.mean() * TRADING_DAYS),
        annual_volatility=float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        total_return=float(np.prod(1 + r) - 1),
    )


@dataclass(frozen=True)
class ShadowRecord:
    """A challenger's record against the rule on the days both have."""

    days: tuple[date, ...]
    challenger: tuple[float, ...]  # as it traded: its raw results
    rule: tuple[float, ...]
    scale: float  # the card's: the challenger at the rule's risk
    evidence: Evidence  # the test, at equal risk

    def summaries(self) -> dict[str, Summary]:
        """The rule, the challenger raw, and the challenger at the rule's risk."""
        return {
            "the rule": summarise(self.rule),
            "the challenger, raw": summarise(self.challenger),
            "the challenger at the rule's risk": summarise(
                [self.scale * c for c in self.challenger]
            ),
        }

    def mean_difference(self, equal_risk: bool) -> float:
        """The challenger's mean day less the rule's: raw, or at the rule's risk."""
        scale = self.scale if equal_risk else 1.0
        return float(
            np.mean(
                [scale * c - r for c, r in zip(self.challenger, self.rule, strict=True)]
            )
        )


def shadow_record(
    card: ChallengerCard,
    challenger: dict[date, float],
    rule: dict[date, float],
    until: date | None = None,
) -> ShadowRecord:
    """The record from the card's first day, to `until` or its retirement, and its evidence.

    Only days on which both accounts closed count; a day one of them lacks is left out of
    both, never filled.
    """
    ends = [d for d in (until, card.retired_on) if d is not None]
    last = min(ends) if ends else None
    days = sorted(
        d
        for d in challenger.keys() & rule.keys()
        if d >= card.record_from and (last is None or d <= last)
    )
    c = [challenger[d] for d in days]
    r = [rule[d] for d in days]
    return ShadowRecord(
        tuple(days), tuple(c), tuple(r), card.scale, card.test().run(c, r)
    )


def describe(record: ShadowRecord) -> str:
    """The record in words: the test's result, with the raw results beside it."""
    if len(record.days) < 2:
        return f"{len(record.days)} days: too few to describe"
    evidence = record.evidence
    lines = [
        f"{record.days[0]} to {record.days[-1]}, {len(record.days)} days",
        f"{'':36}{'a year':>8}{'volatility':>12}{'Sharpe':>8}{'in all':>9}",
    ]
    for name, s in record.summaries().items():
        lines.append(
            f"{name:36}{s.annual_return:>+8.1%}{s.annual_volatility:>12.1%}"
            f"{s.sharpe:>8.2f}{s.total_return:>+9.1%}"
        )
    lines.append(
        f"a day against the rule: raw {record.mean_difference(False) * 1e4:+.2f} bps, "
        f"at the rule's risk (x {record.scale}) "
        f"{record.mean_difference(True) * 1e4:+.2f} bps"
    )
    if evidence.promoted:
        verdict = f"promoted on {record.days[evidence.promoted_on]}"
    else:
        verdict = "not promoted"
    lines.append(
        f"the test, at the rule's risk: wealth {evidence.wealth[-1]:.3f} "
        f"(at most {max(evidence.wealth):.3f}; promoted at {evidence.threshold:g}): "
        f"{verdict}"
    )
    beyond = ", ".join(str(record.days[i]) for i in evidence.clipped) or "none"
    lines.append(f"days beyond the clip: {beyond}")
    return "\n".join(lines)
