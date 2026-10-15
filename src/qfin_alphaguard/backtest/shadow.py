"""The shadow record of a challenger against the rule (decision 0020).

Each runs in its own account through the engine (run_backtest) on the same minutes, with the
same broker, costs and Guard, so the difference between their days is the difference between
their decisions. The rule's account is the champion's; the challenger's never reaches a
broker. Their daily results, from the record's first day, go into the challenger's betting
test (evidence.py). In paper trading the same accounts run on live bars (Phase 3).
"""

from dataclasses import dataclass
from datetime import date

from qfin_alphaguard.backtest.engine import BacktestResult
from qfin_alphaguard.challengers import ChallengerCard
from qfin_alphaguard.evidence import Evidence


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
class ShadowRecord:
    """A challenger's record against the rule on the days both have."""

    days: tuple[date, ...]
    challenger: tuple[float, ...]
    rule: tuple[float, ...]
    evidence: Evidence


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
    return ShadowRecord(tuple(days), tuple(c), tuple(r), card.test().run(c, r))
