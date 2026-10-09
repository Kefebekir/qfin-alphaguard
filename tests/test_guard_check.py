from datetime import UTC, datetime

import pytest

from qfin_alphaguard.events import OrderIntent, RiskAction, RiskEvent, Side
from qfin_alphaguard.guard.check import GuardDecision

T = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)
INTENT = OrderIntent("A", Side.BUY, 10, 100.0, T)
SMALLER = OrderIntent("A", Side.BUY, 5, 100.0, T)


def event(action):
    return RiskEvent(T, "max_order_usd", action, "test", "A")


@pytest.mark.parametrize(
    ("intent", "action"),
    [
        (INTENT, None),  # approve
        (SMALLER, RiskAction.REDUCE),
        (None, RiskAction.REJECT),
        (None, RiskAction.HALT),
    ],
)
def test_the_four_answers_guard_can_give(intent, action):
    decision = GuardDecision(intent, event(action) if action else None)
    assert decision.intent is intent


@pytest.mark.parametrize(
    ("intent", "action"),
    [
        (None, None),  # nothing sent and no reason
        (None, RiskAction.REDUCE),  # reduced to nothing
        (INTENT, RiskAction.REJECT),  # rejected but sent
        (INTENT, RiskAction.HALT),  # halted but sent
    ],
)
def test_answers_that_contradict_themselves_are_refused(intent, action):
    with pytest.raises(ValueError):
        GuardDecision(intent, event(action) if action else None)
