"""The shape of Guard's answer about one order intent, and the interface it meets.

Every intent a strategy makes passes Guard before it can become an order. The
rules are Phase 1, step 9; this module fixes what Guard answers, so the
backtest and the live engine can rely on it.
"""

from dataclasses import dataclass
from typing import Protocol

from qfin_alphaguard.events import OrderIntent, RiskAction, RiskEvent
from qfin_alphaguard.strategy import View


@dataclass(frozen=True)
class GuardDecision:
    """What may be sent for one intent, and why when it differs from the ask.

    - approve: the intent unchanged, no event
    - reduce: a smaller intent, with a REDUCE event saying why
    - reject: no intent, with a REJECT event
    - halt: no intent, with a HALT event; nothing is sent after it
    """

    intent: OrderIntent | None
    event: RiskEvent | None = None

    def __post_init__(self) -> None:
        if self.event is None:
            if self.intent is None:
                raise ValueError("a decision that sends nothing must say why")
            return
        if self.event.action is RiskAction.REDUCE and self.intent is None:
            raise ValueError("a REDUCE decision needs the reduced intent")
        if self.event.action is not RiskAction.REDUCE and self.intent is not None:
            raise ValueError(
                f"a {self.event.action.name} decision cannot send an intent"
            )


class RiskCheck(Protocol):
    """Anything that can judge an order intent: Guard, or a stand-in in tests."""

    def check(self, intent: OrderIntent, view: View) -> GuardDecision:
        """Guard's decision about `intent`, given what `view` shows."""
        ...
