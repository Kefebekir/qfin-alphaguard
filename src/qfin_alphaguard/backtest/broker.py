"""What the backtest needs from a simulated broker.

The simulated broker itself, with its fill rules and cost model, is Phase 1,
step 6. The engine checks every fill it reports against the contract below and
stops the backtest if one breaks it.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from qfin_alphaguard.events import Bar, Fill, Order


class SimulatedBroker(Protocol):
    """Fills orders from the bars that follow them.

    The contract, which the engine enforces:

    - an order can fill only on a bar that starts at or after its `sent_at`,
      so never on the bar it was decided from
    - a fill's time lies within its bar, and its price within the bar's low
      and high: a price that really traded
    - a buy fills at or below its limit price, a sell at or above it
    - fills for an order never add up to more than its quantity
    - a cancelled order never fills again
    """

    def submit(self, order: Order) -> None:
        """Take an order; it may fill from the next bar on."""
        ...

    def cancel(self, client_order_id: str, time: datetime) -> None:
        """Take an open order off the market at `time`."""
        ...

    def on_bar(self, bar: Bar) -> Sequence[Fill]:
        """The fills this bar brings for orders in `bar.ticker`."""
        ...
