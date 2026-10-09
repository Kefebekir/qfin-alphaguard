"""What the backtest needs from a simulated broker.

NextBarBroker below is the simulated broker, with its fill rules and costs
(Phase 1, step 6). The engine checks every fill it reports against the
contract in SimulatedBroker and stops the backtest if one breaks it.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from qfin_alphaguard.events import Bar, Fill, Order, Side


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


@dataclass(frozen=True)
class CostModel:
    """What a fill costs besides its price. Defaults: IBKR Tiered (decision 0002)."""

    per_share: float = 0.0035  # USD per share, up to 300,000 shares a month
    minimum: float = 0.35  # USD per order, paid on its first fill
    maximum_share: float = 0.01  # commission is at most 1% of a fill's value
    fx_bps: float = 0.0  # currency conversion: 0 when USD is held, 15 at Trading 212
    half_spread_bps: float = 1.0  # what a marketable order pays over the traded price

    def __post_init__(self) -> None:
        for name in (
            "per_share",
            "minimum",
            "maximum_share",
            "fx_bps",
            "half_spread_bps",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")

    def commission(self, quantity: int, price: float, first_fill: bool) -> float:
        """Commission and currency conversion for one fill, in USD.

        - the broker charges per_share for each share; the first fill of an
          order pays at least `minimum`; and the commission is at most
          maximum_share of the fill's value (quantity * price)
        - currency conversion adds fx_bps of the fill's value on top
        """
        value = quantity * price
        broker = self.per_share * quantity
        if first_fill:
            broker = max(broker, self.minimum)
        broker = min(broker, self.maximum_share * value)
        return broker + self.fx_bps / 10_000 * value


class NextBarBroker:
    """The simulated broker for backtests. Fills limit orders from the bars."""

    def __init__(
        self, costs: CostModel | None = None, participation: float = 0.1
    ) -> None:
        if not (0 < participation <= 1):
            raise ValueError(f"participation must be in (0, 1], got {participation!r}")
        self.costs = costs or CostModel()
        self.participation = participation  # most of a bar's volume one order takes
        self.waiting: dict[str, Order] = {}  # open orders by id, oldest first
        self.filled: dict[str, int] = {}  # shares filled so far, by order id

    def submit(self, order: Order) -> None:
        self.waiting[order.client_order_id] = order
        self.filled[order.client_order_id] = 0

    def cancel(self, client_order_id: str, time: datetime) -> None:
        del self.waiting[client_order_id]

    def on_bar(self, bar: Bar) -> list[Fill]:
        """Fills for the waiting orders in `bar.ticker`, oldest order first.

        An order takes part only if it was sent at or before `bar.start`.

        Price, for a buy with limit L (a sell is the mirror image):
        - if the bar opens at or below L, the order was marketable on arrival:
          it fills at the open plus half the spread, but never above L or above
          the bar's high; its time is the bar's start
        - otherwise, if the price fell below L during the bar, it fills at L;
          its time is the bar's end
        - a price that only touches L does not fill it: other orders wait at L
          before ours



        Quantity: what is left of the order, but at most `participation` of the
        bar's volume, rounded down; nothing if that is 0. The rest waits.

        Commission: self.costs.commission(quantity, price, first_fill), where
        first_fill says whether this is the order's first fill. An order that
        is completely filled stops waiting.
        """
        fills = []
        half_spread = self.costs.half_spread_bps / 10_000
        for client_order_id, order in list(self.waiting.items()):
            intent = order.intent
            if intent.ticker != bar.ticker or order.sent_at > bar.start:
                continue
            limit = intent.limit_price
            buy = intent.side is Side.BUY
            marketable = bar.open <= limit if buy else bar.open >= limit
            through = bar.low < limit if buy else bar.high > limit
            if marketable:
                if buy:
                    price = min(bar.open * (1 + half_spread), limit, bar.high)
                else:
                    price = max(bar.open * (1 - half_spread), limit, bar.low)
                time = bar.start
            elif through:
                price, time = limit, bar.end
            else:
                continue
            left = intent.quantity - self.filled[client_order_id]
            quantity = min(left, math.floor(self.participation * bar.volume))
            if quantity == 0:
                continue
            first_fill = self.filled[client_order_id] == 0
            commission = self.costs.commission(quantity, price, first_fill)
            fills.append(
                Fill(
                    client_order_id,
                    intent.ticker,
                    intent.side,
                    quantity,
                    price,
                    commission,
                    time,
                )
            )
            self.filled[client_order_id] += quantity
            if self.filled[client_order_id] == intent.quantity:
                del self.waiting[client_order_id]
        return fills
