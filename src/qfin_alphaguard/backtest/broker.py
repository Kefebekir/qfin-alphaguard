"""What the backtest needs from a simulated broker.

NextBarBroker below is the simulated broker, with its fill rules and costs
(Phase 1, step 6; Alpaca's costs and fractional shares since step 7e). The
engine checks every fill it reports against the contract in SimulatedBroker
and stops the backtest if one breaks it.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from qfin_alphaguard.events import Bar, Fill, Order, Side, round_shares


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
    """What a fill costs besides its price: Alpaca's (decision 0012).

    No commission. Two regulatory fees, which Alpaca adds up per day and
    rounds up to the cent (NextBarBroker.charge): the SEC fee on the value of
    each sale, and the CAT fee on every share bought or sold.
    """

    sec_fee_rate: float = 0.0000206  # SEC fee: this share of each sale's value
    cat_per_share: float = 0.000003  # CAT fee: USD per share bought or sold
    half_spread_bps: float = 1.0  # what a marketable order pays over the traded price

    def __post_init__(self) -> None:
        for name in ("sec_fee_rate", "cat_per_share", "half_spread_bps"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")

    def fees(self, side: Side, quantity: float, price: float) -> dict[str, float]:
        """Each fee one fill owes, in USD, before the day's rounding."""
        sec = self.sec_fee_rate * quantity * price if side is Side.SELL else 0.0
        return {"SEC": sec, "CAT": self.cat_per_share * quantity}


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
        self.filled: dict[str, float] = {}  # shares filled so far, by order id
        self.fee_day: date | None = None  # the day the two totals below are for
        self.owed: dict[str, float] = {}  # each fee's exact total that day
        self.charged: dict[str, float] = {}  # what was charged of each fee that day

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
        bar's volume, in shares to 9 decimals; nothing if that is 0. The rest
        waits.

        Commission: the fees the fill adds to the day's, rounded up to the
        cent (charge). An order that is completely filled stops waiting.
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
            quantity = round_shares(min(left, self.participation * bar.volume))
            if quantity == 0:
                continue
            commission = self.charge(intent.side, quantity, price, time)
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
            self.filled[client_order_id] = round_shares(
                self.filled[client_order_id] + quantity
            )
            if self.filled[client_order_id] == intent.quantity:
                del self.waiting[client_order_id]
        return fills

    def charge(
        self, side: Side, quantity: float, price: float, time: datetime
    ) -> float:
        """What a fill pays in fees: how far it takes each day total, in cents.

        Alpaca adds each fee up per day and charges the total rounded up to
        the cent. So the day's first fill pays at least a cent of each fee it
        owes, and a later fill only what takes a total past another cent.
        """
        if time.date() != self.fee_day:
            self.fee_day, self.owed, self.charged = time.date(), {}, {}
        charge = 0.0
        for name, amount in self.costs.fees(side, quantity, price).items():
            self.owed[name] = self.owed.get(name, 0.0) + amount
            # Rounded before going up, or floating point would make 30 cents,
            # 30.000000000000004, into 31.
            total = math.ceil(round(self.owed[name] * 100, 6)) / 100
            charge += total - self.charged.get(name, 0.0)
            self.charged[name] = total
        return charge
