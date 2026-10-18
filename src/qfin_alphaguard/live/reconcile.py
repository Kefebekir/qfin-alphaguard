"""Reconciliation: the engine's orders and positions against Alpaca's (decision 0021).

The engine reconciles when it starts, after a lost connection, and after a send or a
cancel whose answer was lost. Three checks:
1. each order the engine holds open is asked for by its client order id, and what Alpaca
   says is taken through the order rules: fills and ends missed while the connection was
   down are caught up. An order the engine sent that Alpaca never got was not placed;
2. each order Alpaca holds open must be one the engine made, and open there too;
3. the positions at Alpaca must equal what the engine held at the session's start and
   what its fills since add up to. The QQQ strategy holds nothing overnight, so the start
   is flat unless a close failed.

Alpaca's answers can lag its trade updates by a moment: on 17 October 2026 an order the
stream had already reported cancelled was still listed as open. So a difference is asked
about again, ATTEMPTS times PAUSE seconds apart, and only one that lasts is reported. A
missing report is never read as flat: whatever cannot be matched is a difference, and the
engine does not trade until a person looks.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from qfin_alphaguard.events import round_shares
from qfin_alphaguard.live.alpaca_broker import AlpacaBroker
from qfin_alphaguard.live.journal import filled_positions
from qfin_alphaguard.live.orders import (
    LiveOrder,
    OrderEvent,
    Rejected,
    Status,
    apply,
)

ATTEMPTS = 3
PAUSE = 2.0  # seconds; the lag seen was under one


@dataclass(frozen=True)
class Reconciliation:
    """What Alpaca's answers changed, and what still differs.

    `events` are the ones that changed an order, oldest first, for the journal; `orders`
    are the engine's orders after them. The engine may trade only if `differences` is
    empty.
    """

    events: tuple[OrderEvent, ...]
    orders: dict[str, LiveOrder]
    differences: tuple[str, ...]

    @property
    def agrees(self) -> bool:
        return not self.differences


def reconcile(
    orders: Mapping[str, LiveOrder],
    broker: AlpacaBroker,
    start_positions: Mapping[str, float] | None = None,
    attempts: int = ATTEMPTS,
    pause: float = PAUSE,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Reconciliation:
    """Compare `orders`, the engine's, and `start_positions`, what it held when the session
    started, with Alpaca's orders and positions.

    Raises orders.InvalidTransition if Alpaca reports what the order rules refuse, and
    alpaca_broker.BrokerError if an answer cannot be read: the engine stops either way.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    current = dict(orders)
    events: list[OrderEvent] = []

    def take(event: OrderEvent) -> None:
        before = current[event.client_order_id]
        after = apply(before, event)
        if after != before:
            current[event.client_order_id] = after
            events.append(event)

    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        differences = []
        # 1. The engine's open orders, as Alpaca has them.
        for cid, order in list(current.items()):
            if not order.is_open:
                continue
            placed = broker.order(cid)
            if placed is None:
                if order.status is Status.SUBMITTED and last:
                    # The send was lost on its way: nothing was placed.
                    take(Rejected(cid, now(), "never reached Alpaca"))
                else:
                    differences.append(
                        f"{cid} is {order.status.value} here and unknown to Alpaca"
                    )
                continue
            if not placed.matches(order.order):
                differences.append(
                    f"{cid} at Alpaca is {placed.side.value} {placed.quantity} "
                    f"{placed.symbol}, not the order the engine sent"
                )
                continue
            for event in placed.events():
                take(event)
        # 2. Alpaca's open orders, as the engine has them.
        for placed in broker.open_orders():
            cid = placed.client_order_id
            order = current.get(cid)
            if order is None:
                differences.append(
                    f"{cid} is open at Alpaca, an order the engine did not make"
                )
            elif not order.is_open:
                differences.append(
                    f"{cid} is open at Alpaca and {order.status.value} here"
                )
        # 3. The positions.
        mine = dict(start_positions or {})
        for ticker, shares in filled_positions(current.values()).items():
            mine[ticker] = round_shares(mine.get(ticker, 0.0) + shares)
        mine = {ticker: shares for ticker, shares in mine.items() if shares}
        theirs = broker.positions()
        for ticker in sorted(set(mine) | set(theirs)):
            if mine.get(ticker, 0.0) != theirs.get(ticker, 0.0):
                differences.append(
                    f"{ticker}: {mine.get(ticker, 0.0)} shares here, "
                    f"{theirs.get(ticker, 0.0)} at Alpaca"
                )
        if not differences or last:
            return Reconciliation(tuple(events), current, tuple(differences))
        sleep(pause)
    raise AssertionError("unreachable: the last attempt returns")
