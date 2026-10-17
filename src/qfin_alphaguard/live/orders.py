"""The live engine's orders, and the status each one is in (decision 0021).

The simulated broker fills an order at the next bar, all at once. A real broker answers
later, in parts, and sometimes twice: an order is sent, accepted, filled piece by piece,
and a fill can arrive while a cancel is on its way. Each order the live engine sends is a
LiveOrder, and every report about it, from the engine or from Alpaca, is an event that
`apply` turns into the order's next status.

`transition`, the rules themselves, is Efe's. The rest makes a wrong order impossible to
hold: a LiveOrder checks itself, so a rule that leaves a filled order with shares still to
fill fails where it is made. An order is rebuilt after a restart by replaying its events
from the event log.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from qfin_alphaguard.events import (
    Order,
    _require_price,
    _require_shares,
    _require_utc,
    round_shares,
)


class Status(Enum):
    NEW = "new"  # made by the engine, not sent yet
    SUBMITTED = "submitted"  # sent; the broker has not said it is working
    ACCEPTED = "accepted"  # working at the exchange, nothing filled
    PARTLY_FILLED = "partly_filled"  # working, some of it filled
    PENDING_CANCEL = "pending_cancel"  # cancel asked, not confirmed: still open
    FILLED = "filled"  # every share filled
    CANCELLED = "cancelled"  # taken off the market; the shares filled before stay
    REJECTED = "rejected"  # refused by the broker, nothing filled
    EXPIRED = "expired"  # a day order the broker ended at the close


# Orders that can still fill: Guard counts them as open (decision 0019).
OPEN = frozenset(
    {Status.SUBMITTED, Status.ACCEPTED, Status.PARTLY_FILLED, Status.PENDING_CANCEL}
)
FINISHED = frozenset({Status.FILLED, Status.CANCELLED, Status.REJECTED, Status.EXPIRED})
NOTHING_FILLED = frozenset(
    {Status.NEW, Status.SUBMITTED, Status.ACCEPTED, Status.REJECTED}
)
SHARES_LEFT = frozenset({Status.PENDING_CANCEL, Status.CANCELLED, Status.EXPIRED})


class InvalidTransition(Exception):
    """A report the order cannot take in its status. The engine stops and reconciles."""


@dataclass(frozen=True)
class OrderEvent:
    """Something that happened to one order, at `time`."""

    client_order_id: str
    time: datetime

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id cannot be empty")
        _require_utc("time", self.time)


@dataclass(frozen=True)
class Sent(OrderEvent):
    """The engine sent the order to the broker."""


@dataclass(frozen=True)
class Accepted(OrderEvent):
    """The broker says the order is working at the exchange."""


@dataclass(frozen=True)
class Rejected(OrderEvent):
    """The broker refused the order."""

    reason: str = ""


@dataclass(frozen=True)
class FillReport(OrderEvent):
    """The order's fills so far: every share filled up to now, and their average price.

    Alpaca reports each fill with the order's totals, and so does its answer when the
    engine asks for the order. Totals make a report safe to receive twice, and one missed
    while the connection was down is covered by the next.
    """

    filled_quantity: float
    average_price: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_shares("filled_quantity", self.filled_quantity)
        _require_price("average_price", self.average_price)


@dataclass(frozen=True)
class CancelRequested(OrderEvent):
    """The engine asked the broker to cancel the order."""


@dataclass(frozen=True)
class Cancelled(OrderEvent):
    """The broker took the order off the market."""


@dataclass(frozen=True)
class CancelRefused(OrderEvent):
    """The broker refused a cancel, most often because the order had just filled."""

    reason: str = ""


@dataclass(frozen=True)
class Expired(OrderEvent):
    """The broker ended a day order at the close."""


@dataclass(frozen=True)
class LiveOrder:
    """An order Guard approved, its status at the broker, and what has filled."""

    order: Order
    status: Status = Status.NEW
    filled_quantity: float = 0.0
    average_price: float | None = None  # of the shares filled; None before any

    def __post_init__(self) -> None:
        if not isinstance(self.status, Status):
            raise TypeError(f"status must be a Status, got {self.status!r}")
        filled, quantity = self.filled_quantity, self.quantity
        if filled != 0:
            _require_shares("filled_quantity", filled)
            if filled > quantity:
                raise ValueError(f"{filled} shares filled of an order for {quantity}")
        if (filled == 0) != (self.average_price is None):
            raise ValueError("average_price is given if and only if shares filled")
        if self.average_price is not None:
            _require_price("average_price", self.average_price)
        # What each status says about the fills.
        status = self.status
        if status is Status.FILLED and filled != quantity:
            raise ValueError("a filled order has every share filled")
        if status is Status.PARTLY_FILLED and not 0 < filled < quantity:
            raise ValueError("a partly filled order has some shares filled, not all")
        if status in NOTHING_FILLED and filled != 0:
            raise ValueError(f"a {status.value} order has nothing filled")
        if status in SHARES_LEFT and filled == quantity:
            raise ValueError(f"a {status.value} order has shares left unfilled")

    @property
    def client_order_id(self) -> str:
        return self.order.client_order_id

    @property
    def quantity(self) -> float:
        return self.order.intent.quantity

    @property
    def remaining(self) -> float:
        """Shares not filled; on an open order, shares that may still fill."""
        return round_shares(self.quantity - self.filled_quantity)

    @property
    def is_open(self) -> bool:
        return self.status in OPEN


def apply(order: LiveOrder, event: OrderEvent) -> LiveOrder:
    """The order after `event`. InvalidTransition if the order cannot take it."""
    if event.client_order_id != order.client_order_id:
        raise InvalidTransition(
            f"{type(event).__name__} for {event.client_order_id} "
            f"applied to {order.client_order_id}"
        )
    return transition(order, event)


def replay(order: Order, events: Iterable[OrderEvent]) -> LiveOrder:
    """The order rebuilt from its events, oldest first, as after a restart."""
    live = LiveOrder(order)
    for event in events:
        live = apply(live, event)
    return live


# ---------------------------------------------------------------------------------------
# Efe's rules
# ---------------------------------------------------------------------------------------


def transition(order: LiveOrder, event: OrderEvent) -> LiveOrder:
    """The order after `event`, or InvalidTransition if it cannot happen.

    A report the order already knows (a fill total it has, a second cancel request, the
    end it already reached) leaves it as it is.
    """
    status = order.status
    match event:
        # Sent once: an order whose outcome is unknown is never sent again.
        case Sent() if status is Status.NEW:
            return replace(order, status=Status.SUBMITTED)
        # Accepted is news only to a submitted order; every later status implies it.
        case Accepted() if status is Status.SUBMITTED:
            return replace(order, status=Status.ACCEPTED)
        case Accepted() if status not in (Status.NEW, Status.REJECTED):
            return order
        # Rejected only before anything fills.
        case Rejected() if status in (Status.SUBMITTED, Status.ACCEPTED):
            return replace(order, status=Status.REJECTED)
        case Rejected() if status is Status.REJECTED:
            return order
        # Fills by their totals: a total already known changes nothing.
        case FillReport() if event.filled_quantity <= order.filled_quantity:
            return order
        case FillReport() if order.is_open and event.filled_quantity <= order.quantity:
            if event.filled_quantity == order.quantity:
                new_status = Status.FILLED
            elif status is Status.PENDING_CANCEL:
                new_status = Status.PENDING_CANCEL  # still open until confirmed
            else:
                new_status = Status.PARTLY_FILLED
            return replace(
                order,
                status=new_status,
                filled_quantity=event.filled_quantity,
                average_price=event.average_price,
            )
        # A cancel only for an order the broker is working.
        case CancelRequested() if status in (Status.ACCEPTED, Status.PARTLY_FILLED):
            return replace(order, status=Status.PENDING_CANCEL)
        case CancelRequested() if status is Status.PENDING_CANCEL:
            return order
        # The broker's ends are final.
        case Cancelled() if order.is_open:
            return replace(order, status=Status.CANCELLED)
        case Cancelled() if status is Status.CANCELLED:
            return order
        case Expired() if order.is_open:
            return replace(order, status=Status.EXPIRED)
        case Expired() if status is Status.EXPIRED:
            return order
        # A refused cancel: back to working, unless the order has ended meanwhile.
        case CancelRefused() if status is Status.PENDING_CANCEL:
            if order.filled_quantity > 0:
                return replace(order, status=Status.PARTLY_FILLED)
            return replace(order, status=Status.ACCEPTED)
        case CancelRefused() if status in FINISHED:
            return order
    raise InvalidTransition(
        f"{type(event).__name__} on {order.client_order_id}, a {status.value} order"
    )
