"""Alpaca's paper trading account: orders out; their status, positions and the account back.

Decision 0021. The engine knows an order by its client order id; Alpaca's own id stays in
here. This module knows only the paper account's address: trading real money needs a
reviewed change of code here, as well as `live_trading` in guard_qqq.yaml.

What each answer means:
- An order Alpaca reports, in an answer or a trade update, becomes the events its status
  implies (AlpacaOrder.events), for orders.apply to take.
- A send Alpaca refuses (4xx) placed nothing: Rejected.
- A send or a cancel whose answer is lost (a network error or a 5xx) is never repeated. It
  raises OutcomeUnknown, and the engine asks for the order by its client order id.
- Reads (an order, the open orders, the positions, the account) are retried, as the market
  data client does.
- An answer that makes no sense, such as an order that is not the one sent or a status the
  engine does not use, raises BrokerError: the engine stops and reconciles.

The keys travel in request headers, never in the URL, and no error carries them.
"""

import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from qfin_alphaguard.events import Order, Side, round_shares
from qfin_alphaguard.live.orders import (
    Accepted,
    Cancelled,
    CancelRefused,
    Expired,
    FillReport,
    OrderEvent,
    Rejected,
)

PAPER_URL = "https://paper-api.alpaca.markets"
ATTEMPTS = (
    4  # for a read, after a rate limit (429), a server error (5xx) or a network error
)
TIMEOUT = 10  # seconds; an order's answer comes in well under one
MAX_OPEN_ORDERS = (
    500  # the most Alpaca lists in one answer; the engine has a few at most
)

# Alpaca's order statuses, by what they mean for the engine. Any other (replaced,
# stopped, suspended...) comes from what the engine never does, or needs a person to look.
WORKING = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "partially_filled",
        "pending_cancel",
    }
)
ENDS = {
    "filled": None,  # the fills say it all
    "canceled": Cancelled,
    "expired": Expired,
    "done_for_day": Expired,
    "rejected": Rejected,
}
# The trade updates the engine expects; the same rule holds for the rest.
TRADE_UPDATES = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "partial_fill",
        "fill",
        "pending_cancel",
        "canceled",
        "expired",
        "done_for_day",
        "rejected",
        "order_cancel_rejected",
    }
)


class BrokerError(Exception):
    """An answer that makes no sense to the engine. It stops and reconciles."""


class OutcomeUnknown(Exception):
    """A send or a cancel whose answer was lost: ask for the order before anything else."""


def _time(text: str) -> datetime:
    # Alpaca writes nanoseconds and, in places, New York's offset; Python keeps
    # microseconds.
    return datetime.fromisoformat(text).astimezone(UTC)


def _shares(text: str | None, name: str) -> float:
    if text is None:
        raise BrokerError(f"{name} is missing")
    value = round_shares(float(text))
    if not math.isfinite(value):
        raise BrokerError(f"{name} is {text!r}")
    return value


@dataclass(frozen=True)
class AlpacaOrder:
    """One order as Alpaca reports it."""

    broker_id: str
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    status: str  # Alpaca's word for it
    filled_quantity: float
    average_price: float | None
    updated_at: datetime

    @classmethod
    def parse(cls, answer: dict) -> "AlpacaOrder":
        try:
            if answer.get("type") != "limit" or answer.get("order_class") not in (
                "",
                "simple",
                None,
            ):
                raise BrokerError(
                    f"{answer.get('client_order_id')} is a {answer.get('order_class')} "
                    f"{answer.get('type')} order; the engine sends only simple limits"
                )
            average = answer.get("filled_avg_price")
            return cls(
                broker_id=answer["id"],
                client_order_id=answer["client_order_id"],
                symbol=answer["symbol"],
                side=Side(answer["side"]),
                quantity=_shares(answer.get("qty"), "qty"),
                status=answer["status"],
                filled_quantity=_shares(answer.get("filled_qty"), "filled_qty"),
                average_price=None if average is None else float(average),
                updated_at=_time(answer["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BrokerError(
                f"an order Alpaca reported cannot be read: {error!r}"
            ) from error

    def events(self, time: datetime | None = None) -> list[OrderEvent]:
        """What the status says has happened to the order, oldest first.

        Working orders are Accepted; any fills are one FillReport of the totals; an
        ended order ends. Events the order already took change nothing in orders.apply,
        so the same status can be reported again safely.
        """
        at = time or self.updated_at
        out: list[OrderEvent] = []
        if self.status in WORKING:
            out.append(Accepted(self.client_order_id, at))
        elif self.status not in ENDS:
            raise BrokerError(
                f"{self.client_order_id} is {self.status!r}, a status the engine "
                "does not use"
            )
        if self.filled_quantity > 0:
            if self.average_price is None:
                raise BrokerError(f"{self.client_order_id} has fills but no price")
            out.append(
                FillReport(
                    self.client_order_id, at, self.filled_quantity, self.average_price
                )
            )
        if self.status == "filled" and self.filled_quantity != self.quantity:
            raise BrokerError(
                f"{self.client_order_id} is filled with {self.filled_quantity} of "
                f"{self.quantity} shares"
            )
        end = ENDS.get(self.status)
        if end is not None:
            out.append(end(self.client_order_id, at))
        return out

    def matches(self, order: Order) -> bool:
        """Whether this is `order`, as the engine sent it."""
        intent = order.intent
        return (
            self.client_order_id == order.client_order_id
            and self.symbol == intent.ticker
            and self.side is intent.side
            and self.quantity == intent.quantity
        )


def trade_update_events(update: dict) -> list[OrderEvent]:
    """The events in one message of Alpaca's trade_updates stream (its "data")."""
    kind = update.get("event")
    if kind not in TRADE_UPDATES:
        raise BrokerError(f"a trade update the engine does not expect: {kind!r}")
    if not isinstance(update.get("order"), dict):
        raise BrokerError(f"a {kind} trade update without its order")
    order = AlpacaOrder.parse(update["order"])
    stamp = update.get("timestamp")
    at = _time(stamp) if stamp else order.updated_at
    events = order.events(at)
    if kind == "order_cancel_rejected":
        events.append(CancelRefused(order.client_order_id, at, "cancel rejected"))
    return events


@dataclass(frozen=True)
class Account:
    """The account's money and whether it may trade."""

    status: str
    currency: str
    cash: float
    equity: float
    last_equity: float  # at the last close
    buying_power: float
    blocked: bool  # trading or the account blocked, or trading suspended by its owner
    shorting_enabled: bool

    @property
    def can_trade(self) -> bool:
        return self.status == "ACTIVE" and not self.blocked


class AlpacaBroker:
    name = "Alpaca paper"

    def __init__(
        self,
        key_id: str,
        secret: str,
        urlopen: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self._urlopen = urlopen
        self._sleep = sleep
        self._now = now
        self._broker_ids: dict[str, str] = {}  # client order id -> Alpaca's id

    # Requests.

    def _call(self, method: str, path: str, body: dict | None = None):
        """Alpaca's status code and its JSON answer, None when empty. A network error
        raises; an HTTP error is an answer like any other."""
        data = None if body is None else json.dumps(body).encode()
        headers = dict(self._headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            PAPER_URL + path, data=data, headers=headers, method=method
        )
        try:
            with self._urlopen(request, timeout=TIMEOUT) as response:
                code, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, raw = error.code, error.read(500)
        if not raw:
            return code, None
        try:
            return code, json.loads(raw)
        except ValueError:
            return code, raw.decode(errors="replace")

    def _read(self, path: str):
        """A GET, retried after a rate limit, a server error or a network error."""
        for attempt in range(1, ATTEMPTS + 1):
            try:
                code, answer = self._call("GET", path)
            except (OSError, http.client.HTTPException) as error:
                problem = str(getattr(error, "reason", error)) or type(error).__name__
            else:
                if code != 429 and code < 500:
                    return code, answer
                problem = f"HTTP {code} {_message(answer)}"
            if attempt < ATTEMPTS:
                self._sleep(2**attempt)
        # Raised outside the except blocks, so nothing chained to it carries the keys.
        raise BrokerError(f"Alpaca read {path.split('?')[0]} failed: {problem}")

    # Orders.

    def submit(self, order: Order) -> list[OrderEvent]:
        """Send `order` as a day limit order; the events Alpaca's answer implies.

        The engine records Sent before calling this, so a crash in between leaves an
        order to ask for, never one to send again.
        """
        intent = order.intent
        if len(order.client_order_id) > 128:
            raise ValueError("Alpaca takes client order ids of at most 128 characters")
        body = {
            "symbol": intent.ticker,
            "qty": _quantity(intent.quantity),
            "side": intent.side.value,
            "type": "limit",
            "time_in_force": "day",
            "limit_price": _price(intent.limit_price),
            "client_order_id": order.client_order_id,
        }
        try:
            code, answer = self._call("POST", "/v2/orders", body)
        except (OSError, http.client.HTTPException) as error:
            problem = str(getattr(error, "reason", error)) or type(error).__name__
        else:
            if code == 200 and isinstance(answer, dict):
                placed = AlpacaOrder.parse(answer)
                if not placed.matches(order):
                    raise BrokerError(
                        f"Alpaca answered the send of {order.client_order_id} with "
                        f"{placed.client_order_id}: {placed.side.value} "
                        f"{placed.quantity} {placed.symbol}"
                    )
                self._broker_ids[order.client_order_id] = placed.broker_id
                return placed.events()
            message = _message(answer)
            if 400 <= code < 500:
                if "client_order_id" in message:
                    # An order with this id exists: the engine sent it twice.
                    raise BrokerError(f"{order.client_order_id}: {message}")
                return [Rejected(order.client_order_id, self._now(), message)]
            problem = f"HTTP {code} {message}"
        raise OutcomeUnknown(f"send of {order.client_order_id}: {problem}")

    def order(self, client_order_id: str) -> AlpacaOrder | None:
        """The order with this client order id, or None if Alpaca has none."""
        query = urllib.parse.urlencode({"client_order_id": client_order_id})
        code, answer = self._read(f"/v2/orders:by_client_order_id?{query}")
        if code == 404:
            return None
        placed = AlpacaOrder.parse(_expect(code, answer, dict))
        if placed.client_order_id != client_order_id:
            raise BrokerError(
                f"asked for {client_order_id}, got {placed.client_order_id}"
            )
        self._broker_ids[client_order_id] = placed.broker_id
        return placed

    def cancel(self, client_order_id: str) -> list[OrderEvent]:
        """Ask Alpaca to cancel the order. Nothing yet if it will: the trade updates
        confirm it. CancelRefused if it will not, most often because it has filled.

        The engine records CancelRequested before calling this.
        """
        broker_id = self._broker_ids.get(client_order_id)
        if broker_id is None:
            placed = self.order(client_order_id)
            if placed is None:
                raise BrokerError(
                    f"cancel of {client_order_id}: Alpaca has no such order"
                )
            broker_id = placed.broker_id
        try:
            code, answer = self._call("DELETE", f"/v2/orders/{broker_id}")
        except (OSError, http.client.HTTPException) as error:
            problem = str(getattr(error, "reason", error)) or type(error).__name__
        else:
            if code in (200, 204):
                return []
            message = _message(answer)
            if code == 404:
                raise BrokerError(f"cancel of {client_order_id}: {message}")
            if 400 <= code < 500:
                return [CancelRefused(client_order_id, self._now(), message)]
            problem = f"HTTP {code} {message}"
        raise OutcomeUnknown(f"cancel of {client_order_id}: {problem}")

    def open_orders(self) -> list[AlpacaOrder]:
        """Every order Alpaca is still working, oldest first."""
        query = urllib.parse.urlencode(
            {"status": "open", "limit": MAX_OPEN_ORDERS, "direction": "asc"}
        )
        code, answer = self._read(f"/v2/orders?{query}")
        orders = [AlpacaOrder.parse(o) for o in _expect(code, answer, list)]
        if len(orders) >= MAX_OPEN_ORDERS:
            raise BrokerError(f"{len(orders)} open orders; the engine expects a few")
        for o in orders:
            self._broker_ids[o.client_order_id] = o.broker_id
        return orders

    # The account.

    def positions(self) -> dict[str, float]:
        """Shares held by symbol: positive long, negative short."""
        code, answer = self._read("/v2/positions")
        held: dict[str, float] = {}
        for p in _expect(code, answer, list):
            try:
                symbol, side = p["symbol"], p["side"]
                if side not in ("long", "short") or symbol in held:
                    raise ValueError(f"{symbol}: side {side!r} or listed twice")
                quantity = abs(_shares(p["qty"], "qty"))
            except (KeyError, TypeError, ValueError) as error:
                raise BrokerError(
                    f"a position Alpaca reported cannot be read: {error!r}"
                ) from error
            held[symbol] = quantity if side == "long" else -quantity
        return held

    def account(self) -> Account:
        code, answer = self._read("/v2/account")
        a = _expect(code, answer, dict)
        try:
            return Account(
                status=a["status"],
                currency=a["currency"],
                cash=float(a["cash"]),
                equity=float(a["equity"]),
                last_equity=float(a["last_equity"]),
                buying_power=float(a["buying_power"]),
                blocked=bool(
                    a["trading_blocked"]
                    or a["account_blocked"]
                    or a.get("trade_suspended_by_user")
                ),
                shorting_enabled=bool(a["shorting_enabled"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BrokerError(
                f"the account Alpaca reported cannot be read: {error!r}"
            ) from error


def _message(answer) -> str:
    """Alpaca's reason, from an answer such as {"code": 40310000, "message": "..."}."""
    if isinstance(answer, dict):
        return str(answer.get("message", answer))[:200]
    return str(answer or "")[:200]


def _expect(code: int, answer, kind: type):
    if code != 200 or not isinstance(answer, kind):
        raise BrokerError(f"Alpaca answered HTTP {code} {_message(answer)}")
    return answer


def _quantity(shares: float) -> str:
    """Shares as Alpaca takes them: up to 9 decimals, no trailing zeros."""
    return f"{shares:.9f}".rstrip("0").rstrip(".")


def _price(price: float) -> str:
    """A limit price in the ticks US exchanges take: whole cents from 1 USD up, and
    hundredths of a cent below. A price between ticks is refused, not rounded: the
    strategy decides its limits."""
    decimals = 2 if price >= 1 else 4
    ticks = round(price * 10**decimals)
    if abs(ticks - price * 10**decimals) > 1e-6:
        raise ValueError(f"a limit price of {price} is not a whole tick")
    return f"{ticks / 10**decimals:.{decimals}f}"
