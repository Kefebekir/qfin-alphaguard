"""Event types shared by the backtester and the live engine.

Every event is an immutable record that checks itself when it is created, so a
bad bar, score or order is rejected where it is made instead of turning into a
wrong trade later. The backtester and the live engine pass the same events, so
a strategy cannot tell which one it is running in.

All times are timezone-aware UTC datetimes. Convert to local time only for display.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from numbers import Integral
from types import MappingProxyType


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class RiskAction(Enum):
    REJECT = "reject"  # the order is dropped
    REDUCE = "reduce"  # the order goes out with a smaller quantity
    HALT = "halt"  # no new orders until a person restarts trading


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a UTC datetime, got {value!r}")


def _require_price(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive number, got {value!r}")


def _require_shares(name: str, value: int) -> None:
    # Integral accepts Python and NumPy integers but not 2.5, so a fractional
    # quantity has to be rounded on purpose before it reaches an order.
    if not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive whole number, got {value!r}")


def _require_side(value: Side) -> None:
    if not isinstance(value, Side):
        raise TypeError(f"side must be Side.BUY or Side.SELL, got {value!r}")


@dataclass(frozen=True)
class Bar:
    """One completed price bar for one stock."""

    ticker: str
    start: datetime
    seconds: int  # 60 for a one-minute bar; a regular US session is 23,400
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        _require_utc("start", self.start)
        if self.seconds <= 0:
            raise ValueError(f"seconds must be positive, got {self.seconds}")
        # Checked one by one because min() and max() can silently skip a NaN.
        for name in ("open", "high", "low", "close"):
            _require_price(name, getattr(self, name))
        if not (
            self.low <= min(self.open, self.close)
            and max(self.open, self.close) <= self.high
        ):
            raise ValueError(
                "prices must satisfy low <= open, close <= high; got "
                f"open={self.open} high={self.high} low={self.low} close={self.close}"
            )
        if self.volume < 0:
            raise ValueError(f"volume cannot be negative, got {self.volume}")

    @property
    def end(self) -> datetime:
        """The moment the bar is complete. Nothing may act on it before then."""
        return self.start + timedelta(seconds=self.seconds)


@dataclass(frozen=True)
class Signal:
    """The intraday model's score for one stock after one bar.

    Positive means the next minutes look good for buying, negative for selling.
    """

    ticker: str
    time: datetime
    score: float
    model_version: str

    def __post_init__(self) -> None:
        _require_utc("time", self.time)
        # A chained comparison is False for NaN, so a broken model output fails too.
        if not (-1 <= self.score <= 1):
            raise ValueError(f"score must be between -1 and 1, got {self.score!r}")


@dataclass(frozen=True)
class OrderIntent:
    """What the strategy wants to do, before Guard has checked it."""

    ticker: str
    side: Side
    quantity: int  # always positive; the direction lives in `side`
    limit_price: float
    time: datetime
    reason: str = ""  # short note for the log, e.g. "slice 1 of 2"

    def __post_init__(self) -> None:
        _require_side(self.side)
        _require_shares("quantity", self.quantity)
        _require_price("limit_price", self.limit_price)
        _require_utc("time", self.time)

    @property
    def notional(self) -> float:
        """Order value in dollars at the limit price."""
        return self.quantity * self.limit_price


@dataclass(frozen=True)
class Order:
    """An intent that Guard approved, as sent to the broker.

    If Guard reduced the order, `intent` holds the reduced quantity and the
    original request is recorded in the matching RiskEvent.
    """

    client_order_id: str
    intent: OrderIntent
    sent_at: datetime

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id cannot be empty")
        _require_utc("sent_at", self.sent_at)


@dataclass(frozen=True)
class Fill:
    """An execution reported by the broker. One order can fill in several parts."""

    client_order_id: str
    ticker: str
    side: Side
    quantity: int
    price: float
    commission: float
    time: datetime

    def __post_init__(self) -> None:
        _require_side(self.side)
        _require_shares("quantity", self.quantity)
        _require_price("price", self.price)
        if not (math.isfinite(self.commission) and self.commission >= 0):
            raise ValueError(f"commission cannot be negative, got {self.commission!r}")
        _require_utc("time", self.time)


@dataclass(frozen=True)
class Cancellation:
    """An open order taken off the market.

    A strategy asks for one to replace an unfilled order; the engine makes one
    for every open order at the close, and after Guard halts trading.
    """

    client_order_id: str
    time: datetime
    reason: str = ""  # short note for the log, e.g. "session close"

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id cannot be empty")
        _require_utc("time", self.time)


@dataclass(frozen=True)
class ExecutionSettings:
    """How the engine turns a day's target change into orders."""

    band_pct: float = 2.0  # no trade if the weight gap is smaller (percentage points)
    max_children: int = 3  # the day's change is split into at most this many orders
    min_trade_usd: float = 1000.0  # no order smaller than this
    theta0: float = 0.6  # score threshold at the start of the day, falls to -theta0
    min_gap_min: int = 30  # minutes between two orders in the same stock
    start_after_open_min: int = 5  # skip the noisy first minutes after the open
    deadline_before_close_min: int = 30  # send what is left this long before close

    def __post_init__(self) -> None:
        if self.band_pct < 0:
            raise ValueError(f"band_pct cannot be negative, got {self.band_pct}")
        if self.max_children < 1:
            raise ValueError(
                f"max_children must be at least 1, got {self.max_children}"
            )
        if self.min_trade_usd <= 0:
            raise ValueError(
                f"min_trade_usd must be positive, got {self.min_trade_usd}"
            )
        if not (0 < self.theta0 <= 1):
            raise ValueError(f"theta0 must be in (0, 1], got {self.theta0}")
        for name in (
            "min_gap_min",
            "start_after_open_min",
            "deadline_before_close_min",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative, got {getattr(self, name)}"
                )


@dataclass(frozen=True)
class DailyPlan:
    """The nightly job's output: target weights and execution settings for one day."""

    trading_date: date
    model_version: str
    target_weights: Mapping[str, float]
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    def __post_init__(self) -> None:
        # Copy first, so a later change to the caller's dict cannot reach the plan.
        weights = dict(self.target_weights)
        for ticker, weight in weights.items():
            if not (math.isfinite(weight) and weight >= 0):
                raise ValueError(f"weight for {ticker} must be >= 0, got {weight!r}")
        total = sum(weights.values())
        if total > 1 + 1e-9:  # small tolerance for floating-point rounding
            raise ValueError(f"weights sum to {total:.6f}; must be at most 1")
        # frozen=True blocks normal assignment, so the read-only view is stored
        # with object.__setattr__, once, while the object is being created.
        object.__setattr__(self, "target_weights", MappingProxyType(weights))


@dataclass(frozen=True)
class RiskEvent:
    """Guard rejected or reduced an order, or halted trading."""

    time: datetime
    rule: str  # the guard.yaml setting that triggered, e.g. "max_order_usd"
    action: RiskAction
    detail: str
    ticker: str | None = None  # None for account-wide events such as a halt

    def __post_init__(self) -> None:
        _require_utc("time", self.time)
        if not isinstance(self.action, RiskAction):
            raise TypeError(f"action must be a RiskAction, got {self.action!r}")
