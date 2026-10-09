"""The Strategy interface, shared by the backtester and the live engine.

A strategy reacts to events and answers with order intents and cancellations.
It never touches the data feed, the broker or the clock: it sees the world
only through a View of what is known at that moment, which it cannot change.
The same strategy class therefore runs on history and, from Phase 3, live.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType

from qfin_alphaguard.events import Bar, Cancellation, Fill, Order, OrderIntent
from qfin_alphaguard.sessions import Session

# What a strategy may answer with after a slice of bars.
Decision = OrderIntent | Cancellation


@dataclass(frozen=True)
class View:
    """What a strategy may know at `now`. Read-only.

    `positions` and `last_prices` are read-only windows onto the engine's
    books, not copies: use a View during the call it is given in and do not
    keep it, because they show the latest state when read later.
    """

    now: datetime
    session: Session
    cash: float
    positions: Mapping[str, int]  # shares held; tickers with none are absent
    last_prices: Mapping[str, float]  # the latest close seen for each ticker
    open_orders: Sequence[Order]  # sent and not yet filled or cancelled

    def __post_init__(self) -> None:
        if self.now.tzinfo is None or self.now.utcoffset() != timedelta(0):
            raise ValueError(f"now must be a UTC datetime, got {self.now!r}")
        # Read-only even when built from plain dicts; wrapping does not copy.
        for name in ("positions", "last_prices"):
            if not isinstance(getattr(self, name), MappingProxyType):
                object.__setattr__(self, name, MappingProxyType(getattr(self, name)))
        object.__setattr__(self, "open_orders", tuple(self.open_orders))


class Strategy:
    """Base class for strategies. Override the callbacks you need.

    The engine calls them in this order on each trading day: on_session_start
    once, then for every slice of bars on_fill for each new fill and on_bars
    once, and on_session_end once. Times are UTC.
    """

    def on_session_start(self, view: View) -> None:
        """The session is about to open: load the day's plan here."""

    def on_bars(self, bars: Mapping[str, Bar], view: View) -> Sequence[Decision]:
        """Bars that ended at `view.now`, by ticker. Answer with decisions.

        Each OrderIntent and Cancellation must carry `view.now` as its time:
        a decision cannot be dated earlier than the data it was made from.
        """
        return ()

    def on_fill(self, fill: Fill, view: View) -> None:
        """An order (partly) filled; `view` already includes it."""

    def on_session_end(self, view: View) -> None:
        """The session closed and its open orders were cancelled."""
