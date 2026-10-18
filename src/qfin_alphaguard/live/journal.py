"""The live engine's journal: every input written to disk before the engine acts on it.

Decision 0021. The engine writes what it is told (the session's start, each bar, the
clock, each event about an order) and what it answers (decisions, Guard's events, orders,
fills), one JSON line at a time, and forces each to disk before acting: Sent is on disk
before the order goes out, so a crash can leave an order to ask about, never one the
engine does not know it sent. Each line keeps the moment the engine saw it. A restart
replays the day through the same rules (engine.replay_day, orders.apply), which rebuilds
the engine exactly as it was.

The file is plain text, one record a line, with a sequence number and the time it was
written. A crash in the middle of a write leaves a last line cut short: that write was not
finished, so nothing acted on it, and it is dropped when the journal is opened again. Any
other damage, such as a broken line before the last or a number out of sequence, raises
JournalError: the engine does not start until a person looks. So does a journal another
engine has open: two engines trading the same account would send every order twice.
"""

import fcntl
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Self, get_type_hints

from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DecisionNote,
    Fill,
    Order,
    OrderIntent,
    RiskEvent,
    Side,
    round_shares,
)
from qfin_alphaguard.live.engine import CancelAll, Flatten, SessionStart, Stop, Tick
from qfin_alphaguard.live.orders import (
    Accepted,
    Cancelled,
    CancelRefused,
    CancelRequested,
    Expired,
    FillReport,
    LiveOrder,
    OrderEvent,
    Rejected,
    Sent,
    apply,
)

# What a journal can hold, by the name written in each record.
RECORDS = {
    cls.__name__: cls
    for cls in (
        SessionStart,
        Bar,
        Tick,
        Stop,
        CancelAll,
        Flatten,
        DecisionNote,
        Cancellation,
        OrderIntent,
        RiskEvent,
        Order,
        Fill,
        Sent,
        Accepted,
        Rejected,
        FillReport,
        CancelRequested,
        Cancelled,
        CancelRefused,
        Expired,
    )
}


class JournalError(Exception):
    """A journal that cannot be read as written. The engine does not start."""


def _encode(value):
    if isinstance(value, date):  # a datetime too
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(cls: type, data: dict):
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise JournalError(f"{cls.__name__} has no field {sorted(unknown)}")
    return cls(**{name: _convert(hints[name], value) for name, value in data.items()})


def _convert(hint, value):
    if hint is datetime:
        return datetime.fromisoformat(value)
    if hint is date:
        return date.fromisoformat(value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(value)
    if hint is OrderIntent:
        return _decode(OrderIntent, value)
    return value


def encode(record) -> dict:
    """A record as the JSON object written for it, without its sequence number."""
    name = type(record).__name__
    if RECORDS.get(name) is not type(record):
        raise TypeError(f"a journal cannot hold a {name}")
    return {"type": name, **_encode(record)}


def decode(data: dict):
    """The record a JSON object holds."""
    data = dict(data)
    cls = RECORDS.get(data.pop("type", None))
    if cls is None:
        raise JournalError(f"a record of no known type: {data}")
    try:
        return _decode(cls, data)
    except (TypeError, ValueError, KeyError) as error:
        raise JournalError(f"a {cls.__name__} that cannot be read: {error}") from error


class Entry(NamedTuple):
    at: datetime  # when the engine saw it
    record: object


class Journal:
    """One session's journal, opened to read what it holds and to add to it.

    `entries` holds what was written before, oldest first, each with the moment the
    engine saw it; `dropped` is the line cut short by a crash, if there was one.
    """

    def __init__(self, path: Path, now=lambda: datetime.now(UTC)) -> None:
        self.path = path
        self._now = now
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        try:
            # Held until the file is closed or the process ends, however it ends.
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            raise JournalError(f"{path} is open in another engine") from None
        try:
            self.entries, self.dropped = _read(path)
        except BaseException:
            self._file.close()
            raise
        self._seq = len(self.entries)

    @property
    def records(self) -> list:
        return [entry.record for entry in self.entries]

    def write(self, record, at: datetime | None = None) -> None:
        """Add `record`, seen at `at` (now if not given), and force it to disk; only then
        may the engine act on it."""
        at = self._now() if at is None else at
        line = {"seq": self._seq + 1, "at": at.isoformat(), **encode(record)}
        self._file.write(json.dumps(line, separators=(",", ":")) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._seq += 1
        self.entries.append(Entry(at, record))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_journal(path: Path) -> list[Entry]:
    """A journal's entries, read without taking it: it may be in use by an engine.

    Nothing is changed in the file; a last line cut short is left out.
    """
    return _read(path, mend=False)[0]


def _read(path: Path, mend: bool = True) -> tuple[list[Entry], str | None]:
    """The entries in `path`, and the last line if a crash cut it short.

    With `mend`, a cut line is removed from the file, so the next record starts on a line
    of its own.
    """
    if not path.exists():
        return [], None
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    # A finished write ends with a newline, so a whole file splits into its lines and
    # one empty string. Anything after the last newline is a write cut short.
    *whole, tail = lines
    dropped = tail or None
    entries = []
    for number, line in enumerate(whole, start=1):
        try:
            data = json.loads(line)
        except ValueError as error:
            raise JournalError(
                f"{path}, line {number}, is broken: {line[:80]!r}"
            ) from error
        if not isinstance(data, dict) or data.pop("seq", None) != number:
            raise JournalError(f"{path}, line {number}, is out of sequence")
        try:
            at = datetime.fromisoformat(data.pop("at"))
        except (KeyError, TypeError, ValueError) as error:
            raise JournalError(f"{path}, line {number}, has no time") from error
        entries.append(Entry(at, decode(data)))
    if dropped is not None and mend:
        with path.open("r+", encoding="utf-8") as file:
            file.truncate(len(text.encode("utf-8")) - len(tail.encode("utf-8")))
    return entries, dropped


@dataclass(frozen=True)
class Rebuilt:
    """The engine's state from a journal: its orders, and the bars it saw, in order."""

    orders: dict[str, LiveOrder]
    bars: list[Bar]


def rebuild(records: Iterable) -> Rebuilt:
    """Replay a journal's records through the order rules, as the engine took them.

    An event the rules refused when it arrived is refused again: orders.InvalidTransition,
    and the engine does not start.
    """
    orders: dict[str, LiveOrder] = {}
    bars: list[Bar] = []
    for record in records:
        if isinstance(record, Bar):
            bars.append(record)
        elif isinstance(record, Order):
            if record.client_order_id in orders:
                raise JournalError(f"{record.client_order_id} was made twice")
            orders[record.client_order_id] = LiveOrder(record)
        elif isinstance(record, OrderEvent):
            order = orders.get(record.client_order_id)
            if order is None:
                raise JournalError(
                    f"{type(record).__name__} for {record.client_order_id}, "
                    "an order the engine did not make"
                )
            orders[record.client_order_id] = apply(order, record)
        elif RECORDS.get(type(record).__name__) is not type(record):
            raise JournalError(f"a journal cannot hold a {type(record).__name__}")
        # The rest (decisions, Guard's events, fills, the clock) changes no order.
    return Rebuilt(orders, bars)


def filled_positions(orders: Iterable[LiveOrder]) -> dict[str, float]:
    """The shares the orders' fills add up to, by ticker: positive long, negative short."""
    held: dict[str, float] = {}
    for order in orders:
        intent = order.order.intent
        sign = 1 if intent.side is Side.BUY else -1
        held[intent.ticker] = round_shares(
            held.get(intent.ticker, 0.0) + sign * order.filled_quantity
        )
    return {ticker: shares for ticker, shares in held.items() if shares}
