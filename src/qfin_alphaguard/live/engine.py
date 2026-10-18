"""The live engine's core: the backtest's loop, for a broker that answers late.

Decision 0021. The core makes every decision and does no input or output. It is told what
happened, with the moment the engine saw it: a session's start, bars, the clock, an order
event from Alpaca, a stop. It answers with a Step: the records to write to the journal,
and then the actions to take (orders to send, cancels to ask for). The runner writes the
records first, then acts. The same inputs always give the same answers, so a day replayed
from its journal (replay_day) remakes the day's decisions: the check of Phase 3's gate,
and how a restarted engine finds its place.

What differs from the backtest's engine:
- `now` is when the engine saw an input, not the end of a bar, so Guard's stale data rule
  measures how late the feed is.
- Orders answer late. An order is open until Alpaca says it ended, and a cancel is open
  until Alpaca confirms it (orders.py, Efe's rules). An order Alpaca has not accepted yet
  cannot be cancelled: its cancel goes the moment Alpaca accepts it. Fills come as totals;
  each new total becomes the Fill the books and the strategy expect.
- One order at a time for a ticker: an order decided while another is open for the same
  ticker waits until that one ends. The QQQ strategy cancels before it decides again.
- A replacement waits for its cancel (Efe, 17 October 2026: like the backtest). When the
  strategy cancels an order and decides again, the engine sends nothing for that ticker
  until Alpaca confirms the old order ended; then it sends what still takes the position
  to where the strategy wanted it, at the latest price. The old order may have filled
  meanwhile; in the backtest a cancel is instant, so this is what the backtest would hold.
- The clock: a Tick lets the strategy act without a bar. The runner ticks only at the
  day's last minute if its bar is late, so the close is never skipped.
- Stopping (Efe: close, then stop): stop() cancels every open order and makes no more
  decisions; flatten() then closes what Alpaca says is held, with Guard judging the
  orders against Alpaca's positions, not the engine's.
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from types import MappingProxyType

from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DecisionNote,
    Fill,
    Order,
    OrderIntent,
    RiskAction,
    Side,
    _require_utc,
    round_shares,
)
from qfin_alphaguard.guard.check import GuardDecision, RiskCheck
from qfin_alphaguard.live.orders import (
    CancelRequested,
    LiveOrder,
    OrderEvent,
    Sent,
    Status,
    apply,
)
from qfin_alphaguard.portfolio import Portfolio
from qfin_alphaguard.sessions import Session, sessions
from qfin_alphaguard.strategy import Strategy, View


class EngineError(RuntimeError):
    """An input or a part broke the engine's contract. The engine stops."""


# Inputs the runner writes to the journal, besides bars and Alpaca's order events.


@dataclass(frozen=True)
class SessionStart:
    """The account at the session's start, as Alpaca gave it and reconciliation agreed.

    `positions` are shares held from an earlier day, and `prices` their last prices.
    """

    day: date
    time: datetime
    cash: float
    last_close_value: float  # Alpaca's equity at the last close
    positions: Mapping[str, float] = field(default_factory=dict)
    prices: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_utc("time", self.time)
        if not math.isfinite(self.cash):
            raise ValueError(f"cash must be a finite number, got {self.cash!r}")
        if not (math.isfinite(self.last_close_value) and self.last_close_value > 0):
            raise ValueError(
                f"last_close_value must be positive, got {self.last_close_value!r}"
            )
        missing = sorted(set(self.positions) - set(self.prices))
        if missing:
            raise ValueError(f"no price for held {', '.join(missing)}")
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))


@dataclass(frozen=True)
class Tick:
    """The clock, without a bar: the strategy may act on what it has seen."""

    time: datetime

    def __post_init__(self) -> None:
        _require_utc("time", self.time)


@dataclass(frozen=True)
class Stop:
    """The engine stops: it cancels its open orders and decides nothing more."""

    time: datetime
    reason: str

    def __post_init__(self) -> None:
        _require_utc("time", self.time)
        if not self.reason:
            raise ValueError("a stop needs a reason")


@dataclass(frozen=True)
class CancelAll:
    """After a stop: cancel every order still open, such as a close that did not fill."""

    time: datetime
    reason: str

    def __post_init__(self) -> None:
        _require_utc("time", self.time)


@dataclass(frozen=True)
class Flatten:
    """After a stop: close these positions, Alpaca's, shares by ticker."""

    time: datetime
    positions: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_utc("time", self.time)
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))


# What the core answers.


@dataclass(frozen=True)
class Submit:
    order: Order


@dataclass(frozen=True)
class Cancel:
    client_order_id: str


Action = Submit | Cancel


@dataclass(frozen=True)
class Step:
    """Records for the journal, then the actions they lead to, in order."""

    records: tuple = ()
    actions: tuple[Action, ...] = ()


@dataclass(frozen=True)
class _Waiting:
    """A position the strategy wanted, held back until its ticker's cancels end."""

    ticker: str
    target: float  # shares: positive long, negative short
    reason: str


def _limit(price: float, side: Side, collar_bps: float) -> float:
    """The strategy's limit: the last price plus the collar to buy, minus it to sell,
    on the cent and inside the collar (momentum.py)."""
    collar = collar_bps / 10_000
    if side is Side.BUY:
        return math.floor(round(price * (1 + collar) * 100, 6)) / 100
    return math.ceil(round(price * (1 - collar) * 100, 6)) / 100


class LiveEngine:
    """One session of the live engine."""

    def __init__(
        self,
        strategy: Strategy,
        guard: RiskCheck,
        start: SessionStart,
        collar_bps: float,
        prefix: str = "qqq",
    ) -> None:
        found = sessions(start.day, start.day)
        if not found:
            raise EngineError(f"no session on {start.day}")
        self.session: Session = found[0]
        if start.time >= self.session.close:
            raise EngineError(f"started at {start.time}, after the session closed")
        self.strategy = strategy
        self.guard = guard
        self.collar_bps = collar_bps
        self.prefix = prefix
        self.portfolio = Portfolio(start.cash, start.positions)
        self.last_close_value = start.last_close_value
        self.last_prices: dict[str, float] = dict(start.prices)
        self.price_times: dict[str, datetime] = dict.fromkeys(start.prices, start.time)
        self.orders: dict[str, LiveOrder] = {}  # today's, oldest first
        self.waiting: dict[str, _Waiting] = {}  # by ticker
        # Cancels asked for orders Alpaca has not accepted yet.
        self.cancel_when_accepted: set[str] = set()
        self.stopped: Stop | None = None
        self.now = start.time
        strategy.on_session_start(self.view(start.time))

    # What the strategy and Guard see.

    def view(self, now: datetime, positions: Mapping[str, float] | None = None) -> View:
        return View(
            now=now,
            session=self.session,
            cash=self.portfolio.cash,
            positions=self.portfolio.positions if positions is None else positions,
            last_prices=MappingProxyType(self.last_prices),
            open_orders=[o.order for o in self.orders.values() if o.is_open],
            price_times=MappingProxyType(self.price_times),
            last_close_value=self.last_close_value,
        )

    def _clock(self, now: datetime) -> datetime:
        """The engine's time: `now`, or the last time seen if the machine's clock went
        back, as it can by a few milliseconds when it is set from the network."""
        _require_utc("now", now)
        self.now = max(self.now, now)
        return self.now

    # Inputs.

    def on_bars(self, bars: Sequence[Bar], now: datetime) -> Step:
        """Bars that ended at the same moment, seen at `now`."""
        now = self._clock(now)
        if not bars:
            raise EngineError("on_bars needs bars; the clock alone is a Tick")
        end = bars[0].end
        by_ticker: dict[str, Bar] = {}
        for bar in bars:
            if bar.end != end or bar.ticker in by_ticker:
                raise EngineError(f"bars of one slice end together, once each: {bars}")
            if not (self.session.open <= bar.start and bar.end <= self.session.close):
                raise EngineError(f"{bar} lies outside the session")
            by_ticker[bar.ticker] = bar
        if now < end:
            raise EngineError(f"a bar ending at {end} seen at {now}, before it ended")
        for ticker, bar in sorted(by_ticker.items()):
            self.last_prices[ticker] = bar.close
            self.price_times[ticker] = bar.end
        if self.stopped:
            return Step()
        slice_ = MappingProxyType(dict(sorted(by_ticker.items())))
        return self._decide(self.strategy.on_bars(slice_, self.view(now)), now)

    def on_tick(self, tick: Tick) -> Step:
        now = self._clock(tick.time)
        if self.stopped:
            return Step()
        empty: Mapping[str, Bar] = MappingProxyType({})
        return self._decide(self.strategy.on_bars(empty, self.view(now)), now)

    def on_order_event(self, event: OrderEvent, now: datetime) -> Step:
        """An event about one of the engine's orders, from Alpaca or reconciliation."""
        now = self._clock(now)
        cid = event.client_order_id
        before = self.orders.get(cid)
        if before is None:
            raise EngineError(
                f"{type(event).__name__} for {cid}, an order the engine did not make"
            )
        after = apply(before, event)  # orders.InvalidTransition stops the engine
        self.orders[cid] = after
        records: list = []
        actions: list[Action] = []
        if cid in self.cancel_when_accepted and after.status is not Status.SUBMITTED:
            self.cancel_when_accepted.discard(cid)
            if after.status in (Status.ACCEPTED, Status.PARTLY_FILLED):
                self._cancel(cid, now, records, actions)
        fill = _new_fill(before, after, event.time)
        if fill is not None:
            self.portfolio.apply(fill)
            records.append(fill)
            self.strategy.on_fill(fill, self.view(now))
        # A stop cleared what was waiting, so nothing is released after one.
        self._release(after.order.intent.ticker, now, records, actions)
        return Step(tuple(records), tuple(actions))

    def stop(self, stop: Stop) -> Step:
        """Cancel every open order and decide nothing more."""
        now = self._clock(stop.time)
        if self.stopped:
            return Step()
        self.stopped = stop
        self.waiting.clear()
        records: list = [stop]
        actions: list[Action] = []
        self._cancel_all(now, records, actions)
        return Step(tuple(records), tuple(actions))

    def cancel_all(self, cancel: CancelAll) -> Step:
        """After a stop, cancel every order still open."""
        now = self._clock(cancel.time)
        if not self.stopped:
            raise EngineError("cancel_all only after a stop")
        records: list = [cancel]
        actions: list[Action] = []
        self._cancel_all(now, records, actions)
        return Step(tuple(records), tuple(actions))

    def _cancel_all(self, now: datetime, records: list, actions: list) -> None:
        for order in list(self.orders.values()):
            if order.is_open and order.status is not Status.PENDING_CANCEL:
                self._cancel(order.client_order_id, now, records, actions)

    def flatten(self, flatten: Flatten) -> Step:
        """After a stop, orders that close what Alpaca says is held.

        A ticker with an order still open is left for the next call, once Alpaca has
        confirmed its cancel. Guard judges each order against Alpaca's positions.
        """
        now = self._clock(flatten.time)
        if not self.stopped:
            raise EngineError("flatten only after a stop")
        records: list = [flatten]
        actions: list[Action] = []
        busy = {o.order.intent.ticker for o in self.orders.values() if o.is_open}
        for ticker, shares in sorted(flatten.positions.items()):
            if not shares or ticker in busy or ticker not in self.last_prices:
                continue
            side = Side.SELL if shares > 0 else Side.BUY
            limit = _limit(self.last_prices[ticker], side, self.collar_bps)
            quantity = round_shares(abs(shares))
            intent = OrderIntent(
                ticker, side, quantity, limit, now, "close after a stop"
            )
            records.append(intent)
            self._send(intent, now, records, actions, positions=flatten.positions)
        return Step(tuple(records), tuple(actions))

    # Decisions.

    def _decide(self, decisions: Iterable, now: datetime) -> Step:
        records: list = []
        actions: list[Action] = []
        for decision in decisions:
            if not isinstance(decision, OrderIntent | Cancellation | DecisionNote):
                raise EngineError(
                    f"a strategy answers with orders, cancels and notes, not {decision!r}"
                )
            if decision.time != now:
                raise EngineError(f"a decision made at {now} is dated {decision.time}")
            records.append(decision)
            if isinstance(decision, Cancellation):
                order = self.orders.get(decision.client_order_id)
                if order is None or not order.is_open:
                    raise EngineError(
                        f"cannot cancel {decision.client_order_id}: not open"
                    )
                if order.status is not Status.PENDING_CANCEL:
                    self._cancel(decision.client_order_id, now, records, actions)
            elif isinstance(decision, OrderIntent):
                if self._busy(decision.ticker):
                    # The ticker's cancels are not confirmed yet: hold the position the
                    # strategy wants, and send for it once they are.
                    held = self.portfolio.positions.get(decision.ticker, 0.0)
                    change = (
                        decision.quantity
                        if decision.side is Side.BUY
                        else -decision.quantity
                    )
                    self.waiting[decision.ticker] = _Waiting(
                        decision.ticker, round_shares(held + change), decision.reason
                    )
                else:
                    self.waiting.pop(decision.ticker, None)
                    self._send(decision, now, records, actions)
            if self.stopped:
                break  # Guard halted trading
        # A ticker whose orders were all cancelled, with no order decided after them,
        # still waits for its cancels: the strategy wanted the position it held.
        for cancelled in {
            self.orders[r.client_order_id].order.intent.ticker
            for r in records
            if isinstance(r, Cancellation)
        }:
            if cancelled not in self.waiting and not self.stopped:
                held = self.portfolio.positions.get(cancelled, 0.0)
                self.waiting[cancelled] = _Waiting(
                    cancelled, held, "kept after a cancel"
                )
        return Step(tuple(records), tuple(actions))

    def _busy(self, ticker: str) -> bool:
        return any(
            o.is_open and o.order.intent.ticker == ticker for o in self.orders.values()
        )

    def _cancel(self, cid: str, now: datetime, records: list, actions: list) -> None:
        if self.orders[cid].status is Status.SUBMITTED:
            # Alpaca cancels by its own id, which comes with its acceptance.
            self.cancel_when_accepted.add(cid)
            return
        event = CancelRequested(cid, now)
        self.orders[cid] = apply(self.orders[cid], event)
        records.append(event)
        actions.append(Cancel(cid))

    def _send(
        self,
        intent: OrderIntent,
        now: datetime,
        records: list,
        actions: list,
        positions: Mapping[str, float] | None = None,
    ) -> None:
        answer = self.guard.check(intent, self.view(now, positions))
        _check_answer(intent, answer, now)
        if answer.event is not None:
            records.append(answer.event)
            if answer.event.action is RiskAction.HALT and not self.stopped:
                step = self.stop(Stop(now, f"Guard halted: {answer.event.detail}"))
                records.extend(step.records)
                actions.extend(step.actions)
        if answer.intent is None:
            return
        cid = f"{self.prefix}-{self.session.day:%Y%m%d}-{len(self.orders) + 1:04d}"
        order = Order(cid, answer.intent, sent_at=now)
        sent = Sent(cid, now)
        self.orders[cid] = apply(LiveOrder(order), sent)
        records.extend([order, sent])
        actions.append(Submit(order))

    def _release(
        self, ticker: str, now: datetime, records: list, actions: list
    ) -> None:
        """Once a ticker's cancels have ended, send what still reaches its target."""
        waiting = self.waiting.get(ticker)
        if waiting is None or self._busy(ticker):
            return
        del self.waiting[ticker]
        held = self.portfolio.positions.get(ticker, 0.0)
        change = round_shares(waiting.target - held)
        if not change or ticker not in self.last_prices:
            return
        side = Side.BUY if change > 0 else Side.SELL
        limit = _limit(self.last_prices[ticker], side, self.collar_bps)
        reason = f"{waiting.reason}, after its cancel"
        intent = OrderIntent(
            ticker, side, round_shares(abs(change)), limit, now, reason
        )
        records.append(intent)
        self._send(intent, now, records, actions)


def _new_fill(before: LiveOrder, after: LiveOrder, time: datetime) -> Fill | None:
    """The shares a new fill total adds, at the price they cost."""
    added = round_shares(after.filled_quantity - before.filled_quantity)
    if added <= 0:
        return None
    paid = after.filled_quantity * after.average_price - before.filled_quantity * (
        before.average_price or 0.0
    )
    price = paid / added
    if not (math.isfinite(price) and price > 0):
        price = after.average_price  # a fraction of a share too small to divide
    intent = after.order.intent
    return Fill(
        after.client_order_id, intent.ticker, intent.side, added, price, 0.0, time
    )


def _check_answer(asked: OrderIntent, answer: GuardDecision, now: datetime) -> None:
    """Guard may send the intent as asked or fewer shares of it, nothing else."""
    if not isinstance(answer, GuardDecision):
        raise EngineError(f"Guard must answer with a GuardDecision, not {answer!r}")
    if answer.event is not None and answer.event.time != now:
        raise EngineError(f"Guard dated its event {answer.event.time}, not {now}")
    sent = answer.intent
    if sent is None:
        return
    if answer.event is None and sent != asked:
        raise EngineError(f"Guard changed {asked} without saying why")
    if (
        replace(sent, quantity=asked.quantity) != asked
        or sent.quantity > asked.quantity
    ):
        raise EngineError(f"Guard may only reduce the quantity of {asked}, sent {sent}")


# Replaying a day.

# What the core is told; everything else in a journal is what it answered.
INPUTS = (SessionStart, Bar, Tick, Stop, CancelAll, Flatten)


def _same(a, b) -> bool:
    # Compared as written, so a NaN a strategy saw equals itself. Imported here: the
    # journal imports this module's records.
    from qfin_alphaguard.live.journal import encode

    return json.dumps(encode(a), sort_keys=True) == json.dumps(
        encode(b), sort_keys=True
    )


def replay_day(
    entries: Iterable[tuple[datetime, object]],
    make_engine,
) -> tuple[LiveEngine, list]:
    """A day's journal fed back through a fresh core, built by make_engine(start).

    Every record the core answers must be the one in the journal, in order; the first
    that is not raises EngineError. Returns the core, now where the day left off, and
    what it answered that the journal does not hold: the outputs of a step a crash cut
    short, which were never acted on. The runner writes them and does not act on them:
    an order among them, sent in the core's books and never to Alpaca, is one
    reconciliation finds Alpaca never got.
    """
    engine: LiveEngine | None = None
    expected: list = []  # what the core answered, still to be met in the journal
    bars: list[Bar] = []
    bars_at: datetime | None = None

    def call(step: Step, own=None) -> None:
        nonlocal expected
        if expected:
            raise EngineError(
                f"the journal lacks {expected[0]!r} before the next input"
            )
        records = list(step.records)
        if own is not None:
            # A stop or a flatten is written by the core as its own first record.
            if not records or not _same(records[0], own):
                raise EngineError(f"{own!r} did not replay")
            records = records[1:]
        expected = records

    def flush_bars() -> None:
        nonlocal bars, bars_at
        if bars:
            if engine is None:
                raise EngineError("bars before the session's start")
            call(engine.on_bars(bars, bars_at))
            bars, bars_at = [], None

    for at, record in entries:
        if isinstance(record, Bar):
            if bars and (at != bars_at or record.end != bars[0].end):
                flush_bars()
            bars.append(record)
            bars_at = at
            continue
        is_output = not isinstance(record, INPUTS) and not (
            isinstance(record, OrderEvent)
            and not isinstance(record, Sent | CancelRequested)
        )
        if is_output:
            if bars:
                flush_bars()
            if not expected or not _same(expected[0], record):
                found = expected[0] if expected else "nothing"
                raise EngineError(
                    f"the journal holds {record!r}; the core answered {found!r}"
                )
            expected.pop(0)
            continue
        flush_bars()
        if isinstance(record, SessionStart):
            if engine is not None:
                raise EngineError("a journal holds one session")
            engine = make_engine(record)
            continue
        if engine is None:
            raise EngineError(f"{record!r} before the session's start")
        if isinstance(record, Tick):
            call(engine.on_tick(record))
        elif isinstance(record, Stop):
            call(engine.stop(record), own=record)
        elif isinstance(record, CancelAll):
            call(engine.cancel_all(record), own=record)
        elif isinstance(record, Flatten):
            call(engine.flatten(record), own=record)
        else:
            call(engine.on_order_event(record, at))
    flush_bars()
    if engine is None:
        raise EngineError("the journal holds no session's start")
    return engine, expected
