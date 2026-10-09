"""The backtest's event loop.

It replays bars in time order and keeps the rule that matters most: a decision
uses only what had happened by then, and an order fills no earlier than the
bar after the one it was decided from. Bars that end at the same moment form a
slice, which is handled in a fixed order:

1. The broker fills waiting orders from the slice's bars. Every waiting order
   was sent at the end of an earlier slice, so it fills on a later bar than the
   one it was decided from.
2. The portfolio books the fills, and the strategy hears of each one.
3. The strategy sees the slice, now complete, and answers with decisions.
4. Each order intent passes Guard; what Guard lets through becomes an order,
   sent at the end of this slice.

The engine is the referee. It checks every fill against the broker contract
(backtest/broker.py) and every Guard answer against GuardDecision, and stops
with an error rather than go on with a backtest whose results would be wrong.
At each session close it cancels the open orders, values the portfolio and
logs both.

Not handled yet (Phase 1, step 5b): splits and dividends. Prices are as traded,
so a split while a position is held distorts its value until corporate
actions adjust the share count.
"""

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType

from qfin_alphaguard.backtest.broker import SimulatedBroker
from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    Fill,
    Order,
    OrderIntent,
    RiskAction,
    RiskEvent,
    Side,
)
from qfin_alphaguard.guard.check import GuardDecision, RiskCheck
from qfin_alphaguard.portfolio import Portfolio
from qfin_alphaguard.sessions import Session, sessions
from qfin_alphaguard.strategy import Strategy, View


class BacktestError(RuntimeError):
    """A part of the backtest broke its contract; its results would be wrong."""


class LookAheadError(BacktestError):
    """Something used, or was dated by, information it could not yet have had."""


@dataclass(frozen=True)
class SessionClose:
    """The portfolio at a session's close, after its open orders were cancelled."""

    day: date
    time: datetime
    cash: float
    value: float  # cash plus every position at the last price seen


Record = OrderIntent | Order | Fill | Cancellation | RiskEvent | SessionClose


@dataclass(frozen=True)
class BacktestResult:
    log: tuple[Record, ...]  # every decision and what came of it, in order
    closes: tuple[SessionClose, ...]  # one per session: the daily values
    cash: float
    positions: Mapping[str, int]

    def fingerprint(self) -> str:
        """A hash of the log. The same inputs must always give the same one."""
        text = "\n".join(repr(record) for record in self.log)
        return hashlib.sha256(text.encode()).hexdigest()


def run_backtest(
    feed: Iterable[Bar],
    strategy: Strategy,
    broker: SimulatedBroker,
    guard: RiskCheck,
    cash: float,
) -> BacktestResult:
    """Replay `feed`, bars in time order, through `strategy`, Guard and `broker`."""
    engine = _Engine(strategy, broker, guard, cash)
    for bars in _slices(feed):
        engine.handle(bars)
    engine.finish()
    return engine.result()


@dataclass
class _OpenOrder:
    order: Order
    remaining: int


class _Engine:
    def __init__(
        self, strategy: Strategy, broker: SimulatedBroker, guard: RiskCheck, cash: float
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.guard = guard
        self.portfolio = Portfolio(cash)
        self.last_prices: dict[str, float] = {}
        self.prices = MappingProxyType(self.last_prices)
        self.open: dict[str, _OpenOrder] = {}  # by client_order_id, oldest first
        self.log: list[Record] = []
        self.closes: list[SessionClose] = []
        self.session: Session | None = None
        self.orders_sent = 0
        self.halted = False

    def view(self, now: datetime) -> View:
        return View(
            now=now,
            session=self.session,
            cash=self.portfolio.cash,
            positions=self.portfolio.positions,
            last_prices=self.prices,
            open_orders=tuple(entry.order for entry in self.open.values()),
        )

    def handle(self, bars: dict[str, Bar]) -> None:
        first = next(iter(bars.values()))
        if self.session is None or first.start.date() != self.session.day:
            if self.session is not None:
                self.close_session()
            self.session = _session_on(first.start.date())
            self.strategy.on_session_start(self.view(self.session.open))
        if not (self.session.open <= first.start and first.end <= self.session.close):
            raise BacktestError(
                f"bars from {first.start} to {first.end} lie outside the session "
                f"from {self.session.open} to {self.session.close}"
            )
        now = first.end  # the moment the slice is complete
        for ticker, bar in bars.items():
            self.last_prices[ticker] = bar.close
        for bar in bars.values():
            for fill in self.broker.on_bar(bar):
                self.book(fill, bar)
                self.strategy.on_fill(fill, self.view(now))
        for decision in self.strategy.on_bars(MappingProxyType(bars), self.view(now)):
            self.decide(decision, now)

    def book(self, fill: Fill, bar: Bar) -> None:
        """Check a fill against the broker contract, then book it."""
        entry = self.open.get(fill.client_order_id)
        if entry is None:
            raise BacktestError(f"fill for {fill.client_order_id}, not an open order")
        intent = entry.order.intent
        if (fill.ticker, fill.side) != (intent.ticker, intent.side):
            raise BacktestError(f"fill {fill} does not match its order {entry.order}")
        if fill.ticker != bar.ticker:
            raise BacktestError(f"fill for {fill.ticker} came with a {bar.ticker} bar")
        if fill.time < entry.order.sent_at:
            raise LookAheadError(
                f"{fill.client_order_id} filled at {fill.time}, "
                f"before it was sent at {entry.order.sent_at}"
            )
        if not (bar.start <= fill.time <= bar.end):
            raise BacktestError(f"fill at {fill.time} is outside its bar {bar}")
        if not (bar.low <= fill.price <= bar.high):
            raise BacktestError(
                f"fill at {fill.price} is outside the bar's range "
                f"{bar.low} to {bar.high}: that price never traded"
            )
        above = intent.side is Side.BUY and fill.price > intent.limit_price
        below = intent.side is Side.SELL and fill.price < intent.limit_price
        if above or below:
            raise BacktestError(f"fill at {fill.price} breaks the limit of {intent}")
        if fill.quantity > entry.remaining:
            raise BacktestError(
                f"fill of {fill.quantity} exceeds the {entry.remaining} shares "
                f"left on {fill.client_order_id}"
            )
        self.portfolio.apply(fill)
        entry.remaining -= fill.quantity
        if entry.remaining == 0:
            del self.open[fill.client_order_id]
        self.log.append(fill)

    def decide(self, decision: object, now: datetime) -> None:
        if not isinstance(decision, OrderIntent | Cancellation):
            raise TypeError(
                f"a strategy answers with OrderIntent or Cancellation, not {decision!r}"
            )
        if decision.time != now:
            raise LookAheadError(
                f"a decision made at {now} is dated {decision.time}: {decision}"
            )
        if isinstance(decision, Cancellation):
            if decision.client_order_id not in self.open:
                raise BacktestError(
                    f"cannot cancel {decision.client_order_id}: not open"
                )
            self.cancel(decision)
            return
        self.log.append(decision)
        if self.halted:
            self.log.append(
                RiskEvent(
                    time=now,
                    rule="halt",
                    action=RiskAction.REJECT,
                    detail="trading was halted earlier in this run",
                    ticker=decision.ticker,
                )
            )
            return
        answer = self.guard.check(decision, self.view(now))
        _check_answer(decision, answer, now)
        if answer.event is not None:
            self.log.append(answer.event)
            if answer.event.action is RiskAction.HALT:
                self.halt(now)
        if answer.intent is not None:
            self.send(answer.intent, now)

    def send(self, intent: OrderIntent, now: datetime) -> None:
        self.orders_sent += 1
        # Numbered in order, so the same run always gives the same ids.
        order = Order(f"bt-{self.orders_sent:07d}", intent, sent_at=now)
        self.broker.submit(order)
        self.open[order.client_order_id] = _OpenOrder(order, intent.quantity)
        self.log.append(order)

    def cancel(self, cancellation: Cancellation) -> None:
        self.broker.cancel(cancellation.client_order_id, cancellation.time)
        del self.open[cancellation.client_order_id]
        self.log.append(cancellation)

    def halt(self, now: datetime) -> None:
        self.halted = True
        for client_order_id in list(self.open):
            self.cancel(Cancellation(client_order_id, now, "halt"))

    def close_session(self) -> None:
        close = self.session.close
        for client_order_id in list(self.open):
            self.cancel(Cancellation(client_order_id, close, "session close"))
        record = SessionClose(
            day=self.session.day,
            time=close,
            cash=self.portfolio.cash,
            value=self.portfolio.value(self.last_prices),
        )
        self.log.append(record)
        self.closes.append(record)
        self.strategy.on_session_end(self.view(close))

    def finish(self) -> None:
        if self.session is not None:
            self.close_session()

    def result(self) -> BacktestResult:
        return BacktestResult(
            log=tuple(self.log),
            closes=tuple(self.closes),
            cash=self.portfolio.cash,
            positions=MappingProxyType(dict(self.portfolio.positions)),
        )


def _check_answer(asked: OrderIntent, answer: GuardDecision, now: datetime) -> None:
    """Guard may send the intent as asked or fewer shares of it, nothing else."""
    if not isinstance(answer, GuardDecision):
        raise BacktestError(f"Guard must answer with a GuardDecision, not {answer!r}")
    if answer.event is not None and answer.event.time != now:
        raise LookAheadError(f"Guard dated its event {answer.event.time}, not {now}")
    sent = answer.intent
    if sent is None:
        return
    if answer.event is None and sent != asked:
        raise BacktestError(f"Guard changed {asked} without saying why")
    same = (sent.ticker, sent.side, sent.limit_price, sent.time) == (
        asked.ticker,
        asked.side,
        asked.limit_price,
        asked.time,
    )
    if not same or sent.quantity > asked.quantity:
        raise BacktestError(
            f"Guard may only reduce the quantity of {asked}, sent {sent}"
        )


def _slices(feed: Iterable[Bar]) -> Iterator[dict[str, Bar]]:
    """Bars grouped by the moment they end, in time order, by ticker."""
    current: dict[str, Bar] = {}
    for bar in feed:
        if current:
            end = next(iter(current.values())).end
            if bar.end < end:
                raise LookAheadError(
                    f"bars out of time order: {bar.ticker} ends at {bar.end}, "
                    f"after a bar that ended at {end}"
                )
            if bar.end > end:
                yield dict(sorted(current.items()))
                current = {}
        if current:
            if bar.ticker in current:
                raise BacktestError(f"two bars for {bar.ticker} end at {bar.end}")
            if bar.start != next(iter(current.values())).start:
                raise BacktestError(f"bars of different lengths end at {bar.end}")
        current[bar.ticker] = bar
    if current:
        yield dict(sorted(current.items()))


def _session_on(day: date) -> Session:
    found = sessions(day, day)
    if not found:
        raise BacktestError(f"bars on {day}, when the exchange was closed")
    return found[0]
