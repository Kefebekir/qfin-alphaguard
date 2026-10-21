"""The live engine's runner: the core connected to Alpaca, for one session (decision 0021).

`qfin live` runs one session, started before the open on the engine's server.

1. The morning (prepare). Guard's limits of the QQQ account, with `live_trading` false:
   this runner trades only the paper account. The day's journal, which one engine at a
   time may hold. The sessions before today, fetched once and kept beside the journal,
   which warm the strategy (warmup.py). Alpaca's account. A journal already started means
   the engine was restarted: the day is replayed from it (engine.replay_day), and what a
   crash cut short is written and not acted on.
2. The day (Runner.run). Alpaca's minute bars and trade updates arrive on their streams,
   and a clock watches the time. All three put what they see in one queue, so the core
   takes one input at a time, in the order they came. Each input is written to the journal
   before the core sees it, and the core's answer is written before it is acted on.
   Orders go out and cancels are asked for by REST. Reconciliation runs at the start,
   whenever the trade updates reconnect, and when an answer to a send or a cancel is lost.
3. The close. Day orders expire at 16:00 New York time. The runner asks Alpaca about any
   still open, reconciles a last time and ends.

What stops the engine, and what follows (Efe, 17 October 2026: close, then stop):
- a difference reconciliation cannot explain;
- an answer from Alpaca the engine cannot read, or an event the order rules refuse;
- a stream down for STREAM_GRACE;
- an error in the runner itself.
The core cancels every open order and the runner waits for Alpaca to confirm. Then it
closes what Alpaca says is held, over a few rounds if a close does not fill, and ends
with an alert. If Alpaca cannot be reached, or holds open orders the engine did not make,
it only alerts: a person must look.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import boto3
import polars as pl
from botocore.exceptions import BotoCoreError, ClientError
from websockets.exceptions import WebSocketException

from qfin_alphaguard.data.alpaca import AlpacaClient
from qfin_alphaguard.events import Bar
from qfin_alphaguard.guard.config import GuardConfig
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.live.alpaca_broker import AlpacaBroker, BrokerError, OutcomeUnknown
from qfin_alphaguard.live.engine import (
    CancelAll,
    Flatten,
    LiveEngine,
    SessionStart,
    Step,
    Stop,
    Submit,
    Tick,
    replay_day,
)
from qfin_alphaguard.live.journal import Journal
from qfin_alphaguard.live.orders import OrderEvent, Rejected
from qfin_alphaguard.live.reconcile import reconcile
from qfin_alphaguard.live.streams import StreamError, minute_bars, trade_updates
from qfin_alphaguard.live.warmup import history, sessions_before, warm_up
from qfin_alphaguard.momentum import MomentumSettings, NoiseAreaMomentum
from qfin_alphaguard.sessions import Session

log = logging.getLogger("qfin.live")

TICKER = "QQQ"
STREAM_GRACE = timedelta(minutes=2)  # a stream down this long stops the engine
LAST_MINUTE_TICK = timedelta(seconds=55)  # before the close: the day's last decision
END_WAIT = timedelta(minutes=2)  # after the close: Alpaca has expired the day's orders
WATCH_EVERY = 5.0  # seconds between the clock's looks
RECONNECT = (1.0, 2.0, 5.0, 10.0, 20.0)  # seconds before each try to reconnect
CANCEL_WAIT = 30.0  # seconds for Alpaca to confirm the cancels of a stop
FILL_WAIT = 60.0  # seconds for a close after a stop to fill
FLATTEN_ROUNDS = 3
POLL = 2.0  # seconds between questions to Alpaca while waiting


class RunnerError(RuntimeError):
    """The session cannot start, or must stop."""


@dataclass(frozen=True)
class DayFiles:
    """Where one session's files live: data/live/<day>/."""

    root: Path
    day: date

    @property
    def folder(self) -> Path:
        return self.root / self.day.isoformat()

    @property
    def journal(self) -> Path:
        return self.folder / "journal.jsonl"

    @property
    def warm_up(self) -> Path:
        return self.folder / "warm_up.parquet"

    @property
    def alert(self) -> Path:
        return self.folder / "ALERT.txt"


# The morning.


def past_sessions(
    client: AlpacaClient, day: date, path: Path
) -> list[tuple[Session, list[Bar]]]:
    """The sessions before `day` with IEX's bars, fetched once and kept at `path`.

    A restarted engine warms its strategy on the same bars as in the morning, so its
    journal replays to the same decisions even if Alpaca's history changes meanwhile.
    """
    by_day = {s.day: s for s in sessions_before(day)}
    if not path.exists():
        past = history(client, TICKER, day)
        rows = [
            (b.start, b.open, b.high, b.low, b.close, b.volume)
            for _, bars in past
            for b in bars
        ]
        frame = pl.DataFrame(
            rows,
            schema={
                "start": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
            orient="row",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    frame = pl.read_parquet(path).sort("start")
    out: dict[date, list[Bar]] = {}
    for r in frame.iter_rows(named=True):
        bar = Bar(
            TICKER,
            r["start"],
            60,
            r["open"],
            r["high"],
            r["low"],
            r["close"],
            r["volume"],
        )
        out.setdefault(bar.start.date(), []).append(bar)
    return [(by_day[d], bars) for d, bars in sorted(out.items()) if d in by_day]


def warmed_strategy(
    limits: GuardConfig, past: Sequence[tuple[Session, Sequence[Bar]]], cash: float
) -> NoiseAreaMomentum:
    """The QQQ strategy, with Guard's collar, warmed on `past`."""
    settings = MomentumSettings(ticker=TICKER, collar_bps=limits.price_collar_bps)
    strategy = NoiseAreaMomentum(settings)
    warm_up(strategy, past, cash)
    return strategy


def make_engine_for(
    limits: GuardConfig, past: Sequence[tuple[Session, Sequence[Bar]]]
) -> Callable[[SessionStart], LiveEngine]:
    """What builds the day's core: a strategy warmed on `past`, behind Guard."""

    def make(start: SessionStart) -> LiveEngine:
        strategy = warmed_strategy(limits, past, start.cash)
        return LiveEngine(strategy, Guard(limits), start, limits.price_collar_bps)

    return make


def prepare(
    day: date,
    files: DayFiles,
    limits: GuardConfig,
    broker: AlpacaBroker,
    data: AlpacaClient,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[LiveEngine, SessionStart, Journal]:
    """The day's core and journal, from the morning or from where a restart left it."""
    if limits.live_trading:
        raise RunnerError(
            "live_trading is true; this runner trades only the paper account"
        )
    journal = Journal(files.journal)
    try:
        past = past_sessions(data, day, files.warm_up)
        make = make_engine_for(limits, past)
        if journal.entries:
            core, leftover = replay_day(journal.entries, make)
            at = now()
            for record in leftover:
                journal.write(record, at=at)  # cut short by a crash; never acted on
            log.warning(
                "restarted: %d records replayed, %d left by the crash written",
                len(journal.entries) - len(leftover),
                len(leftover),
            )
            return core, journal.entries[0].record, journal
        account = broker.account()
        if not account.can_trade:
            raise RunnerError(f"the paper account cannot trade: {account.status}")
        held = broker.positions()
        last = {b.ticker: b.close for _, bars in past for b in bars[-1:]}
        missing = sorted(set(held) - set(last))
        if missing:
            raise RunnerError(f"held, with no history to price them: {missing}")
        at = now()
        start = SessionStart(
            day,
            at,
            account.cash,
            account.last_equity,
            held,
            {t: last[t] for t in held},
        )
        journal.write(start, at=at)
        return make(start), start, journal
    except BaseException:
        journal.close()
        raise


# What the streams and the clock put in the queue. Not written: what they carry is.


@dataclass(frozen=True)
class Bars:
    bars: tuple[Bar, ...]
    at: datetime


@dataclass(frozen=True)
class Events:
    events: tuple[OrderEvent, ...]
    at: datetime


@dataclass(frozen=True)
class StreamState:
    name: str  # "bars" or "trades"
    up: bool
    at: datetime


@dataclass(frozen=True)
class Clock:
    at: datetime
    last_minute: bool = False  # time for the day's last decision
    closed: bool = False  # the session is over


@dataclass(frozen=True)
class Fatal:
    reason: str
    at: datetime


Input = Bars | Events | StreamState | Clock | Fatal


# The day.


class Runner:
    """Takes the inputs one at a time: writes, asks the core, acts."""

    def __init__(
        self,
        core: LiveEngine,
        start: SessionStart,
        journal: Journal,
        broker: AlpacaBroker,
        alert: Callable[[str], None],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], object] = asyncio.sleep,
        blocking_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.core = core
        self.start = start
        self.journal = journal
        self.broker = broker
        self.alert = alert
        self.now = now
        self.sleep = sleep
        self.blocking_sleep = blocking_sleep
        self.down_since: dict[str, datetime] = {}  # a stream down, since when

    async def run(self, inputs: asyncio.Queue) -> str:
        """Until the session closes ("closed") or the engine stops ("stopped")."""
        try:
            await self.reconcile("the session starts")
            while True:
                item = await inputs.get()
                if isinstance(item, Clock) and item.closed:
                    await self.close()
                    return "closed"
                await self.handle(item)
        except Exception as error:  # noqa: BLE001 - every failure ends in the same stop
            await self.stop(f"{type(error).__name__}: {error}")
            return "stopped"

    async def handle(self, item: Input) -> None:
        if isinstance(item, Bars):
            for bar in item.bars:
                self.journal.write(bar, at=item.at)
            await self.take(self.core.on_bars(list(item.bars), item.at), item.at)
        elif isinstance(item, Events):
            for event in item.events:
                await self.event(event, item.at)
        elif isinstance(item, StreamState):
            if item.up:
                self.down_since.pop(item.name, None)
                if item.name == "trades":
                    # Alpaca does not send again what the stream missed.
                    await self.reconcile("the trade updates connected")
            else:
                self.down_since.setdefault(item.name, item.at)
        elif isinstance(item, Clock):
            # Data matters while the market is open: a stream down before it does not stop
            # the engine, unless it is still down STREAM_GRACE after the open.
            for name, since in self.down_since.items():
                if item.at - max(since, self.core.session.open) >= STREAM_GRACE:
                    raise RunnerError(f"the {name} stream has been down since {since}")
            if item.last_minute:
                tick = Tick(item.at)
                self.journal.write(tick, at=item.at)
                await self.take(self.core.on_tick(tick), item.at)
        elif isinstance(item, Fatal):
            raise RunnerError(item.reason)
        else:
            raise RunnerError(f"an input the runner does not know: {item!r}")

    async def call(self, fn, *args, **kwargs):
        """A REST call, off the event loop, so the streams keep being read."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def take(self, step: Step, at: datetime) -> None:
        """Write what the core answered, then act on it."""
        for record in step.records:
            self.journal.write(record, at=at)
        for action in step.actions:
            if isinstance(action, Submit):
                cid = action.order.client_order_id
                try:
                    events = await self.call(self.broker.submit, action.order)
                except OutcomeUnknown as lost:
                    log.warning("%s", lost)
                    await self.reconcile(f"the answer to {cid}'s send was lost")
                    continue
            else:
                cid = action.client_order_id
                try:
                    events = await self.call(self.broker.cancel, cid)
                except OutcomeUnknown as lost:
                    log.warning("%s", lost)
                    await self.reconcile(f"the answer to {cid}'s cancel was lost")
                    continue
            for event in events:
                if isinstance(event, Rejected):
                    log.warning("Alpaca refused %s: %s", cid, event.reason)
                await self.event(event, self.now())

    async def event(self, event: OrderEvent, at: datetime) -> None:
        self.journal.write(event, at=at)
        await self.take(self.core.on_order_event(event, at), at)

    async def reconcile(self, why: str) -> None:
        log.info("reconciling: %s", why)
        result = await self.call(
            reconcile,
            dict(self.core.orders),
            self.broker,
            self.start.positions,
            sleep=self.blocking_sleep,
        )
        at = self.now()
        for event in result.events:
            await self.event(event, at)
        if not result.agrees:
            raise RunnerError(
                f"reconciliation ({why}): " + "; ".join(result.differences)
            )

    async def settle(self, seconds: float) -> bool:
        """Ask Alpaca about the engine's open orders until none is left, for `seconds`."""
        waited = 0.0
        while True:
            open_ = [cid for cid, o in self.core.orders.items() if o.is_open]
            if not open_:
                return True
            if waited >= seconds:
                return False
            await self.sleep(POLL)
            waited += POLL
            for cid in open_:
                placed = await self.call(self.broker.order, cid)
                if placed is None:
                    raise RunnerError(f"{cid} is open here and unknown to Alpaca")
                at = self.now()
                for event in placed.events():
                    await self.event(event, at)

    async def stop(self, reason: str) -> None:
        """Close, then stop: cancel, wait, close what Alpaca holds, alert."""
        self.alert(f"stopping: {reason}")
        try:
            at = self.now()
            await self.take(self.core.stop(Stop(at, reason[:500])), at)
            if not await self.settle(CANCEL_WAIT):
                self.alert("orders still open after their cancels: a person must look")
                return
            for _ in range(FLATTEN_ROUNDS):
                foreign = [
                    o.client_order_id
                    for o in await self.call(self.broker.open_orders)
                    if o.client_order_id not in self.core.orders
                ]
                if foreign:
                    self.alert(
                        f"open orders the engine did not make, {foreign}: not closing"
                    )
                    return
                held = await self.call(self.broker.positions)
                if not held:
                    self.alert("nothing is held; the engine has stopped")
                    return
                at = self.now()
                await self.take(self.core.flatten(Flatten(at, held)), at)
                if await self.settle(FILL_WAIT):
                    continue  # filled: look at what Alpaca holds now
                at = self.now()
                cancel = CancelAll(at, "the close did not fill")
                await self.take(self.core.cancel_all(cancel), at)
                if not await self.settle(CANCEL_WAIT):
                    self.alert("a close's cancel was not confirmed: a person must look")
                    return
            held = await self.call(self.broker.positions)
            self.alert(f"still held after {FLATTEN_ROUNDS} rounds of closing: {held}")
        except Exception as error:  # noqa: BLE001 - the stop must end in an alert
            self.alert(
                f"the stop itself failed, {type(error).__name__}: {error}; a person must look"
            )

    async def close(self) -> None:
        """The session is over: the day's orders have expired; a last reconciliation."""
        if not await self.settle(CANCEL_WAIT):
            raise RunnerError("orders still open after the close")
        await self.reconcile("the session closed")
        held = await self.call(self.broker.positions)
        if held:
            self.alert(f"held overnight, to be closed at the next open: {held}")
        log.info("session closed: %d orders", len(self.core.orders))


# The streams and the clock.


async def feed_bars(
    queue: asyncio.Queue,
    keys: tuple[str, str],
    now: Callable[[], datetime],
    sleep: Callable[[float], object] = asyncio.sleep,
    stream=minute_bars,
) -> None:
    """IEX's minute bars into the queue, reconnecting when the connection is lost."""
    attempt = 0

    def connected() -> None:
        queue.put_nowait(StreamState("bars", True, now()))

    while True:
        try:
            async for bar in stream(
                *keys, [TICKER], feed="iex", on_connected=connected
            ):
                attempt = 0
                queue.put_nowait(Bars((bar,), now()))
            problem = "the stream ended"
        except (StreamError, WebSocketException, OSError, TimeoutError) as error:
            problem = f"{type(error).__name__}: {error}"
        log.warning("bars stream: %s", problem)
        queue.put_nowait(StreamState("bars", False, now()))
        await sleep(RECONNECT[min(attempt, len(RECONNECT) - 1)])
        attempt += 1


async def feed_trades(
    queue: asyncio.Queue,
    keys: tuple[str, str],
    now: Callable[[], datetime],
    sleep: Callable[[float], object] = asyncio.sleep,
    stream=trade_updates,
) -> None:
    """The account's trade updates into the queue, reconnecting when the connection is
    lost. An update the engine cannot read stops it."""
    attempt = 0

    def connected() -> None:
        queue.put_nowait(StreamState("trades", True, now()))

    while True:
        try:
            async for events in stream(*keys, on_connected=connected):
                attempt = 0
                queue.put_nowait(Events(tuple(events), now()))
            problem = "the stream ended"
        except BrokerError as error:
            queue.put_nowait(Fatal(f"a trade update: {error}", now()))
            return
        except (StreamError, WebSocketException, OSError, TimeoutError) as error:
            problem = f"{type(error).__name__}: {error}"
        log.warning("trade updates: %s", problem)
        queue.put_nowait(StreamState("trades", False, now()))
        await sleep(RECONNECT[min(attempt, len(RECONNECT) - 1)])
        attempt += 1


async def keep_time(
    queue: asyncio.Queue,
    session: Session,
    now: Callable[[], datetime],
    sleep: Callable[[float], object] = asyncio.sleep,
) -> None:
    """Every few seconds a look at the clock; the day's last minute; the close."""
    last_minute = session.close - LAST_MINUTE_TICK
    end = session.close + END_WAIT
    ticked = False
    while True:
        at = now()
        if at >= end:
            queue.put_nowait(Clock(at, closed=True))
            return
        due = not ticked and at >= last_minute
        ticked = ticked or due
        queue.put_nowait(Clock(at, last_minute=due))
        wait = WATCH_EVERY
        if not ticked:
            wait = min(wait, max((last_minute - at).total_seconds(), 0.0))
        await sleep(max(wait, 0.01))


class Backup:
    """The day's folder kept in S3 as well: a server's disk does not outlive its task.

    `pull` fetches what the folder lacks before the day starts, so a task started again
    finds its journal; `push` sends the folder up, every minute and at the end. A push
    that fails is logged and tried again a minute later: trading goes on.
    """

    NAMES = (
        "journal.jsonl",
        "warm_up.parquet",
        "ALERT.txt",
        "report.txt",
        "report.html",
    )

    def __init__(self, files: DayFiles, bucket: str, s3=None) -> None:
        self.files = files
        self.bucket = bucket
        self.s3 = s3 or boto3.client("s3")

    def key(self, name: str) -> str:
        return f"data/live/{self.files.day.isoformat()}/{name}"

    def pull(self) -> list[str]:
        pulled = []
        for name in self.NAMES:
            local = self.files.folder / name
            if local.exists():
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.s3.download_file(self.bucket, self.key(name), str(local))
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in (
                    "404",
                    "NoSuchKey",
                ):
                    raise
                continue
            pulled.append(name)
        return pulled

    def push(self) -> None:
        for name in self.NAMES:
            local = self.files.folder / name
            if not local.exists():
                continue
            try:
                self.s3.upload_file(str(local), self.bucket, self.key(name))
            except (ClientError, BotoCoreError, OSError) as error:
                log.warning("backup of %s failed: %s", name, error)


async def back_up(
    backup: Backup,
    every: float = 60.0,
    sleep: Callable[[float], object] = asyncio.sleep,
) -> None:
    while True:
        await sleep(every)
        await asyncio.to_thread(backup.push)


def alert_to(path: Path, topic: str | None = None, sns=None) -> Callable[[str], None]:
    """An alert: logged, added to the day's ALERT.txt, and e-mailed through SNS's `topic`."""
    if topic and sns is None:
        sns = boto3.client("sns")

    def alert(message: str) -> None:
        log.error("ALERT: %s", message)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(f"{datetime.now(UTC).isoformat()} {message}\n")
        if topic:
            try:
                sns.publish(
                    TopicArn=topic,
                    Subject=f"qfin live {path.parent.name}: alert",
                    Message=message,
                )
            except (ClientError, BotoCoreError) as error:
                log.error("the alert could not be e-mailed: %s", error)

    return alert


async def run_session(
    day: date,
    root: Path,
    limits: GuardConfig,
    keys: tuple[str, str],
    bucket: str | None = None,
    topic: str | None = None,
) -> str:
    """One session of the paper account, from the morning to the close."""
    files = DayFiles(root, day)
    backup = Backup(files, bucket) if bucket else None
    if backup:
        pulled = backup.pull()
        if pulled:
            log.warning("started again: %s fetched from S3", ", ".join(pulled))
    broker = AlpacaBroker(*keys)
    data = AlpacaClient(*keys)
    alert = alert_to(files.alert, topic)
    try:
        core, start, journal = prepare(day, files, limits, broker, data)
    except Exception as error:
        alert(f"the session could not start, {type(error).__name__}: {error}")
        raise
    queue: asyncio.Queue = asyncio.Queue()

    def now() -> datetime:
        return datetime.now(UTC)

    tasks = [
        asyncio.create_task(feed_bars(queue, keys, now)),
        asyncio.create_task(feed_trades(queue, keys, now)),
        asyncio.create_task(keep_time(queue, core.session, now)),
    ]
    if backup:
        tasks.append(asyncio.create_task(back_up(backup)))
    try:
        outcome = await Runner(core, start, journal, broker, alert).run(queue)
        if topic and outcome == "closed":
            held = await asyncio.to_thread(broker.positions)
            filled = sum(o.filled_quantity > 0 for o in core.orders.values())
            sns = boto3.client("sns")
            sns.publish(
                TopicArn=topic,
                Subject=f"qfin live {day}: closed",
                Message=(
                    f"The session of {day} closed: {len(core.orders)} orders, "
                    f"{filled} filled; held now: {held or 'nothing'}."
                ),
            )
        return outcome
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        journal.close()
        if backup:
            await asyncio.to_thread(backup.push)


async def check_session(
    day: date,
    root: Path,
    limits: GuardConfig,
    keys: tuple[str, str],
    seconds: float = 30.0,
    broker=None,
    data=None,
    bars=minute_bars,
    trades=trade_updates,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> str:
    """The morning, a reconciliation and both streams, then out: nothing is traded.

    The check keeps its own folder, `root`/check/, away from the day's journal. Alpaca's
    free plan allows one market data connection, so do not check during a session.
    """
    files = DayFiles(root / "check", day)
    broker = broker or AlpacaBroker(*keys)
    data = data or AlpacaClient(*keys)
    core, start, journal = prepare(day, files, limits, broker, data, now=now)
    try:
        result = await asyncio.to_thread(
            reconcile, dict(core.orders), broker, start.positions
        )
        if not result.agrees:
            raise RunnerError("reconciliation: " + "; ".join(result.differences))
        connected: set[str] = set()

        async def listen(name: str, stream) -> None:
            async for _ in stream:
                pass

        tasks = [
            asyncio.create_task(
                listen(
                    "bars",
                    bars(
                        *keys,
                        [TICKER],
                        feed="iex",
                        on_connected=lambda: connected.add("bars"),
                    ),
                )
            ),
            asyncio.create_task(
                listen(
                    "trades",
                    trades(*keys, on_connected=lambda: connected.add("trades")),
                )
            ),
        ]
        waited = 0.0
        while connected != {"bars", "trades"} and waited < seconds:
            await asyncio.sleep(0.1)
            waited += 0.1
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        missing = {"bars", "trades"} - connected
        if missing:
            raise RunnerError(f"not connected in {seconds:g} s: {sorted(missing)}")
        held = dict(start.positions) or "nothing"
        return (
            f"checked: account {start.cash:,.2f} USD, held {held}; strategy warmed on "
            f"{len(core.strategy.closes)} closes; reconciled; both streams connected"
        )
    finally:
        journal.close()
