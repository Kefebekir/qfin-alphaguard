"""Alpaca's live streams: the account's trade updates, and minute bars (decision 0021).

Both are WebSockets, read as async iterators. A stream connects, authenticates and
subscribes, then yields what arrives:
- trade_updates yields the events in each update, for orders.apply;
- minute_bars yields Bar events, the same as the backtest's: a bar is labelled with the
  minute it starts, in UTC, and arrives just after that minute ends. A minute without a
  trade on the feed has no bar.

The handshake must finish within HANDSHAKE_SECONDS. A refused key or subscription, an
error from Alpaca, or a message that cannot be read raises StreamError: the engine stops.
A lost connection ends the iterator or raises the websockets library's ConnectionClosed;
the engine reconnects. Alpaca does not send again the trade updates of the time the
connection was down, so after a reconnect the engine asks for its open orders.

Alpaca's free plan allows one market data connection at a time. Shapes as Alpaca sent
them on 17 October 2026: the trade updates come as binary frames, the bars as text.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable

from websockets.asyncio.client import connect as websocket_connect

from qfin_alphaguard.events import Bar
from qfin_alphaguard.live.alpaca_broker import _time, trade_update_events
from qfin_alphaguard.live.orders import OrderEvent

TRADING_STREAM = "wss://paper-api.alpaca.markets/stream"
DATA_STREAM = "wss://stream.data.alpaca.markets/v2/{feed}"
FEEDS = ("iex", "sip", "test")  # test: Alpaca's stream of the fake FAKEPACA, always on
HANDSHAKE_SECONDS = 10


class StreamError(Exception):
    """A stream that refused the engine, or said what it cannot read. The engine stops."""


def _json(raw: str | bytes):
    try:
        return json.loads(raw)
    except ValueError as error:
        raise StreamError(f"a message that is not JSON: {raw[:100]!r}") from error


def _object(raw: str | bytes) -> dict:
    message = _json(raw)
    if not isinstance(message, dict):
        raise StreamError(f"trade updates come one object at a time, got {message!r}")
    return message


async def trade_updates(
    key_id: str, secret: str, connect: Callable = websocket_connect
) -> AsyncIterator[list[OrderEvent]]:
    """The events of each trade update on the paper account, as they arrive."""
    async with connect(TRADING_STREAM) as socket:
        async with asyncio.timeout(HANDSHAKE_SECONDS):
            await socket.send(
                json.dumps({"action": "auth", "key": key_id, "secret": secret})
            )
            reply = _object(await socket.recv())
            if reply.get("stream") != "authorization" or (
                reply.get("data", {}).get("status") != "authorized"
            ):
                raise StreamError(f"trade updates refused the keys: {reply}")
            await socket.send(
                json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}})
            )
            reply = _object(await socket.recv())
            if reply.get("stream") != "listening" or "trade_updates" not in (
                reply.get("data", {}).get("streams") or []
            ):
                raise StreamError(f"trade updates refused to listen: {reply}")
        async for raw in socket:
            message = _object(raw)
            if message.get("stream") != "trade_updates":
                raise StreamError(f"an unexpected message on trade updates: {message}")
            yield trade_update_events(message.get("data") or {})


async def minute_bars(
    key_id: str,
    secret: str,
    symbols: Iterable[str],
    feed: str = "iex",
    connect: Callable = websocket_connect,
) -> AsyncIterator[Bar]:
    """Each completed minute bar of `symbols` on `feed`, as it arrives."""
    if feed not in FEEDS:
        raise ValueError(f"feed must be one of {FEEDS}, got {feed!r}")
    wanted = sorted(set(symbols))
    if not wanted:
        raise ValueError("no symbols to subscribe to")
    async with connect(DATA_STREAM.format(feed=feed)) as socket:
        async with asyncio.timeout(HANDSHAKE_SECONDS):
            await _control(socket, "connected")
            await socket.send(
                json.dumps({"action": "auth", "key": key_id, "secret": secret})
            )
            await _control(socket, "authenticated")
            await socket.send(json.dumps({"action": "subscribe", "bars": wanted}))
            reply = await _messages(socket)
            if len(reply) != 1 or reply[0].get("T") != "subscription":
                raise StreamError(f"the bars subscription was not confirmed: {reply}")
            if sorted(reply[0].get("bars") or []) != wanted:
                raise StreamError(
                    f"subscribed to bars of {reply[0].get('bars')}, not {wanted}"
                )
        async for raw in socket:
            for message in _list(_json(raw)):
                kind = message.get("T")
                if kind == "b":
                    yield _bar(message)
                elif kind == "error":
                    raise StreamError(f"{feed} bars: {_error(message)}")
                elif kind not in ("success", "subscription"):
                    raise StreamError(
                        f"an unexpected message on {feed} bars: {message}"
                    )


async def _messages(socket) -> list[dict]:
    return _list(_json(await socket.recv()))


def _list(messages) -> list[dict]:
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise StreamError(f"market data comes as a list of messages, got {messages!r}")
    return messages


async def _control(socket, expected: str) -> None:
    """One handshake answer, which must be [{"T": "success", "msg": expected}]."""
    reply = await _messages(socket)
    if reply != [{"T": "success", "msg": expected}]:
        raise StreamError(f"market data said {_error(reply[0]) if reply else reply}")


def _error(message: dict) -> str:
    # Alpaca's codes: 402 auth failed, 406 connection limit exceeded, 409 the plan
    # does not include the feed...
    if message.get("T") == "error":
        return f"error {message.get('code')} {message.get('msg')}"
    return str(message)


def _bar(message: dict) -> Bar:
    try:
        return Bar(
            ticker=message["S"],
            start=_time(message["t"]),
            seconds=60,
            open=float(message["o"]),
            high=float(message["h"]),
            low=float(message["l"]),
            close=float(message["c"]),
            volume=int(message["v"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StreamError(f"a bar that cannot be read: {message}: {error}") from error
