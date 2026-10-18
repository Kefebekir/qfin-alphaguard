import math
import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.events import Bar, OrderIntent, Side
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.live.warmup import (
    WARM_UP_SESSIONS,
    history,
    sessions_before,
    warm_up,
)
from qfin_alphaguard.momentum import NoiseAreaMomentum
from qfin_alphaguard.sessions import sessions
from qfin_alphaguard.strategy import Strategy

QQQ_GUARD = load_guard_config(Path(__file__).parents[1] / "guard_qqq.yaml")
MONDAY = date(2026, 10, 19)


def random_sessions(first, last, seed=7):
    """Each session's minutes as a random walk, with a gap now and then as on IEX."""
    rng = random.Random(seed)
    price = 600.0
    out = []
    for session in sessions(first, last):
        bars = []
        minutes = int((session.close - session.open).total_seconds() // 60)
        price *= 1 + rng.gauss(0, 0.01)  # the overnight gap
        for minute in range(minutes):
            step = price * rng.gauss(0, 0.0008)
            open_, close = price, price + step
            if rng.random() < 0.05:
                price = close
                continue  # no trade on the feed this minute
            high, low = max(open_, close) + 0.05, min(open_, close) - 0.05
            start = session.open + timedelta(minutes=minute)
            bars.append(
                Bar("QQQ", start, 60, open_, high, low, close, rng.randint(200, 5000))
            )
            price = close
        out.append((session, bars))
    return out


def learned(strategy):
    """What the strategy keeps from past sessions; a minute before the first bar is NaN,
    written as None so that it equals itself."""
    moves = [[None if math.isnan(m) else m for m in day] for day in strategy.moves]
    return moves, list(strategy.closes)


def test_a_warmed_strategy_has_learned_what_the_backtest_s_had():
    past = random_sessions(date(2026, 8, 3), date(2026, 9, 30))
    assert len(past) > 20
    backtested = NoiseAreaMomentum()
    run_backtest(
        [bar for _, bars in past for bar in bars],
        backtested,
        NextBarBroker(CostModel()),
        Guard(QQQ_GUARD),
        30_000.0,
    )
    warmed = NoiseAreaMomentum()
    warm_up(warmed, past[-WARM_UP_SESSIONS:], 30_000.0)
    assert learned(warmed) == learned(backtested)
    assert len(warmed.closes) == warmed.closes.maxlen  # ready to trade


def test_too_few_sessions_leave_the_strategy_unready():
    warmed = NoiseAreaMomentum()
    warm_up(warmed, random_sessions(date(2026, 9, 1), date(2026, 9, 10)), 30_000.0)
    assert len(warmed.closes) < warmed.closes.maxlen


class Greedy(Strategy):
    """Wants to buy on every bar; a warm-up must send nothing."""

    def __init__(self):
        self.calls = []

    def on_session_start(self, view):
        self.calls.append(("start", view.now, view.cash, dict(view.positions)))

    def on_bars(self, bars, view):
        self.calls.append(("bars", view.now, list(bars), view.last_prices["QQQ"]))
        return [OrderIntent("QQQ", Side.BUY, 1.0, 700.0, view.now)]

    def on_session_end(self, view):
        self.calls.append(("end", view.now))


def test_the_strategy_is_called_as_the_backtest_s_engine_calls_it():
    (session, bars), *_ = random_sessions(date(2026, 10, 1), date(2026, 10, 1))
    strategy = Greedy()
    warm_up(strategy, [(session, bars[:2])], 30_000.0)
    assert strategy.calls == [
        ("start", session.open, 30_000.0, {}),
        ("bars", bars[0].end, ["QQQ"], bars[0].close),
        ("bars", bars[1].end, ["QQQ"], bars[1].close),
        ("end", session.close),
    ]


def test_a_session_without_bars_is_skipped():
    strategy = Greedy()
    (session, _), *_ = random_sessions(date(2026, 10, 1), date(2026, 10, 1))
    warm_up(strategy, [(session, [])], 30_000.0)
    assert strategy.calls == []


def test_a_bar_outside_its_session_is_refused():
    (_, bars), (other, _) = random_sessions(date(2026, 10, 1), date(2026, 10, 2))
    with pytest.raises(ValueError, match="outside"):
        warm_up(Greedy(), [(other, bars[:1])], 30_000.0)


def test_the_sessions_before_a_day_skip_weekends_and_holidays():
    past = sessions_before(date(2026, 11, 27))  # the day after Thanksgiving
    assert len(past) == WARM_UP_SESSIONS
    days = [s.day for s in past]
    assert days == sorted(days) and days[-1] == date(2026, 11, 25)
    assert date(2026, 11, 26) not in days  # Thanksgiving
    assert all(d.weekday() < 5 for d in days)


class FakeClient:
    def __init__(self, frame):
        self.frame = frame
        self.asked = []

    def minute_bars(self, ticker, first, last, feed="sip"):
        self.asked.append((ticker, first, last, feed))
        return self.frame


def minute_rows(*starts):
    return pl.DataFrame(
        {
            "start": list(starts),
            "ticker": ["QQQ"] * len(starts),
            "open": [600.0] * len(starts),
            "high": [601.0] * len(starts),
            "low": [599.0] * len(starts),
            "close": [600.5] * len(starts),
            "volume": [100] * len(starts),
        },
        schema=MINUTE_SCHEMA,
    )


def test_history_is_iex_s_minutes_of_the_sessions_before_by_session():
    past = sessions_before(MONDAY)
    friday, thursday = past[-1], past[-2]
    frame = minute_rows(
        friday.open + timedelta(minutes=1),
        thursday.open - timedelta(minutes=5),  # before the open: left out
        thursday.open,
        friday.open,
        friday.close,  # after the close: left out
    )
    client = FakeClient(frame)
    got = history(client, "QQQ", MONDAY)
    assert client.asked == [("QQQ", past[0].day, friday.day, "iex")]
    assert [(s.day, [b.start for b in bars]) for s, bars in got] == [
        (thursday.day, [thursday.open]),
        (friday.day, [friday.open, friday.open + timedelta(minutes=1)]),
    ]
    bar = got[0][1][0]
    assert (bar.seconds, bar.close, bar.volume, bar.start.utcoffset()) == (
        60,
        600.5,
        100,
        timedelta(0),
    )


def test_history_of_a_feed_with_nothing_is_empty():
    assert history(FakeClient(pl.DataFrame(schema=MINUTE_SCHEMA)), "QQQ", MONDAY) == []


def test_a_day_too_early_for_the_calendar_is_refused():
    with pytest.raises(ValueError):
        sessions_before(date(2000, 1, 3))
