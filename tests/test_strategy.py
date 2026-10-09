from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from qfin_alphaguard.events import Fill, Side
from qfin_alphaguard.sessions import sessions
from qfin_alphaguard.strategy import Strategy, View

T = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)
(SESSION,) = sessions(date(2026, 10, 6), date(2026, 10, 6))


def view(**changes):
    values = {
        "now": T,
        "session": SESSION,
        "cash": 1_000.0,
        "positions": {"A": 5},
        "last_prices": {"A": 100.0},
        "open_orders": [],
    }
    values.update(changes)
    return View(**values)


def test_the_base_strategy_does_nothing():
    strategy = Strategy()
    strategy.on_session_start(view())
    assert tuple(strategy.on_bars({}, view())) == ()
    strategy.on_fill(Fill("bt-0000001", "A", Side.BUY, 1, 100.0, 0.0, T), view())
    strategy.on_session_end(view())


def test_a_view_cannot_be_changed():
    seen = view()
    with pytest.raises(FrozenInstanceError):
        seen.cash = 1e9
    with pytest.raises(TypeError):
        seen.positions["A"] = 1_000
    with pytest.raises(TypeError):
        seen.last_prices["A"] = 0.01
    assert isinstance(seen.open_orders, tuple)


def test_a_view_needs_a_utc_time():
    with pytest.raises(ValueError, match="UTC"):
        view(now=T.astimezone(timezone(timedelta(hours=-4))))
