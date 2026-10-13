from datetime import UTC, date, datetime, timedelta

import pytest

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.events import (
    Bar,
    Cancellation,
    DailyPlan,
    ExecutionSettings,
    Fill,
    Order,
    OrderIntent,
    Side,
)
from qfin_alphaguard.execution import EqualSpaced
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.sessions import sessions
from qfin_alphaguard.strategy import View

DAY = date(2026, 10, 6)  # summer time: the session runs 13:30 to 20:00 UTC
(SESSION,) = sessions(DAY, DAY)
HALF_DAY = date(2025, 11, 28)  # the day after Thanksgiving closes at 13:00
(HALF_SESSION,) = sessions(HALF_DAY, HALF_DAY)
COLLAR = 50  # Guard's price_collar_bps


def ny(hour, minute, session=SESSION):
    """A New York time on the session's day, in UTC."""
    return session.open + timedelta(hours=hour - 9, minutes=minute - 30)


def plan(weights, day=DAY, **settings):
    """A plan without a band, unless `settings` give one: trades go to the target."""
    settings = {"band_k": 0.0} | settings
    volatility = dict.fromkeys(weights, 0.01)
    return DailyPlan(day, "test", weights, ExecutionSettings(**settings), volatility)


def view(now, prices, positions=None, cash=10_000.0, open_orders=(), session=SESSION):
    return View(now, session, cash, positions or {}, prices, open_orders)


def started(day_plan, session=SESSION, collar=COLLAR):
    strategy = EqualSpaced({day_plan.trading_date: day_plan}, collar)
    strategy.on_session_start(view(session.open, {}, session=session))
    return strategy


def every_minute(strategy, prices, first, last, **kwargs):
    """on_bars at each minute from `first` to `last`; the intents, by time."""
    sent = {}
    now = first
    while now <= last:
        for decision in strategy.on_bars({}, view(now, prices, **kwargs)):
            if isinstance(decision, OrderIntent):
                sent.setdefault(now, []).append(decision)
        now += timedelta(minutes=1)
    return sent


# --- When -------------------------------------------------------------------


def test_nothing_is_decided_before_10_30():
    strategy = started(plan({"A": 0.5}))
    sent = every_minute(strategy, {"A": 100.0}, ny(9, 31), ny(10, 29))
    assert sent == {}


def test_three_children_go_at_10_30_13_00_and_15_30():
    # 50% of 10,000 USD at 100 USD: 50 shares, 5,000 USD, so three children.
    strategy = started(plan({"A": 0.5}))
    sent = every_minute(strategy, {"A": 100.0}, ny(9, 31), ny(16, 0))
    assert list(sent) == [ny(10, 30), ny(13, 0), ny(15, 30)]


def test_on_a_half_day_the_children_end_at_12_30():
    day_plan = plan({"A": 0.5}, day=HALF_DAY)
    strategy = started(day_plan, session=HALF_SESSION)
    sent = every_minute(
        strategy,
        {"A": 100.0},
        ny(9, 31, HALF_SESSION),
        ny(13, 0, HALF_SESSION),
        session=HALF_SESSION,
    )
    assert list(sent) == [
        ny(10, 30, HALF_SESSION),
        ny(11, 30, HALF_SESSION),
        ny(12, 30, HALF_SESSION),
    ]


def test_a_missing_price_puts_the_decision_off_to_the_next_bar():
    strategy = started(plan({"A": 0.5}))
    assert list(strategy.on_bars({}, view(ny(10, 30), {}))) == []
    (first,) = strategy.on_bars({}, view(ny(10, 31), {"A": 100.0}))
    assert first.time == ny(10, 31)


def test_a_held_stock_without_a_price_puts_the_decision_off_too():
    # The band values the account with the price of every stock held.
    strategy = started(plan({"A": 0.5}))
    held = {"OLD": 1.0}
    first = view(ny(10, 30), {"A": 100.0}, positions=held)
    later = view(ny(10, 31), {"A": 100.0, "OLD": 100.0}, positions=held)
    assert list(strategy.on_bars({}, first)) == []
    assert {intent.ticker for intent in strategy.on_bars({}, later)} == {"A", "OLD"}


def test_a_plan_the_band_cannot_use_stops_the_run():
    # Only a missing price waits for the next bar; a broken plan is a bug.
    broken = DailyPlan(DAY, "test", {"A": 0.5})  # without volatility
    strategy = started(broken)
    with pytest.raises(ValueError, match="no volatility"):
        strategy.on_bars({}, view(ny(10, 30), {"A": 100.0}))


def test_children_stay_min_gap_min_apart_even_when_the_day_is_short():
    # Decided only at 15:00: 30 minutes to the deadline leave room for two.
    strategy = started(plan({"A": 0.5}))
    sent = every_minute(strategy, {}, ny(10, 30), ny(14, 59))
    assert sent == {}
    sent = every_minute(strategy, {"A": 100.0}, ny(15, 0), ny(16, 0))
    assert list(sent) == [ny(15, 0), ny(15, 30)]


def test_nothing_is_decided_after_the_deadline():
    # No price for A until 15:31: too late to start the day's trades.
    strategy = started(plan({"A": 0.5}))
    assert every_minute(strategy, {}, ny(10, 30), ny(15, 30)) == {}
    assert every_minute(strategy, {"A": 100.0}, ny(15, 31), ny(16, 0)) == {}


def test_a_day_without_a_plan_trades_nothing():
    strategy = EqualSpaced({}, COLLAR)
    strategy.on_session_start(view(SESSION.open, {}))
    assert every_minute(strategy, {"A": 100.0}, ny(9, 31), ny(16, 0)) == {}


def test_a_day_the_band_holds_back_sends_nothing():
    # 0.15 share short of 10%, 15 USD: under the 20 USD a day must be worth.
    strategy = started(plan({"A": 0.10}))
    sent = every_minute(
        strategy,
        {"A": 100.0},
        ny(10, 30),
        ny(16, 0),
        positions={"A": 9.85},
        cash=9_015.0,
    )
    assert sent == {}


# --- How much ---------------------------------------------------------------


def test_each_child_is_what_is_left_over_the_children_left():
    # Nothing fills here, so the 50 shares stay to buy: 50 / 3, then 50 / 2,
    # then all 50. In the engine each unfilled child is cancelled first.
    strategy = started(plan({"A": 0.5}))
    sent = every_minute(strategy, {"A": 100.0}, ny(10, 30), ny(16, 0))
    assert [intent.quantity for (intent,) in sent.values()] == [16.666666667, 25, 50]
    assert [intent.reason for (intent,) in sent.values()] == [
        "child 1 of 3",
        "child 2 of 3",
        "child 3 of 3",
    ]


def test_an_open_child_is_cancelled_and_its_rest_goes_with_the_next():
    strategy = started(plan({"A": 0.5}))
    (first,) = strategy.on_bars({}, view(ny(10, 30), {"A": 100.0}))
    open_child = Order("bt-0000001", first, ny(10, 30))
    # 10 shares of the first child filled; 40 are left for two children.
    strategy.on_fill(
        Fill("bt-0000001", "A", Side.BUY, 10, 100.0, 0.0, ny(10, 31)),
        view(ny(10, 31), {"A": 100.0}),
    )

    decisions = strategy.on_bars(
        {}, view(ny(13, 0), {"A": 100.0}, open_orders=(open_child,))
    )

    cancel, child = decisions
    assert isinstance(cancel, Cancellation)
    assert cancel.client_order_id == "bt-0000001"
    assert child.quantity == 20


def test_a_small_trade_has_fewer_children():
    # 2.50 USD to buy, with the 1 USD minimum: two children, at 10:30 and 15:30.
    day_plan = plan({"A": 0.25}, min_day_usd=0.0)
    strategy = started(day_plan)
    sent = every_minute(
        strategy,
        {"A": 100.0},
        ny(10, 30),
        ny(16, 0),
        positions={"A": 2.475},
        cash=752.5,
    )
    assert list(sent) == [ny(10, 30), ny(15, 30)]
    assert sent[ny(10, 30)][0].quantity == 0.0125


def test_a_buy_child_under_the_minimum_waits_for_the_next():
    # 0.03 share, 3 USD, to buy: three children of 1 USD. The first fills. At
    # 13:00 the price has fallen to 90 USD, so the second child, 0.01 share,
    # would be worth 0.90 USD: it waits, and the last child takes both.
    day_plan = plan({"A": 0.25}, min_day_usd=0.0)
    strategy = started(day_plan)
    held = {"A": 2.97}
    sent = every_minute(
        strategy, {"A": 100.0}, ny(10, 30), ny(10, 30), positions=held, cash=903.0
    )
    assert [intent.quantity for (intent,) in sent.values()] == [0.01]
    strategy.on_fill(
        Fill("bt-0000001", "A", Side.BUY, 0.01, 100.0, 0.0, ny(10, 31)),
        view(ny(10, 31), {"A": 100.0}),
    )
    later = every_minute(strategy, {"A": 90.0}, ny(10, 31), ny(16, 0))
    assert list(later) == [ny(15, 30)]
    assert later[ny(15, 30)][0].quantity == 0.02


def test_a_sale_of_any_size_goes():
    # The plan dropped OLD, worth 0.50 USD: Alpaca has no minimum for sales.
    day_plan = plan({"A": 0.5}, min_day_usd=0.0)
    strategy = started(day_plan)
    decisions = strategy.on_bars(
        {},
        view(
            ny(10, 30),
            {"A": 100.0, "OLD": 100.0},
            positions={"A": 50, "OLD": 0.005},
            cash=4_999.5,
        ),
    )
    (sale,) = decisions
    assert (sale.ticker, sale.side, sale.quantity) == ("OLD", Side.SELL, 0.005)


# --- At what price ------------------------------------------------------------


@pytest.mark.parametrize(
    ("price", "limit"),
    [
        (100.003, 100.50),  # 100.503, rounded down to the cent
        (100.0, 100.50),  # 10049.999999999998 cents in floating point, still 100.50
        (20.0, 20.10),  # 2009.9999999999998 cents, still 20.10
    ],
)
def test_a_buy_is_limited_to_the_last_price_plus_the_collar_on_the_cent(price, limit):
    strategy = started(plan({"A": 0.5}))
    (child,) = strategy.on_bars({}, view(ny(10, 30), {"A": price}))
    assert (child.side, child.limit_price) == (Side.BUY, limit)


def test_a_sale_is_limited_to_the_last_price_minus_the_collar_on_the_cent():
    # 100 shares held, half of them to sell: 50 shares, in three children.
    strategy = started(plan({"A": 0.5}))
    (sale,) = strategy.on_bars(
        {}, view(ny(10, 30), {"A": 100.0}, positions={"A": 100}, cash=0.0)
    )
    # 100 * 0.995 is 99.5; rounded up to the cent, without floating-point
    # crumbs, it stays 99.50, not 99.51.
    assert (sale.side, sale.quantity, sale.limit_price) == (
        Side.SELL,
        16.666666667,
        99.50,
    )


@pytest.mark.parametrize("price", [20.0, 99.99, 100.003, 457.31, 3_766.76])
def test_limits_never_break_the_collar(price):
    strategy = started(plan({"A": 0.5}))
    (child,) = strategy.on_bars({}, view(ny(10, 30), {"A": price}))
    assert price <= child.limit_price <= price * 1.005 + 1e-9
    assert round(child.limit_price, 2) == child.limit_price


@pytest.mark.parametrize("price", [20.0, 99.99, 100.003, 457.31, 3_766.76])
def test_sale_limits_never_break_the_collar(price):
    # 100 shares held, half to sell. At 100.003 USD, 99.502985 rounds up to
    # 99.51: rounded down, 99.50 would be below the collar.
    strategy = started(plan({"A": 0.5}))
    (sale,) = strategy.on_bars(
        {}, view(ny(10, 30), {"A": price}, positions={"A": 100}, cash=0.0)
    )
    assert sale.side is Side.SELL
    assert price * 0.995 - 1e-9 <= sale.limit_price <= price
    assert round(sale.limit_price, 2) == sale.limit_price


# --- In the engine ------------------------------------------------------------


class AllowAll:
    def check(self, intent, view):
        return GuardDecision(intent)


def minute_bars(ticker, session, price=100.0):
    start, bars = session.open, []
    while start < session.close:
        bars.append(
            Bar(ticker, start, 60, price, price + 0.5, price - 0.5, price, 10_000)
        )
        start += timedelta(minutes=1)
    return bars


def test_the_children_fill_in_the_engine_and_reach_the_target():
    day_plan = plan({"A": 0.5})
    strategy = EqualSpaced({DAY: day_plan}, COLLAR)
    broker = NextBarBroker(
        CostModel(sec_fee_rate=0, cat_per_share=0, half_spread_bps=0)
    )

    result = run_backtest(
        minute_bars("A", SESSION), strategy, broker, AllowAll(), cash=10_000.0
    )

    orders = [record for record in result.log if isinstance(record, Order)]
    assert [order.sent_at for order in orders] == [ny(10, 30), ny(13, 0), ny(15, 30)]
    assert sum(order.intent.quantity for order in orders) == pytest.approx(50)
    assert result.positions == {"A": 50}


def test_a_new_day_decides_afresh():
    # Day one buys 50 shares, half the account; day two wants 60%: 10 more.
    second_day = date(2026, 10, 7)
    (second,) = sessions(second_day, second_day)
    plans = {DAY: plan({"A": 0.5}), second_day: plan({"A": 0.6}, day=second_day)}
    broker = NextBarBroker(
        CostModel(sec_fee_rate=0, cat_per_share=0, half_spread_bps=0)
    )

    result = run_backtest(
        minute_bars("A", SESSION) + minute_bars("A", second),
        EqualSpaced(plans, COLLAR),
        broker,
        AllowAll(),
        cash=10_000.0,
    )

    orders = [record for record in result.log if isinstance(record, Order)]
    on_day_two = [
        order.sent_at for order in orders if order.sent_at.date() == second_day
    ]
    assert on_day_two == [ny(10, 30, second), ny(13, 0, second), ny(15, 30, second)]
    assert result.positions == {"A": 60}


def test_times_are_utc():
    assert ny(10, 30) == datetime(2026, 10, 6, 14, 30, tzinfo=UTC)
