from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.events import (
    Bar,
    Order,
    OrderIntent,
    RiskAction,
    RiskEvent,
    Side,
)
from qfin_alphaguard.guard.config import GuardConfig
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.sessions import sessions
from qfin_alphaguard.strategy import Strategy, View

DAY = date(2026, 10, 6)
(SESSION,) = sessions(DAY, DAY)
NEXT_DAY = date(2026, 10, 7)
(NEXT_SESSION,) = sessions(NEXT_DAY, NEXT_DAY)
NOW = SESSION.open + timedelta(hours=1)  # 10:30 New York time

# guard.yaml's limits today, written out so a hand-made change to the file
# cannot change what these tests mean.
LIMITS = GuardConfig(
    allow_short=False,
    max_weight=0.25,
    max_gross_exposure=1.0,
    max_order_usd=2000,
    price_collar_bps=50,
    max_orders_per_min=10,
    max_trades_per_symbol_day=3,
    no_same_day_reversal=True,
    daily_loss_limit_pct=2.0,
    stale_data_sec=15,
    live_trading=False,
)


def view(
    prices=None,
    positions=None,
    cash=10_000.0,
    open_orders=(),
    now=NOW,
    seen=None,
    last_close=None,
    session=SESSION,
):
    """The account at `now`; by default every price was just seen and nothing moved."""
    prices = {"A": 100.0} if prices is None else prices
    positions = positions or {}
    seen = dict.fromkeys(prices, now) if seen is None else seen
    if last_close is None:
        last_close = cash + sum(n * prices[t] for t, n in positions.items())
    return View(now, session, cash, positions, prices, open_orders, seen, last_close)


def buy(quantity=10.0, ticker="A", limit=100.5, now=NOW):
    return OrderIntent(ticker, Side.BUY, quantity, limit, now)


def sell(quantity=10.0, ticker="A", limit=99.5, now=NOW):
    return OrderIntent(ticker, Side.SELL, quantity, limit, now)


def open_order(intent, number=1):
    return Order(f"bt-{number:07d}", intent, intent.time)


def dropped(decision, rule):
    """The decision drops the intent because of `rule`."""
    return (
        decision.intent is None
        and decision.event.action is RiskAction.REJECT
        and decision.event.rule == rule
    )


# --- What goes through -----------------------------------------------------------


def test_an_order_inside_every_limit_goes_out_unchanged():
    intent = buy()
    decision = Guard(LIMITS).check(intent, view())
    assert decision.intent == intent
    assert decision.event is None


def test_an_event_is_dated_and_names_the_rule_and_the_stock():
    decision = Guard(LIMITS).check(buy(limit=102.0), view())
    event = decision.event
    assert (event.time, event.ticker, event.rule) == (NOW, "A", "price_collar_bps")
    assert "200.0 bps" in event.detail


def test_live_trading_changes_no_answer():
    # live_trading is for the live engine; a backtest never reaches a broker.
    live = Guard(replace(LIMITS, live_trading=True))
    assert live.check(buy(), view()).intent == buy()


def test_the_view_must_say_what_the_account_was_worth_at_the_last_close():
    without = View(NOW, SESSION, 10_000.0, {}, {"A": 100.0}, (), {"A": NOW})
    with pytest.raises(ValueError, match="last close"):
        Guard(LIMITS).check(buy(), without)


# --- The daily loss limit ------------------------------------------------------------


def test_a_two_percent_loss_since_the_last_close_stops_new_orders():
    # 100 shares of A were worth 10,000 USD at the last close.
    guard = Guard(LIMITS)
    held = {"A": 100}
    intent = sell(limit=97.6)
    assert guard.check(intent, view({"A": 98.01}, held, 0, last_close=10_000)).intent
    decision = guard.check(intent, view({"A": 97.99}, held, 0, last_close=10_000))
    assert dropped(decision, "daily_loss_limit_pct")
    assert "it is 9,799.00 USD" in decision.event.detail


def test_a_loss_of_exactly_the_limit_stops_new_orders_too():
    # A 25% limit and a fall from 10,000 to 7,500 USD: exact in floating point.
    guard = Guard(replace(LIMITS, daily_loss_limit_pct=25.0))
    down = view({"A": 75.0}, {"A": 100}, 0, last_close=10_000)
    assert dropped(guard.check(sell(limit=74.7), down), "daily_loss_limit_pct")


def test_the_loss_limit_holds_for_the_rest_of_the_day_after_a_recovery():
    guard = Guard(LIMITS)
    held = {"A": 100}
    down = view({"A": 97.0}, held, 0, last_close=10_000)
    assert dropped(guard.check(sell(limit=96.6), down), "daily_loss_limit_pct")
    # Back to a 1% loss two hours later: still nothing goes out today.
    moment = NOW + timedelta(hours=2)
    later = view({"A": 99.0}, held, 0, now=moment, last_close=10_000)
    assert dropped(
        guard.check(sell(limit=98.6, now=moment), later), "daily_loss_limit_pct"
    )
    tomorrow = NEXT_SESSION.open + timedelta(hours=1)
    next_day = view(
        {"A": 100.0}, held, 0, now=tomorrow, last_close=9_900, session=NEXT_SESSION
    )
    assert guard.check(sell(now=tomorrow), next_day).intent == sell(now=tomorrow)


# --- Stale data ---------------------------------------------------------------------


def test_a_price_older_than_15_seconds_is_stale():
    guard = Guard(LIMITS)
    fresh = view(seen={"A": NOW - timedelta(seconds=15)})
    assert guard.check(buy(), fresh).intent == buy()
    old = view(seen={"A": NOW - timedelta(seconds=16)})
    decision = Guard(LIMITS).check(buy(), old)
    assert dropped(decision, "stale_data_sec")
    assert "16 seconds old" in decision.event.detail


def test_a_stock_without_a_price_is_stale():
    decision = Guard(LIMITS).check(buy(ticker="B"), view())
    assert dropped(decision, "stale_data_sec")


# --- The price collar ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("intent", "goes"),
    [
        (buy(limit=100.5), True),  # 50 bps above the last price of 100
        (buy(limit=100.51), False),
        (buy(limit=99.49), False),  # either side: 51 bps below
        (sell(limit=99.5), True),
        (sell(limit=99.49), False),
        (sell(limit=100.51), False),
    ],
)
def test_a_limit_must_lie_within_50_bps_of_the_last_price(intent, goes):
    held = {"A": 10}
    decision = Guard(LIMITS).check(intent, view(positions=held, cash=20_000.0))
    assert (decision.intent == intent) is goes
    if not goes:
        assert dropped(decision, "price_collar_bps")


def test_a_limit_on_the_cent_at_the_collar_edge_is_inside_it():
    # 20 * 1.005 is 20.099999999999998 in floating point; the limit is 20.10.
    intent = buy(limit=20.1)
    assert Guard(LIMITS).check(intent, view({"A": 20.0})).intent == intent


# --- How many orders ----------------------------------------------------------------


def test_at_most_10_orders_go_out_a_minute():
    guard = Guard(LIMITS)
    tickers = [f"S{number:02d}" for number in range(11)]
    prices = dict.fromkeys(tickers, 100.0)
    answers = [guard.check(buy(1, ticker), view(prices)) for ticker in tickers]
    assert all(answer.intent for answer in answers[:10])
    assert dropped(answers[10], "max_orders_per_min")
    minute_later = NOW + timedelta(minutes=1)
    later = view(prices, now=minute_later)
    assert guard.check(buy(1, "S10", now=minute_later), later).intent


def test_at_most_3_orders_in_a_stock_a_day():
    guard = Guard(LIMITS)
    times = [NOW + timedelta(hours=hours) for hours in range(4)]
    prices = {"A": 100.0, "B": 100.0}
    answers = [guard.check(buy(1, now=t), view(prices, now=t)) for t in times]
    assert all(answer.intent for answer in answers[:3])
    assert dropped(answers[3], "max_trades_per_symbol_day")
    assert guard.check(buy(1, "B", now=times[3]), view(prices, now=times[3])).intent
    tomorrow = NEXT_SESSION.open + timedelta(hours=1)
    next_day = view(prices, now=tomorrow, session=NEXT_SESSION)
    assert guard.check(buy(1, now=tomorrow), next_day).intent


def test_no_order_against_the_day_s_direction_in_a_stock():
    guard = Guard(LIMITS)
    held = {"A": 10}
    assert guard.check(buy(1), view(positions=held)).intent
    later = NOW + timedelta(minutes=5)
    decision = guard.check(sell(1, now=later), view(positions=held, now=later))
    assert dropped(decision, "no_same_day_reversal")
    assert "a buy of A went out today" in decision.event.detail
    allowed = Guard(replace(LIMITS, no_same_day_reversal=False))
    allowed.check(buy(1), view(positions=held))
    assert allowed.check(sell(1, now=later), view(positions=held, now=later)).intent


def test_an_order_guard_drops_is_not_counted():
    guard = Guard(LIMITS)
    for minutes in range(3):
        moment = NOW + timedelta(minutes=minutes)
        guard.check(buy(1, limit=110.0, now=moment), view(now=moment))  # collar
    moment = NOW + timedelta(minutes=3)
    assert guard.check(buy(1, now=moment), view(now=moment)).intent


def test_a_cut_order_is_counted():
    guard = Guard(LIMITS)
    for minutes in range(3):
        moment = NOW + timedelta(minutes=minutes)
        assert guard.check(buy(30, now=moment), view(now=moment)).event  # cut to 2,000
    moment = NOW + timedelta(minutes=3)
    assert dropped(
        guard.check(buy(1, now=moment), view(now=moment)), "max_trades_per_symbol_day"
    )


# --- Cutting the quantity -----------------------------------------------------------


def reduced(decision, rule, quantity):
    """The decision sends the intent with `quantity` shares because of `rule`."""
    return (
        decision.intent is not None
        and decision.intent.quantity == quantity
        and decision.event.action is RiskAction.REDUCE
        and decision.event.rule == rule
    )


def test_an_order_over_2000_usd_is_cut_to_2000_usd():
    # 30 shares at 100.50 are 3,015 USD; 2,000 USD buy 19.900497512 shares.
    decision = Guard(LIMITS).check(buy(30), view(cash=20_000.0))
    assert reduced(decision, "max_order_usd", 19.900497512)
    assert decision.intent.notional <= 2_000
    assert "19.900497512 of the 30 shares go" in decision.event.detail


def test_a_sale_is_cut_to_the_shares_held():
    decision = Guard(LIMITS).check(sell(10), view(positions={"A": 6}))
    assert reduced(decision, "allow_short", 6)


def test_shares_in_open_sales_cannot_be_sold_again():
    selling = open_order(sell(4))
    decision = Guard(LIMITS).check(
        sell(10), view(positions={"A": 6}, open_orders=(selling,))
    )
    assert reduced(decision, "allow_short", 2)


def test_a_sale_of_nothing_held_is_dropped():
    decision = Guard(LIMITS).check(sell(10), view())
    assert dropped(decision, "allow_short")
    assert decision.event.detail.endswith("no shares can go")


def test_short_sales_go_when_guard_yaml_allows_them():
    intent = sell(10)
    assert (
        Guard(replace(LIMITS, allow_short=True)).check(intent, view()).intent == intent
    )


def test_a_buy_is_cut_to_keep_the_stock_at_25_percent():
    # 20 shares of A, 2,000 USD, in an account of 10,000 USD: 500 USD of room.
    decision = Guard(LIMITS).check(buy(10), view(positions={"A": 20}, cash=8_000.0))
    assert reduced(decision, "max_weight", 4.975124378)


def test_open_buys_count_towards_the_weight():
    buying = open_order(buy(20))  # 2,010 USD at its limit
    decision = Guard(LIMITS).check(buy(10), view(open_orders=(buying,)))
    assert reduced(decision, "max_weight", 4.875621890)


def test_a_buy_is_cut_to_the_cash_when_the_account_is_invested():
    # 9,500 USD in B and 500 USD of cash: 500 USD of room.
    prices = {"A": 100.0, "B": 100.0}
    decision = Guard(LIMITS).check(buy(10), view(prices, {"B": 95}, cash=500.0))
    assert reduced(decision, "max_gross_exposure", 4.975124378)


def test_an_open_sale_frees_no_cash_until_it_fills():
    # The account is fully in B; half of B is in an open sale.
    prices = {"A": 100.0, "B": 100.0}
    selling = open_order(sell(50, "B"))
    decision = Guard(LIMITS).check(
        buy(10), view(prices, {"B": 100}, cash=0.0, open_orders=(selling,))
    )
    assert dropped(decision, "max_gross_exposure")


def test_a_buy_is_dropped_when_fees_took_the_cash_below_zero():
    prices = {"A": 100.0, "B": 100.0}
    decision = Guard(LIMITS).check(buy(1), view(prices, {"B": 100}, cash=-1.0))
    assert dropped(decision, "max_gross_exposure")


def test_open_buys_use_up_cash():
    prices = {"A": 100.0, "B": 100.0, "C": 100.0}
    buying = open_order(buy(9, "B"))  # 904.50 USD of the 1,000 USD of cash
    decision = Guard(LIMITS).check(
        buy(10), view(prices, {"C": 90}, cash=1_000.0, open_orders=(buying,))
    )
    assert reduced(decision, "max_gross_exposure", 0.950248756)


def test_the_tightest_rule_sets_the_quantity_and_is_named():
    # 3,015 USD asked: 2,000 USD by order size, 2,500 by weight, 1,000 by cash.
    prices = {"A": 100.0, "B": 100.0}
    decision = Guard(LIMITS).check(buy(30), view(prices, {"B": 90}, cash=1_000.0))
    assert reduced(decision, "max_gross_exposure", 9.950248756)


def test_a_cut_never_breaks_a_whole_share_by_a_floating_point_crumb():
    # 984.90 USD buy exactly 9.8 shares at 100.50, but floating point makes the
    # quotient 9.799999999999999: cut down as it is, 9.799999999 shares.
    guard = Guard(replace(LIMITS, max_order_usd=984.9))
    assert reduced(guard.check(buy(30), view(cash=20_000.0)), "max_order_usd", 9.8)


# --- In the engine ------------------------------------------------------------------


class BuysTwelveAtOnce(Strategy):
    """Buys one share of twelve stocks in the minute that ends at 10:30."""

    def on_bars(self, bars, view):
        if view.now != NOW:
            return ()
        return [OrderIntent(ticker, Side.BUY, 1, 100.5, view.now) for ticker in bars]


class FillsNothing:
    def submit(self, order):
        pass

    def cancel(self, client_order_id, time):
        pass

    def on_bar(self, bar):
        return []


def test_in_the_engine_guard_lets_10_orders_through_a_minute_and_logs_the_rest():
    tickers = [f"S{number:02d}" for number in range(12)]
    start = NOW - timedelta(minutes=1)
    feed = [
        Bar(ticker, start, 60, 100.0, 100.0, 100.0, 100.0, 1000) for ticker in tickers
    ]

    result = run_backtest(
        feed, BuysTwelveAtOnce(), FillsNothing(), Guard(LIMITS), cash=10_000.0
    )

    orders = [record for record in result.log if isinstance(record, Order)]
    events = [record for record in result.log if isinstance(record, RiskEvent)]
    assert [order.intent.ticker for order in orders] == tickers[:10]
    assert [(event.ticker, event.rule) for event in events] == [
        ("S10", "max_orders_per_min"),
        ("S11", "max_orders_per_min"),
    ]


def test_times_are_utc():
    assert NOW == datetime(2026, 10, 6, 14, 30, tzinfo=UTC)
