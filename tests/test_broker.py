from datetime import UTC, datetime, timedelta

import pytest

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import run_backtest
from qfin_alphaguard.events import Bar, Order, OrderIntent, Side
from qfin_alphaguard.guard.check import GuardDecision
from qfin_alphaguard.strategy import Strategy

OPEN = datetime(2026, 10, 6, 13, 30, tzinfo=UTC)
# Only the half spread, so the price rules can be read without fees.
PRICES_ONLY = CostModel(sec_fee_rate=0, cat_per_share=0, half_spread_bps=1.0)
# Alpaca's fees, and no half spread, so the fills are at round prices.
FEES_ONLY = CostModel(half_spread_bps=0)


def at(minute, day=0):
    return OPEN + timedelta(days=day, minutes=minute)


def order(side, quantity, limit, sent=0, ticker="A", number=1, day=0):
    intent = OrderIntent(ticker, side, quantity, limit, at(sent, day))
    return Order(f"bt-{number:07d}", intent, at(sent, day))


def bar(minute, open, high, low, close=None, volume=100_000, ticker="A", day=0):
    close = open if close is None else close
    return Bar(ticker, at(minute, day), 60, open, high, low, close, volume)


def broker(*orders, costs=PRICES_ONLY, participation=0.1):
    simulated = NextBarBroker(costs, participation)
    for waiting in orders:
        simulated.submit(waiting)
    return simulated


# --- Price ------------------------------------------------------------------


def test_a_buy_that_is_marketable_at_the_open_pays_half_the_spread():
    (fill,) = broker(order(Side.BUY, 10, 101.0)).on_bar(bar(0, 100.0, 100.5, 99.5))
    assert fill.price == pytest.approx(100.01)  # 100 plus 1 basis point
    assert fill.time == at(0)  # it traded on arrival, at the bar's start


def test_half_the_spread_never_lifts_a_buy_above_the_bars_high_or_its_limit():
    (above_high,) = broker(order(Side.BUY, 10, 101.0)).on_bar(
        bar(0, 100.0, 100.005, 99.5)
    )
    (above_limit,) = broker(order(Side.BUY, 10, 100.004)).on_bar(
        bar(0, 100.0, 100.5, 99.5)
    )
    assert above_high.price == pytest.approx(100.005)
    assert above_limit.price == pytest.approx(100.004)


def test_a_buy_fills_at_its_limit_when_the_price_falls_below_it():
    (fill,) = broker(order(Side.BUY, 10, 99.0)).on_bar(bar(0, 100.0, 100.5, 98.5))
    assert fill.price == 99.0
    assert fill.time == at(1)  # sometime during the bar: its end


def test_a_buy_does_not_fill_when_the_price_only_touches_its_limit():
    # Other orders were waiting at 99 before ours.
    assert broker(order(Side.BUY, 10, 99.0)).on_bar(bar(0, 100.0, 100.5, 99.0)) == []


def test_a_buy_waits_until_the_price_comes_down():
    simulated = broker(order(Side.BUY, 10, 99.0))
    assert simulated.on_bar(bar(0, 100.0, 100.5, 99.5)) == []
    (fill,) = simulated.on_bar(bar(1, 99.6, 99.7, 98.0))
    assert (fill.price, fill.time) == (99.0, at(2))


def test_a_sell_is_the_mirror_image():
    (marketable,) = broker(order(Side.SELL, 10, 99.0)).on_bar(
        bar(0, 100.0, 100.5, 99.5)
    )
    (through,) = broker(order(Side.SELL, 10, 101.0)).on_bar(bar(0, 100.0, 101.5, 99.5))
    touched = broker(order(Side.SELL, 10, 101.0)).on_bar(bar(0, 100.0, 101.0, 99.5))
    assert marketable.price == pytest.approx(99.99)  # 100 minus 1 basis point
    assert (through.price, through.time) == (101.0, at(1))
    assert touched == []


def test_half_the_spread_never_pushes_a_sell_below_the_bars_low_or_its_limit():
    (below_low,) = broker(order(Side.SELL, 10, 99.0)).on_bar(
        bar(0, 100.0, 100.5, 99.995)
    )
    (below_limit,) = broker(order(Side.SELL, 10, 99.996)).on_bar(
        bar(0, 100.0, 100.5, 99.5)
    )
    assert below_low.price == pytest.approx(99.995)
    assert below_limit.price == pytest.approx(99.996)


# --- Which orders, and how many shares --------------------------------------


def test_an_order_sent_after_the_bar_started_waits_for_the_next_bar():
    late = order(Side.BUY, 10, 101.0, sent=1)
    assert broker(late).on_bar(bar(0, 100.0, 100.5, 99.5)) == []


def test_a_bar_fills_only_orders_in_its_own_stock():
    other = order(Side.BUY, 10, 101.0, ticker="B")
    assert broker(other).on_bar(bar(0, 100.0, 100.5, 99.5)) == []


def test_at_most_a_tenth_of_the_bars_volume_fills_and_the_rest_waits():
    simulated = broker(order(Side.BUY, 12, 101.0))
    shares = [
        sum(
            fill.quantity
            for fill in simulated.on_bar(bar(minute, 100.0, 100.5, 99.5, volume=50))
        )
        for minute in range(4)
    ]
    assert shares == [5, 5, 2, 0]
    assert simulated.waiting == {}  # filled completely: it stops waiting


def test_a_tenth_of_the_volume_fills_even_when_it_is_less_than_a_share():
    simulated = broker(order(Side.BUY, 10, 101.0))
    (fill,) = simulated.on_bar(bar(0, 100.0, 100.5, 99.5, volume=9))
    assert fill.quantity == 0.9


def test_nothing_fills_on_a_bar_without_volume():
    simulated = broker(order(Side.BUY, 10, 101.0))
    assert simulated.on_bar(bar(0, 100.0, 100.5, 99.5, volume=0)) == []


def test_a_fractional_order_fills_whole_and_stops_waiting():
    simulated = broker(order(Side.BUY, 0.37, 101.0))
    (fill,) = simulated.on_bar(bar(0, 100.0, 100.5, 99.5))
    assert fill.quantity == 0.37
    assert simulated.waiting == {}


def test_fractional_parts_add_up_to_the_order_exactly():
    # 0.3 + 0.3 + 0.3 is 0.8999999999999999 in floating point; rounded, the
    # last part is 0.1, not 0.10000000000000009, which no fill could carry.
    simulated = broker(order(Side.BUY, 1, 101.0))
    shares = [
        fill.quantity
        for minute in range(4)
        for fill in simulated.on_bar(bar(minute, 100.0, 100.5, 99.5, volume=3))
    ]
    assert shares == [0.3, 0.3, 0.3, 0.1]
    assert simulated.waiting == {}


def test_older_orders_fill_first():
    first = order(Side.BUY, 10, 101.0, number=1)
    second = order(Side.BUY, 10, 101.0, number=2)
    fills = broker(first, second).on_bar(bar(0, 100.0, 100.5, 99.5))
    assert [fill.client_order_id for fill in fills] == ["bt-0000001", "bt-0000002"]


def test_a_cancelled_order_never_fills():
    simulated = broker(order(Side.BUY, 10, 101.0))
    simulated.cancel("bt-0000001", at(0))
    assert simulated.on_bar(bar(0, 100.0, 100.5, 99.5)) == []


# --- Costs (Alpaca, decision 0012) -------------------------------------------


def test_a_buy_owes_only_the_cat_fee():
    fees = CostModel().fees(Side.BUY, 10, 100.0)
    assert fees == pytest.approx({"SEC": 0.0, "CAT": 10 * 0.000003})


def test_a_sale_owes_the_sec_fee_on_its_value_and_the_cat_fee():
    fees = CostModel().fees(Side.SELL, 10, 100.0)
    assert fees == pytest.approx({"SEC": 1_000 * 0.0000206, "CAT": 10 * 0.000003})


def test_fees_are_owed_on_the_exact_fractional_quantity():
    fees = CostModel().fees(Side.SELL, 0.5, 100.0)
    assert fees == pytest.approx({"SEC": 50 * 0.0000206, "CAT": 0.5 * 0.000003})


def test_a_buy_pays_a_cent_of_cat_fee_and_nothing_else():
    (fill,) = broker(order(Side.BUY, 10, 101.0), costs=FEES_ONLY).on_bar(
        bar(0, 100.0, 100.5, 99.5)
    )
    assert fill.commission == pytest.approx(0.01)


def test_the_first_fill_of_a_day_pays_each_fee_rounded_up_to_the_cent():
    # SEC fee 0.0206 USD, up to 0.03; CAT fee 0.00003 USD, up to 0.01.
    (fill,) = broker(order(Side.SELL, 10, 99.0), costs=FEES_ONLY).on_bar(
        bar(0, 100.0, 100.5, 99.5)
    )
    assert fill.commission == pytest.approx(0.04)


def test_later_fills_that_day_pay_what_takes_a_total_past_another_cent():
    # The day's SEC fees come to 0.0412 USD, charged 0.05; the CAT fees to
    # 0.00006 USD, charged 0.01. The first sale paid 0.03 and 0.01.
    first = order(Side.SELL, 10, 99.0, number=1)
    second = order(Side.SELL, 10, 99.0, number=2)
    fills = broker(first, second, costs=FEES_ONLY).on_bar(bar(0, 100.0, 100.5, 99.5))
    assert [fill.commission for fill in fills] == pytest.approx([0.04, 0.02])


def test_each_day_starts_its_fee_totals_afresh():
    simulated = broker(order(Side.SELL, 10, 99.0), costs=FEES_ONLY)
    simulated.on_bar(bar(0, 100.0, 100.5, 99.5))
    simulated.submit(order(Side.SELL, 10, 99.0, number=2, day=1))
    (next_day,) = simulated.on_bar(bar(0, 100.0, 100.5, 99.5, day=1))
    assert next_day.commission == pytest.approx(0.04)


def test_a_day_total_of_whole_cents_is_not_rounded_up_again():
    # SEC fees of 0.10 and 0.20 USD add up to 0.30000000000000004 in floating
    # point, which must still be charged as 30 cents, not 31.
    costs = CostModel(sec_fee_rate=0.0001, cat_per_share=0, half_spread_bps=0)
    first = order(Side.SELL, 10, 99.0, number=1)  # 1,000 USD sold
    second = order(Side.SELL, 20, 99.0, number=2)  # 2,000 USD sold
    fills = broker(first, second, costs=costs).on_bar(bar(0, 100.0, 100.5, 99.5))
    assert [fill.commission for fill in fills] == pytest.approx([0.10, 0.20])


@pytest.mark.parametrize("name", ["sec_fee_rate", "cat_per_share", "half_spread_bps"])
def test_costs_cannot_be_negative(name):
    with pytest.raises(ValueError, match=name):
        CostModel(**{name: -0.01})


def test_participation_must_be_a_share_of_the_volume():
    with pytest.raises(ValueError):
        NextBarBroker(participation=0)


# --- Together with the engine ------------------------------------------------


class Allow:
    def check(self, intent, view):
        return GuardDecision(intent)


class RoundTrip(Strategy):
    """Buys 10 shares after the first minute and sells them after the third."""

    def on_bars(self, bars, view):
        minute = int((view.now - view.session.open).total_seconds() // 60)
        price = bars["A"].close
        if minute == 1:
            return [OrderIntent("A", Side.BUY, 10, price * 1.005, view.now)]
        if minute == 3:
            return [OrderIntent("A", Side.SELL, 10, price * 0.995, view.now)]
        return ()


def test_the_broker_keeps_the_engines_contract():
    feed = [
        bar(0, 100.0, 100.5, 99.5),
        bar(1, 100.0, 100.5, 99.5),  # the buy fills at its open, plus 1 bp
        bar(2, 101.0, 101.5, 100.5),
        bar(3, 101.0, 101.5, 100.5),  # the sell fills at its open, minus 1 bp
    ]

    result = run_backtest(feed, RoundTrip(), NextBarBroker(), Allow(), cash=10_000.0)

    bought, sold = 10 * 100.0 * 1.0001, 10 * 101.0 * 0.9999
    # The buy pays a cent of CAT fee; the sale 0.0208 USD of SEC fee, charged
    # 0.03, and nothing more of CAT fee, whose day total is still under a cent.
    assert result.positions == {}
    assert result.cash == pytest.approx(10_000.0 - bought - 0.01 + sold - 0.03)
