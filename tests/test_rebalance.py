from datetime import date
from numbers import Integral

import numpy as np
import pytest

from qfin_alphaguard.events import DailyPlan, ExecutionSettings
from qfin_alphaguard.rebalance import trades_at_open

DAY = date(2026, 10, 12)


def plan(weights, volatility=None, **settings):
    """A plan whose stocks move 1% a day unless `volatility` says otherwise."""
    return DailyPlan(
        DAY,
        "test",
        weights,
        ExecutionSettings(**settings),
        {ticker: 0.01 for ticker in weights} | (volatility or {}),
    )


def at_100(*tickers):
    return dict.fromkeys(tickers, 100.0)


# --- The band (decision 0006) ------------------------------------------------
# Unless a test says otherwise the account is worth 100,000 USD, so 300 shares
# at 100 USD are 30% of it.


def test_a_gap_within_the_stocks_band_is_left_alone():
    # 30% held, 31% wanted: a 1-point gap, within 10 * 1% * 31% = 3.1 points.
    assert trades_at_open(plan({"A": 0.31}), {"A": 300}, at_100("A"), 70_000.0) == {}


def test_a_gap_beyond_the_band_is_traded_to_the_target_in_whole_shares():
    # 40% of 100,000 USD buys 404.86 shares at 98.80 USD: 405, so 105 more.
    trades = trades_at_open(plan({"A": 0.40}), {"A": 300}, {"A": 98.8}, 70_360.0)
    assert trades == {"A": 105}


def test_each_stock_has_its_own_band():
    # Both are 4 points below their 20%; CALM's band is 1 point, WILD's 6.
    trades = trades_at_open(
        plan({"CALM": 0.20, "WILD": 0.20}, {"CALM": 0.005, "WILD": 0.03}),
        {"CALM": 160, "WILD": 160},
        at_100("CALM", "WILD"),
        68_000.0,
    )
    assert trades == {"CALM": 40}


def test_the_band_grows_with_the_larger_of_target_and_current_weight():
    # Both gaps are 4 points and both bands 10 * 1.5% * 30% = 4.5 points.
    # With the smaller weight, 26%, the bands would be 3.9 points.
    trades = trades_at_open(
        plan({"DOWN": 0.26, "UP": 0.30}, {"DOWN": 0.015, "UP": 0.015}),
        {"DOWN": 300, "UP": 260},
        at_100("DOWN", "UP"),
        44_000.0,
    )
    assert trades == {}


def test_band_k_comes_from_the_plan():
    # A 2-point gap: within a 10-day band (3.2 points), traded with none.
    assert trades_at_open(plan({"A": 0.32}), {"A": 300}, at_100("A"), 70_000.0) == {}
    no_band = plan({"A": 0.32}, band_k=0.0)
    assert trades_at_open(no_band, {"A": 300}, at_100("A"), 70_000.0) == {"A": 20}


# --- The minimum trade, and stocks entering and leaving -----------------------


def test_a_trade_worth_less_than_the_minimum_waits():
    # A 20,000 USD account: 6% held, 10% wanted, so 8 shares, 800 USD.
    calm = {"A": 0.001}
    assert (
        trades_at_open(plan({"A": 0.10}, calm), {"A": 12}, at_100("A"), 18_800.0) == {}
    )
    lower = plan({"A": 0.10}, calm, min_trade_usd=500.0)
    assert trades_at_open(lower, {"A": 12}, at_100("A"), 18_800.0) == {"A": 8}


def test_a_new_stock_is_bought():
    trades = trades_at_open(plan({"NEW": 0.05}), {}, {"NEW": 50.0}, 100_000.0)
    assert trades == {"NEW": 100}


def test_a_stock_the_plan_dropped_is_sold_completely_even_below_the_minimum():
    # OLD is worth 500 USD; A is exactly at its target.
    trades = trades_at_open(
        plan({"A": 0.5}), {"A": 500, "OLD": 5}, at_100("A", "OLD"), 49_500.0
    )
    assert trades == {"OLD": -5}


# --- Paying for the buys ------------------------------------------------------


def test_sales_pay_for_buys_on_the_same_day():
    trades = trades_at_open(plan({"B": 0.9}), {"A": 100}, at_100("A", "B"), 0.0)
    assert trades == {"A": -100, "B": 90}


def test_buys_are_cut_by_the_same_factor_when_cash_is_short():
    # A is 10 points over its target but within its band (18 points), so it
    # is not sold. Cash and the sale of OLD bring 40,000 USD for 50,000 USD of
    # buys: each buy gets 80%, and the sale itself is not cut.
    trades = trades_at_open(
        plan({"A": 0.50, "B": 0.25, "C": 0.25}, {"A": 0.03}),
        {"A": 600, "OLD": 50},
        at_100("A", "B", "C", "OLD"),
        35_000.0,
    )
    assert trades == {"OLD": -50, "B": 200, "C": 200}


def test_a_buy_the_cut_takes_below_the_minimum_waits():
    # 40,000 USD for 46,100 USD of buys: B's 450 shares become 390, and D's
    # 11 become 9, worth 900 USD.
    trades = trades_at_open(
        plan({"A": 0.50, "B": 0.45, "D": 0.011}, {"A": 0.03}),
        {"A": 600},
        at_100("A", "B", "D"),
        40_000.0,
    )
    assert trades == {"B": 390}


# --- What it needs ------------------------------------------------------------


def test_every_stock_in_the_plan_or_held_needs_a_price():
    with pytest.raises(ValueError, match="no price for B"):
        trades_at_open(plan({"A": 0.5}), {"B": 10}, at_100("A"), 1_000.0)


def test_a_plan_without_volatility_has_no_band():
    bare = DailyPlan(DAY, "test", {"A": 0.5})
    with pytest.raises(ValueError, match="volatility"):
        trades_at_open(bare, {}, at_100("A"), 10_000.0)


def test_an_empty_account_cannot_trade():
    with pytest.raises(ValueError, match="equity"):
        trades_at_open(plan({"A": 0.5}), {}, at_100("A"), 0.0)


# --- Rules that hold for every account ----------------------------------------


def test_no_short_no_borrowing_no_small_trades_and_every_trade_towards_the_plan():
    rng = np.random.default_rng(7)
    tickers = ["A", "B", "C", "D", "E", "F"]
    for _ in range(300):
        names = [ticker for ticker in tickers if rng.random() < 0.6] or ["A"]
        sizes = rng.dirichlet(np.ones(len(names))) * rng.uniform(0.7, 1.0)
        weights = dict(zip(names, map(float, sizes), strict=True))
        volatility = {ticker: float(rng.uniform(0.005, 0.04)) for ticker in names}
        shares = {t: int(rng.integers(0, 300)) for t in tickers if rng.random() < 0.6}
        prices = {ticker: float(rng.uniform(20, 500)) for ticker in tickers}
        cash = float(rng.uniform(0, 50_000))
        band_k = float(rng.choice([0.0, 5.0, 10.0]))

        trades = trades_at_open(
            plan(weights, volatility, band_k=band_k), shares, prices, cash
        )

        equity = cash + sum(n * prices[ticker] for ticker, n in shares.items())
        sold = sum(-n * prices[ticker] for ticker, n in trades.items() if n < 0)
        bought = sum(n * prices[ticker] for ticker, n in trades.items() if n > 0)
        assert bought <= cash + sold + 1e-6  # no borrowing
        for ticker, held in shares.items():
            if held and ticker not in weights:
                assert trades.get(ticker) == -held  # a dropped stock is sold
        for ticker, n in trades.items():
            assert isinstance(n, Integral) and n != 0
            held = shares.get(ticker, 0)
            assert held + n >= 0  # no short selling
            if ticker in weights:
                assert abs(n) * prices[ticker] >= 1_000.0  # no small trades
                goal = round(weights[ticker] * equity / prices[ticker])
                assert min(held, goal) <= held + n <= max(held, goal)  # towards it
