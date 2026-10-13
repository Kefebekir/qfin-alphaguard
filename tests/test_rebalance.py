from datetime import date

import numpy as np
import pytest

from qfin_alphaguard.events import DailyPlan, ExecutionSettings, round_shares
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
# at 100 USD are 30% of it. Shares are fractional, to 9 decimals (decision 0012).


def test_a_gap_within_the_stocks_band_is_left_alone():
    # 30% held, 31% wanted: a 1-point gap, within 10 * 1% * 31% = 3.1 points.
    assert trades_at_open(plan({"A": 0.31}), {"A": 300}, at_100("A"), 70_000.0) == {}


def test_a_stock_below_its_band_is_bought_up_to_the_bands_edge():
    # 29.52% held, 40% wanted; the band is 10 * 1% * 40% = 4 points, so the
    # edge is 36%: 365.853658537 shares at 98.40 USD, and 65.853658537 more.
    # Going all the way to the target would have bought 106.5 shares.
    trades = trades_at_open(plan({"A": 0.40}), {"A": 300}, {"A": 98.4}, 70_480.0)
    assert trades == {"A": 65.853658537}


def test_a_stock_above_its_band_is_sold_down_to_the_bands_edge():
    # 30% held, 20% wanted; the band is 10 * 1% * 30% = 3 points, so the edge
    # is 23%: 230 shares, and 70 to sell. Going to the target would sell 100.
    assert trades_at_open(plan({"A": 0.20}), {"A": 300}, at_100("A"), 70_000.0) == {
        "A": -70
    }


def test_each_stock_has_its_own_band():
    # Both are 4 points below their 20%; CALM's band is 1 point, WILD's 6.
    # CALM is bought up to its edge, 19%.
    trades = trades_at_open(
        plan({"CALM": 0.20, "WILD": 0.20}, {"CALM": 0.005, "WILD": 0.03}),
        {"CALM": 160, "WILD": 160},
        at_100("CALM", "WILD"),
        68_000.0,
    )
    assert trades == {"CALM": 30}


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
    # A 2-point gap: within a 10-day band (3.2 points), traded with none, and
    # then all the way to the target, which is the band's edge.
    assert trades_at_open(plan({"A": 0.32}), {"A": 300}, at_100("A"), 70_000.0) == {}
    no_band = plan({"A": 0.32}, band_k=0.0)
    assert trades_at_open(no_band, {"A": 300}, at_100("A"), 70_000.0) == {"A": 20}


def test_a_fractional_holding_is_traded_to_the_edge_in_fractions():
    # 0.5 shares held, 10% of 1,000 USD wanted at 100 USD with no band: 1 share,
    # so 0.5 more, 50 USD.
    trades = trades_at_open(
        plan({"A": 0.10}, band_k=0.0), {"A": 0.5}, at_100("A"), 950.0
    )
    assert trades == {"A": 0.5}


# --- The minimum trade, and stocks entering and leaving -----------------------


def test_the_default_minimum_is_alpacas_1_usd():
    # No band: 10% of 10,000 USD wanted at 100 USD a share, so 10 shares.
    wants = plan({"A": 0.10}, band_k=0.0)
    assert wants.execution.min_trade_usd == 1.0
    # 9.995 shares held: 0.005 more would cost 0.50 USD, so it waits.
    assert trades_at_open(wants, {"A": 9.995}, at_100("A"), 9_000.5) == {}
    # 9.98 shares held: 0.02 more cost 2 USD.
    assert trades_at_open(wants, {"A": 9.98}, at_100("A"), 9_002.0) == {"A": 0.02}


def test_a_trade_worth_less_than_the_minimum_waits():
    # A 20,000 USD account: 6% held, 10% wanted, and a band of 0.1 point, so
    # 9.9%: 19.8 shares, 7.8 more, 780 USD.
    calm = {"A": 0.001}
    high = plan({"A": 0.10}, calm, min_trade_usd=1_000.0)
    assert trades_at_open(high, {"A": 12}, at_100("A"), 18_800.0) == {}
    lower = plan({"A": 0.10}, calm, min_trade_usd=500.0)
    assert trades_at_open(lower, {"A": 12}, at_100("A"), 18_800.0) == {"A": 7.8}


def test_a_new_stock_is_bought_up_to_its_bands_edge():
    # 5% wanted, a band of 10 * 1% * 5% = 0.5 points: 4.5%, 90 shares at 50 USD.
    trades = trades_at_open(plan({"NEW": 0.05}), {}, {"NEW": 50.0}, 100_000.0)
    assert trades == {"NEW": 90}


def test_a_stock_the_plan_dropped_is_sold_completely_even_below_the_minimum():
    # OLD is 5.25 shares, worth 525 USD; A is exactly at its target.
    trades = trades_at_open(
        plan({"A": 0.5}, min_trade_usd=1_000.0),
        {"A": 500, "OLD": 5.25},
        at_100("A", "OLD"),
        49_475.0,
    )
    assert trades == {"OLD": -5.25}


# --- Paying for the buys ------------------------------------------------------


def test_sales_pay_for_buys_on_the_same_day():
    # B's edge is 90% minus a 9-point band: 81% of 10,000 USD.
    trades = trades_at_open(plan({"B": 0.9}), {"A": 100}, at_100("A", "B"), 0.0)
    assert trades == {"A": -100, "B": 81}


def test_buys_are_cut_by_the_same_factor_when_cash_is_short():
    # A is 13 points over its target but within its band (19 points), so it
    # is not sold. B and C want 213.8625 shares each (22.5% of 95,050 USD),
    # 42,772.50 USD; cash and the sale of OLD bring 35,050 USD, so each buy
    # gets 81.9% of it: 175.25 shares, all the money pays for. The sale itself
    # is not cut.
    trades = trades_at_open(
        plan({"A": 0.50, "B": 0.25, "C": 0.25}, {"A": 0.03}),
        {"A": 600, "OLD": 50},
        at_100("A", "B", "C", "OLD"),
        30_050.0,
    )
    assert trades == {"OLD": -50, "B": 175.25, "C": 175.25}


def test_a_buy_the_cut_takes_below_the_minimum_waits():
    # B's edge is 40.5% and D's 1.01898%: 405 and 10.1898 shares, so D's buy
    # of 1,018.98 USD passes the 1,000 USD minimum set here. Together they
    # cost 41,518.98 USD for 40,000 USD of cash, so each gets 96.3% of its
    # buy: B 390.182995825 shares, and D 981.70 USD, now under the minimum.
    trades = trades_at_open(
        plan(
            {"A": 0.50, "B": 0.45, "D": 0.0102},
            {"A": 0.03, "D": 0.0001},
            min_trade_usd=1_000.0,
        ),
        {"A": 600},
        at_100("A", "B", "D"),
        40_000.0,
    )
    assert trades == {"B": 390.182995825}


def test_cash_below_zero_buys_nothing_and_never_fails():
    # Costs can take the cash a little below zero. With nothing to buy there
    # is nothing to cut; a buy wanted (B's edge, 42.705 shares) is not made.
    assert trades_at_open(plan({"A": 1.0}), {"A": 500}, at_100("A"), -10.0) == {}
    wants_b = plan({"A": 0.95, "B": 0.05})
    assert trades_at_open(wants_b, {"A": 950}, at_100("A", "B"), -100.0) == {}


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
        shares = {
            t: round_shares(float(rng.uniform(0, 300)))
            for t in tickers
            if rng.random() < 0.6
        }
        prices = {ticker: float(rng.uniform(20, 500)) for ticker in tickers}
        cash = float(rng.uniform(0, 50_000))
        band_k = float(rng.choice([0.0, 5.0, 10.0]))
        minimum = float(rng.choice([1.0, 100.0, 1_000.0]))

        trades = trades_at_open(
            plan(weights, volatility, band_k=band_k, min_trade_usd=minimum),
            shares,
            prices,
            cash,
        )

        equity = cash + sum(n * prices[ticker] for ticker, n in shares.items())
        sold = sum(-n * prices[ticker] for ticker, n in trades.items() if n < 0)
        bought = sum(n * prices[ticker] for ticker, n in trades.items() if n > 0)
        assert bought <= cash + sold + 1e-6  # no borrowing
        for ticker, held in shares.items():
            if held and ticker not in weights:
                assert trades.get(ticker) == -held  # a dropped stock is sold
        for ticker, n in trades.items():
            assert n != 0 and round_shares(n) == n  # an order can take it
            held = shares.get(ticker, 0)
            assert held + n >= -1e-9  # no short selling
            if ticker in weights:
                assert abs(n) * prices[ticker] >= minimum  # no small trades
                goal = weights[ticker] * equity / prices[ticker]
                low, high = min(held, goal), max(held, goal)
                assert low - 1e-9 <= held + n <= high + 1e-9  # towards the plan
