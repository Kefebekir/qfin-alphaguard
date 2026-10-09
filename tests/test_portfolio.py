from datetime import UTC, datetime

import pytest

from qfin_alphaguard.events import Fill, Side
from qfin_alphaguard.portfolio import Portfolio

T = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)


def fill(side, quantity, price, commission=0.0, ticker="A"):
    return Fill("bt-0000001", ticker, side, quantity, price, commission, T)


def test_a_buy_turns_cash_into_shares_and_pays_commission():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0, commission=0.35))
    assert portfolio.positions == {"A": 5}
    assert portfolio.cash == pytest.approx(1_000.0 - 500.0 - 0.35)


def test_selling_every_share_closes_the_position():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0))
    portfolio.apply(fill(Side.SELL, 5, 110.0, commission=0.35))
    assert portfolio.positions == {}
    assert portfolio.cash == pytest.approx(1_000.0 + 50.0 - 0.35)


def test_the_value_is_cash_plus_each_position_at_its_price():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0))
    portfolio.apply(fill(Side.BUY, 2, 50.0, ticker="B"))
    assert portfolio.value({"A": 110.0, "B": 40.0, "C": 1.0}) == pytest.approx(
        1_000.0 - 600.0 + 550.0 + 80.0
    )


def test_valuing_needs_a_price_for_every_holding():
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.BUY, 5, 100.0))
    with pytest.raises(ValueError, match="no price"):
        portfolio.value({})


def test_positions_cannot_be_changed_from_outside():
    portfolio = Portfolio(1_000.0)
    with pytest.raises(TypeError):
        portfolio.positions["A"] = 100


def test_the_books_record_a_sale_beyond_the_holding_as_short():
    # Guard forbids short selling (allow_short: false); the books only record.
    portfolio = Portfolio(1_000.0)
    portfolio.apply(fill(Side.SELL, 3, 100.0))
    assert portfolio.positions == {"A": -3}


def test_cash_must_be_a_finite_number():
    with pytest.raises(ValueError):
        Portfolio(float("nan"))


def test_held_shares_follow_a_split():
    portfolio = Portfolio(0.0)
    portfolio.apply(fill(Side.BUY, 10, 1_100.0))
    portfolio.split("A", 10.0, price=110.0)
    assert portfolio.positions == {"A": 100}
    assert portfolio.value({"A": 110.0}) == pytest.approx(-11_000.0 + 11_000.0)


def test_a_fraction_of_a_share_left_by_a_split_is_paid_in_cash():
    portfolio = Portfolio(0.0)
    portfolio.apply(fill(Side.BUY, 3, 90.0))
    portfolio.split("A", 1.5, price=60.0)  # 4.5 shares: 4 kept, half a share paid
    assert portfolio.positions == {"A": 4}
    assert portfolio.cash == pytest.approx(-270.0 + 30.0)


def test_a_split_does_not_lose_a_share_to_floating_point_rounding():
    # A 15% stock dividend is a 23-for-20 split; 100 * 1.15 is 114.99999999999999
    # in floating point, which must still make 115 shares, not 114 and some cash.
    portfolio = Portfolio(0.0)
    portfolio.apply(fill(Side.BUY, 100, 23.0))
    portfolio.split("A", 1.15, price=20.0)
    assert portfolio.positions == {"A": 115}
    assert portfolio.cash == pytest.approx(-2_300.0)


def test_a_reverse_split_keeps_whole_shares():
    portfolio = Portfolio(0.0)
    portfolio.apply(fill(Side.BUY, 3, 10.0))
    portfolio.split("A", 1 / 3, price=30.0)
    assert portfolio.positions == {"A": 1}
    assert portfolio.cash == pytest.approx(-30.0)


def test_a_split_of_a_stock_not_held_changes_nothing():
    portfolio = Portfolio(100.0)
    portfolio.split("A", 10.0, price=1.0)
    assert (portfolio.positions, portfolio.cash) == ({}, 100.0)


def test_a_dividend_is_paid_per_share_held_and_charged_to_a_short():
    portfolio = Portfolio(0.0)
    portfolio.apply(fill(Side.BUY, 10, 100.0))
    portfolio.apply(fill(Side.SELL, 4, 50.0, ticker="B"))
    portfolio.dividend("A", 0.25)
    portfolio.dividend("B", 0.50)
    assert portfolio.cash == pytest.approx(-1_000.0 + 200.0 + 2.5 - 2.0)
