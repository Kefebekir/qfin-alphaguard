from datetime import date

import polars as pl
import pytest

from qfin_alphaguard.data.corporate import corporate_actions
from qfin_alphaguard.events import Dividend, Split

DAYS = [date(2025, 11, 13), date(2025, 11, 14), date(2025, 11, 17)]


def daily(ticker, closes, adjustments, split_factors):
    n = len(closes)
    return pl.DataFrame(
        {
            "date": DAYS[:n],
            "ticker": [ticker] * n,
            "close": closes,
            "adjustment": adjustments,
            "split_factor": split_factors,
        }
    )


def test_a_fall_in_the_split_factor_is_a_split():
    # Netflix: adjusted closes 115.4 and 111.2 were 1,154 and 1,112 as traded.
    bars = daily("NFLX", [115.423, 111.217, 110.29], [0.1, 0.1, 1.0], [10.0, 10.0, 1.0])
    assert corporate_actions(bars) == [Split("NFLX", date(2025, 11, 17), 10.0)]


def test_a_rise_in_the_dividend_factor_is_a_dividend_of_that_share_of_the_close():
    # Traded at 200 the day before the ex-date; the factor rises by 0.125%.
    bars = daily("AAPL", [199.75, 199.75, 201.0], [0.99875, 0.99875, 1.0], [1.0] * 3)
    (dividend,) = corporate_actions(bars)
    assert isinstance(dividend, Dividend)
    assert dividend.day == date(2025, 11, 17)
    assert dividend.amount == pytest.approx(0.25)


def test_rounding_jitter_in_the_adjustment_is_not_a_dividend():
    bars = daily("AAPL", [200.0, 200.0, 200.0], [0.999999, 0.9999995, 1.0], [1.0] * 3)
    assert corporate_actions(bars) == []


def test_a_fall_in_the_dividend_factor_stops_instead_of_guessing():
    bars = daily("VTR", [50.0, 50.0, 50.0], [1.0, 1.0, 0.86], [1.0] * 3)
    with pytest.raises(ValueError, match="neither split nor dividend"):
        corporate_actions(bars)


def test_only_the_tickers_asked_for_are_derived():
    nflx = daily("NFLX", [115.423, 111.217, 110.29], [0.1, 0.1, 1.0], [10.0, 10.0, 1.0])
    odd = daily("VTR", [50.0, 50.0, 50.0], [1.0, 1.0, 0.86], [1.0] * 3)
    actions = corporate_actions(pl.concat([nflx, odd]), tickers=["NFLX"])
    assert [action.ticker for action in actions] == ["NFLX"]
