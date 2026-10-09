import zlib
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.data.universe import (
    MIN_TRADING_DAYS,
    IndexHistory,
    Membership,
    dollar_volume,
    sp500,
    trading_universe,
)


def span(ticker, start, end=None):
    return Membership(
        ticker, date.fromisoformat(start), date.fromisoformat(end) if end else None
    )


def test_a_member_from_start_up_to_but_not_including_end():
    history = IndexHistory((span("AAA", "2020-01-02", "2020-03-02"),))

    assert history.members_on(date(2020, 1, 1)) == frozenset()
    assert history.members_on(date(2020, 1, 2)) == {"AAA"}
    assert history.members_on(date(2020, 3, 1)) == {"AAA"}
    assert history.members_on(date(2020, 3, 2)) == frozenset()


def test_tickers_between_keeps_stocks_that_left_during_the_window():
    history = IndexHistory(
        (
            span("GONE", "2010-01-04", "2015-01-02"),  # out on the first day
            span("LEFT", "2010-01-04", "2016-06-01"),
            span("STAYS", "2010-01-04"),
            span("JOINS", "2020-01-02"),
            span("LATER", "2027-01-04"),
        )
    )

    tickers = history.tickers_between(date(2015, 1, 2), date(2026, 10, 7))

    assert tickers == ("JOINS", "LEFT", "STAYS")


def test_overlaps_counts_the_first_and_last_day_of_the_window():
    membership = span("AAA", "2020-01-06", "2020-01-09")
    assert membership.overlaps(date(2020, 1, 1), date(2020, 1, 6))
    assert membership.overlaps(date(2020, 1, 8), date(2020, 1, 31))
    assert not membership.overlaps(date(2020, 1, 9), date(2020, 1, 31))


def test_spans_of_lists_a_tickers_stretches_oldest_first():
    later, earlier = span("DOW", "2019-04-02"), span("DOW", "1996-01-02", "2017-09-01")
    history = IndexHistory((later, span("AAA", "2000-01-03"), earlier))
    assert history.spans_of("DOW") == (earlier, later)


def test_a_ticker_can_leave_and_come_back():
    history = IndexHistory(
        (span("DOW", "1996-01-02", "2017-09-01"), span("DOW", "2019-04-02"))
    )

    assert "DOW" in history.members_on(date(2017, 8, 31))
    assert "DOW" not in history.members_on(date(2018, 6, 1))
    assert "DOW" in history.members_on(date(2019, 4, 2))


@pytest.mark.parametrize(
    "spans",
    [
        (span("X", "2010-01-04", "2012-01-03"), span("X", "2011-06-01")),
        (span("X", "2010-01-04"), span("X", "2012-01-03")),  # never left
    ],
)
def test_a_ticker_cannot_be_a_member_twice_at_once(spans):
    with pytest.raises(ValueError, match="overlap"):
        IndexHistory(spans)


def test_a_membership_must_end_after_it_starts():
    with pytest.raises(ValueError):
        span("X", "2020-01-02", "2020-01-02")


def test_tickers_are_upper_case():
    with pytest.raises(ValueError):
        span("aapl", "2020-01-02")


def test_as_of_is_the_last_change():
    history = IndexHistory(
        (span("A", "2010-01-04", "2024-05-01"), span("B", "2023-02-01"))
    )
    assert history.as_of == date(2024, 5, 1)


def test_members_after_as_of_are_the_members_on_as_of():
    history = sp500()
    later = date(2030, 1, 2)
    assert history.members_on(later) == history.members_on(history.as_of)


@pytest.mark.parametrize("year", range(2015, 2027))
def test_sp500_has_about_500_members_in_every_year(year):
    assert 495 <= len(sp500().members_on(date(year, 6, 1))) <= 510


def test_tesla_joined_the_sp500_on_21_december_2020():
    history = sp500()
    assert "TSLA" not in history.members_on(date(2020, 12, 18))
    assert "TSLA" in history.members_on(date(2020, 12, 21))


def test_sp500_keeps_stocks_that_have_left():
    history = sp500()
    # Hess left when Chevron bought it in July 2025.
    assert "HES" in history.tickers_between(date(2015, 1, 2), history.as_of)
    assert "HES" not in history.members_on(history.as_of)


YEAR_2015 = [
    date(2015, 1, 1) + timedelta(n)
    for n in range(365)
    if (date(2015, 1, 1) + timedelta(n)).weekday() < 5
]
FIRST_DAY_2016 = date(2016, 1, 4)


def stock(
    ticker, traded_per_day, *, days=YEAR_2015, member=True, adjustment=1.0, path=None
):
    """2015 bars worth about `traded_per_day` dollars a day, plus 4 January 2016.

    `member` says whether the stock is in the index at the end of 2015.

    Each ticker follows its own random walk around 100 dollars unless `path`
    gives the traded prices. The adjusted close is the traded price times
    `adjustment`, as if dividends paid later had scaled it down.
    """
    rows = [*days, FIRST_DAY_2016]
    n = len(rows)
    if path is None:
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        path = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pl.DataFrame(
        {
            "date": rows,
            "ticker": [ticker] * n,
            "close": path * adjustment,
            "volume": np.round(traded_per_day / path).astype(int),
            "adjustment": [adjustment] * n,
            "split_factor": [1.0] * n,
            "sp500": [member] * n,
        }
    )


def test_dollar_volume_uses_the_price_and_shares_of_that_day():
    # Netflix on 10 November 2025, a week before its 10-for-1 split.
    bars = pl.DataFrame(
        {
            "close": [112.007],
            "adjustment": [0.1],
            "volume": [36_929_000],
            "split_factor": [10.0],
        }
    )
    traded = bars.select(dollar_volume()).item()
    assert traded == pytest.approx(1120.07 * 3_692_900)


def test_trading_universe_takes_the_most_traded_members():
    prices = pl.concat([stock("A", 300_000), stock("B", 200_000), stock("C", 100_000)])
    assert trading_universe(prices, 2016, 2) == ("A", "B")


def test_stocks_outside_the_index_at_the_end_of_the_year_are_left_out():
    prices = pl.concat([stock("IN", 100_000), stock("OUT", 900_000, member=False)])
    assert trading_universe(prices, 2016, 5) == ("IN",)


def test_too_little_trading_last_year_is_left_out():
    recent = YEAR_2015[-(MIN_TRADING_DAYS - 1) :]
    prices = pl.concat([stock("OLD", 100_000), stock("NEW", 900_000, days=recent)])
    assert trading_universe(prices, 2016, 5) == ("OLD",)


def test_later_dividends_do_not_change_the_ranking():
    # PAYS traded slightly more, but dividends paid after 2015 halve its
    # adjusted prices; ranking by adjusted prices would put it second.
    prices = pl.concat([stock("PAYS", 101_000, adjustment=0.5), stock("NONE", 100_000)])
    assert trading_universe(prices, 2016, 1) == ("PAYS",)


def test_nothing_from_the_new_year_counts():
    # The universe must be known the evening before the year's first session.
    prices = pl.concat([stock("A", 200_000), stock("B", 100_000)])
    later = stock("B", 10**9, days=[]).with_columns(
        pl.lit(date(2016, 1, 5)).alias("date")
    )
    assert trading_universe(pl.concat([prices, later]), 2016, 1) == ("A",)


def test_only_the_more_traded_of_two_share_classes_is_kept():
    rng = np.random.default_rng(0)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(YEAR_2015) + 1)))
    prices = pl.concat(
        [
            stock("GOOGL", 300_000, path=walk),
            stock("GOOG", 200_000, path=walk * 1.01),  # one company, another class
            stock("OTHER", 100_000),
        ]
    )
    assert trading_universe(prices, 2016, 2) == ("GOOGL", "OTHER")


def test_a_year_without_the_previous_years_prices_cannot_be_ranked():
    prices = stock("A", 100_000).filter(pl.col("date") >= FIRST_DAY_2016)
    with pytest.raises(ValueError, match="2015"):
        trading_universe(prices, 2016, 1)
