from datetime import date

import pytest

from qfin_alphaguard.data.universe import IndexHistory, Membership, sp500


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
