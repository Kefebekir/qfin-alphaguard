from datetime import UTC, date, datetime

import pytest

from qfin_alphaguard.sessions import session_days, sessions


def session(day):
    (found,) = sessions(day, day)
    return found


def test_a_winter_session_runs_1430_to_2100_utc():
    monday = session(date(2026, 1, 5))  # New York is on EST, UTC-5
    assert monday.open == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    assert monday.close == datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
    assert monday.minutes == 390


def test_summer_time_moves_the_session_an_hour_earlier_in_utc():
    # US summer time began on Sunday 8 March 2026.
    assert session(date(2026, 3, 6)).open.hour == 14
    assert session(date(2026, 3, 9)).open == datetime(2026, 3, 9, 13, 30, tzinfo=UTC)


def test_the_day_after_thanksgiving_closes_at_1300_new_york():
    half_day = session(date(2025, 11, 28))
    assert half_day.close == datetime(2025, 11, 28, 18, 0, tzinfo=UTC)
    assert half_day.minutes == 210


@pytest.mark.parametrize(
    "closed",
    [
        date(2025, 12, 25),  # Christmas
        date(2026, 4, 3),  # Good Friday
        date(2026, 6, 19),  # Juneteenth
        date(2025, 1, 9),  # national day of mourning for President Carter
        date(2018, 12, 5),  # national day of mourning for President Bush
        date(2026, 10, 10),  # a Saturday
    ],
)
def test_no_session_on_holidays_closures_and_weekends(closed):
    assert sessions(closed, closed) == ()


def test_2025_had_250_sessions():
    # 261 weekdays, 10 holidays and the closure for President Carter.
    assert len(session_days(date(2025, 1, 1), date(2025, 12, 31))) == 250


def test_times_are_utc():
    for found in sessions(date(2026, 3, 2), date(2026, 3, 13)):
        assert found.open.tzinfo is UTC
        assert found.close.tzinfo is UTC


def test_days_outside_the_calendar_are_an_error():
    with pytest.raises(ValueError, match="known from"):
        sessions(date(1999, 12, 1), date(2000, 1, 31))
