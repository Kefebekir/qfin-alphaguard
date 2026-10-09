"""When the New York Stock Exchange is open, in UTC.

A regular session runs from 09:30 to 16:00 New York time, or to 13:00 on half
days such as the day after Thanksgiving. There is no session on holidays or on
special closures, such as the national days of mourning for Presidents Bush
(5 December 2018) and Carter (9 January 2025).

The rules come from the exchange_calendars package (calendar XNYS). It knows
future holidays and half days; a special closure is added by its maintainers
once it is announced, so the pinned version needs an update now and then.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cache

import exchange_calendars

# The range the calendar is built for. Asking about a day outside it is an error.
FIRST_DAY = date(2000, 1, 3)
LAST_DAY = date(2035, 12, 31)


@dataclass(frozen=True)
class Session:
    """One trading day: the regular session's open and close, in UTC."""

    day: date
    open: datetime
    close: datetime

    @property
    def minutes(self) -> int:
        """Length of the session: 390 on a full day, 210 on a half day."""
        return int((self.close - self.open).total_seconds() // 60)


@cache
def _nyse() -> exchange_calendars.ExchangeCalendar:
    return exchange_calendars.get_calendar(
        "XNYS", start=FIRST_DAY.isoformat(), end=LAST_DAY.isoformat()
    )


def sessions(first: date, last: date) -> tuple[Session, ...]:
    """Every session from `first` to `last`, both included, oldest first."""
    if not (FIRST_DAY <= first <= last <= LAST_DAY):
        raise ValueError(
            f"sessions are known from {FIRST_DAY} to {LAST_DAY}; "
            f"asked for {first} to {last}"
        )
    schedule = _nyse().schedule.loc[first.isoformat() : last.isoformat()]
    return tuple(
        Session(
            day=label.date(),
            open=opens.to_pydatetime().astimezone(UTC),
            close=closes.to_pydatetime().astimezone(UTC),
        )
        for label, opens, closes in zip(
            schedule.index, schedule["open"], schedule["close"], strict=True
        )
    )


def session_days(first: date, last: date) -> list[date]:
    """The dates of every session from `first` to `last`."""
    return [session.day for session in sessions(first, last)]
