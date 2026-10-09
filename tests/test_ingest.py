from datetime import date, timedelta

import numpy as np
import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.eodhd import BARS_SCHEMA
from qfin_alphaguard.data.ingest import (
    download_sp500,
    load_prices,
    snap_rounding,
    sp500_coverage,
)
from qfin_alphaguard.data.universe import IndexHistory, Membership
from qfin_alphaguard.data.validate import validate_prices

EXPECTED_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adjustment": pl.Float64,
    "split_factor": pl.Float64,
    "sp500": pl.Boolean,
}


def test_load_prices_synthetic():
    df = load_prices(Config(synthetic=True))
    assert dict(df.schema) == EXPECTED_SCHEMA


def test_load_prices_is_sorted_by_ticker_then_date():
    df = load_prices(Config(synthetic=True))
    assert df.equals(df.sort(["ticker", "date"]))


def weekdays(first, last):
    days = (first + timedelta(n) for n in range((last - first).days + 1))
    return [day for day in days if day.weekday() < 5]


def span(ticker, start, end=None):
    return Membership(
        ticker, date.fromisoformat(start), date.fromisoformat(end) if end else None
    )


class FakeClient:
    """Stands in for EodhdClient: flat bars for each code on the given days."""

    def __init__(self, days_by_code, old_codes=None, adjustments=None, splits=None):
        self.days_by_code = days_by_code
        self._old_codes = old_codes or {}
        self.adjustments = adjustments or {}
        self._splits = splits or {}
        self.downloads = []
        self.split_requests = []

    def old_codes(self):
        return self._old_codes

    def splits(self, code):
        self.split_requests.append(code)
        return self._splits.get(code, [])

    def daily_bars(self, code, start, end):
        self.downloads.append(code)
        days = [day for day in self.days_by_code.get(code, []) if start <= day <= end]
        n = len(days)
        return pl.DataFrame(
            {
                "date": days,
                "ticker": [code] * n,
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.5] * n,
                "volume": [1000] * n,
                "adjustment": self.adjustments.get(code, [1.0] * n),
            },
            schema=BARS_SCHEMA,
        )


def test_bars_are_marked_sp500_only_while_a_member():
    history = IndexHistory((span("AAA", "2020-01-06", "2020-01-09"),))
    first, last = date(2020, 1, 2), date(2020, 1, 10)
    client = FakeClient({"AAA": weekdays(first, last)})

    prices = download_sp500(client, history, first, last)

    assert dict(prices.schema) == EXPECTED_SCHEMA
    assert prices.height == 7  # days outside the membership are kept, unmarked
    assert prices.filter(pl.col("sp500"))["date"].to_list() == [
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
    ]


def test_a_reused_ticker_takes_each_company_for_its_own_stretch():
    # DOW was Dow Chemical (EODHD code DOW_old) until 2017 and Dow Inc. from 2019.
    history = IndexHistory(
        (span("DOW", "2015-01-02", "2017-09-01"), span("DOW", "2019-04-02"))
    )
    client = FakeClient(
        {
            "DOW_old": weekdays(date(2017, 8, 28), date(2017, 9, 8)),
            "DOW": weekdays(date(2019, 3, 20), date(2019, 4, 5)),
        },
        old_codes={"DOW": ("DOW_old",)},
    )

    prices = download_sp500(client, history, date(2017, 8, 28), date(2019, 4, 5))

    members = prices.filter(pl.col("sp500"))
    assert members.filter(pl.col("ticker") == "DOW_old")["date"].max() == date(
        2017, 8, 31
    )
    assert members.filter(pl.col("ticker") == "DOW")["date"].min() == date(2019, 4, 2)
    assert client.downloads == ["DOW", "DOW_old"]  # each code fetched once


def test_class_shares_are_fetched_with_a_dash():
    history = IndexHistory((span("BRK.B", "2020-01-02"),))
    client = FakeClient({"BRK-B": weekdays(date(2020, 1, 2), date(2020, 1, 3))})

    prices = download_sp500(client, history, date(2020, 1, 2), date(2020, 1, 3))

    assert client.downloads == ["BRK-B"]
    assert prices["ticker"].unique().to_list() == ["BRK-B"]


def test_split_factor_comes_from_the_split_history_when_the_adjustment_jumps():
    # Netflix split 10-for-1 on 17 November 2025: the adjustment jumps from 0.1 to 1.
    days = [date(2025, 11, 13), date(2025, 11, 14), date(2025, 11, 17)]
    client = FakeClient(
        {"NFLX": days},
        adjustments={"NFLX": [0.1, 0.1, 1.0]},
        splits={"NFLX": [(date(2025, 11, 17), 10.0)]},
    )
    history = IndexHistory((span("NFLX", "2010-12-20"),))

    prices = download_sp500(client, history, days[0], days[-1])

    assert prices["split_factor"].to_list() == [10.0, 10.0, 1.0]
    assert client.split_requests == ["NFLX"]


def test_no_split_history_is_fetched_without_a_jump():
    days = weekdays(date(2020, 1, 2), date(2020, 1, 10))
    client = FakeClient({"AAA": days}, adjustments={"AAA": [0.97] * len(days)})
    history = IndexHistory((span("AAA", "2020-01-02"),))

    prices = download_sp500(client, history, days[0], days[-1])

    assert client.split_requests == []
    assert prices["split_factor"].unique().to_list() == [1.0]


def test_bars_on_days_the_exchange_was_closed_are_dropped():
    # 1 and 9 January 2025: New Year's Day and the closure for President Carter.
    days = weekdays(date(2024, 12, 30), date(2025, 1, 10))
    client = FakeClient({"A": days})
    history = IndexHistory((span("A", "2020-01-02"),))

    prices = download_sp500(client, history, days[0], days[-1])

    dates = prices["date"].to_list()
    assert date(2025, 1, 1) not in dates
    assert date(2025, 1, 9) not in dates
    assert len(dates) == len(days) - 2


def test_a_member_without_data_lowers_the_coverage():
    history = IndexHistory((span("AAA", "2020-01-02"), span("BBB", "2020-01-02")))
    first, last = date(2020, 1, 2), date(2020, 1, 10)
    client = FakeClient({"AAA": weekdays(first, last)})  # nothing for BBB

    prices = download_sp500(client, history, first, last)

    assert prices["ticker"].unique().to_list() == ["AAA"]
    assert sp500_coverage(prices, history) == 0.5


def one_bar(**prices):
    return pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "ticker": ["AAA"],
            **{name: [value] for name, value in prices.items()},
            "volume": [1000],
            "adjustment": [1.0],
            "split_factor": [1.0],
            "sp500": [True],
        }
    )


def test_close_a_rounding_step_above_the_high_is_snapped():
    close = np.nextafter(100.0, 101.0)  # the next float above 100
    bars = one_bar(open=99.0, high=100.0, low=98.0, close=close)

    snapped = snap_rounding(bars)

    assert snapped["high"][0] == close
    assert validate_prices(snapped).ok


def test_open_a_rounding_step_below_the_low_is_snapped():
    low = np.nextafter(98.0, 99.0)  # the next float above the open
    bars = one_bar(open=98.0, high=100.0, low=low, close=99.0)

    snapped = snap_rounding(bars)

    assert snapped["low"][0] == 98.0
    assert validate_prices(snapped).ok


def test_a_real_inconsistency_is_left_for_validation():
    bars = one_bar(open=99.0, high=100.0, low=98.0, close=101.0)

    snapped = snap_rounding(bars)

    assert snapped.equals(bars)
    assert not validate_prices(snapped).ok
