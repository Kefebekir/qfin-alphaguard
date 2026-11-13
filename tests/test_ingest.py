from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.corporate import corporate_actions
from qfin_alphaguard.data.eodhd import BARS_SCHEMA
from qfin_alphaguard.data.ingest import (
    BENCHMARKS,
    continue_prices,
    continue_with,
    download_benchmarks,
    download_sp500,
    load_prices,
    repair_broken_bars,
    snap_rounding,
    sp500_coverage,
)
from qfin_alphaguard.data.universe import IndexHistory, Membership
from qfin_alphaguard.data.validate import validate_prices
from qfin_alphaguard.events import Dividend, Split
from qfin_alphaguard.sessions import session_days

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


def test_the_benchmark_etf_is_stored_like_a_stock_but_never_as_a_member():
    days = weekdays(date(2020, 1, 6), date(2020, 1, 10))
    client = FakeClient({"SPY": days})

    prices = download_benchmarks(client, BENCHMARKS, days[0], days[-1])

    assert BENCHMARKS == ("SPY",)
    assert dict(prices.schema) == EXPECTED_SCHEMA
    assert prices["ticker"].unique().to_list() == ["SPY"]
    assert prices["date"].to_list() == days
    assert not prices["sp500"].any()
    assert validate_prices(prices).ok


def test_no_benchmark_gives_an_empty_frame_in_the_members_columns():
    prices = download_benchmarks(FakeClient({}), (), date(2020, 1, 6), date(2020, 1, 6))
    assert prices.is_empty()
    assert dict(prices.schema) == EXPECTED_SCHEMA


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


def test_a_bar_with_a_wrong_high_and_low_is_widened_to_its_open_and_close():
    # LUMN, 12 October 2026, as EODHD sent it.
    bars = pl.concat(
        [
            one_bar(open=5.25, high=5.05, low=5.05, close=5.05),
            one_bar(open=5.56, high=5.58, low=5.13, close=5.28).with_columns(
                pl.lit(date(2024, 1, 3)).alias("date")
            ),
        ]
    )

    repaired, broken = repair_broken_bars(bars)

    assert broken.equals(bars.head(1))
    assert repaired.select("open", "high", "low", "close").rows() == [
        (5.25, 5.25, 5.05, 5.05),
        (5.56, 5.58, 5.13, 5.28),  # a sound bar is left as it was
    ]
    assert validate_prices(repaired).ok


def test_a_close_below_the_low_moves_the_low_down():
    repaired, broken = repair_broken_bars(
        one_bar(open=4.96, high=4.985, low=4.985, close=4.99)
    )
    assert repaired.select("high", "low").row(0) == (4.99, 4.96)
    assert broken.height == 1


def test_a_real_inconsistency_is_left_for_validation():
    bars = one_bar(open=99.0, high=100.0, low=98.0, close=101.0)

    snapped = snap_rounding(bars)

    assert snapped.equals(bars)
    assert not validate_prices(snapped).ok


# EODHD's history continued by Alpaca's bars (decision 0022)

NOV = session_days(date(2026, 11, 2), date(2026, 11, 20))  # 11 November is a session
HISTORY = IndexHistory((span("AAA", "2015-01-02"), span("BRK.B", "2015-01-02")))


def stored_bars(ticker, days, closes, adjustment=1.0, split_factor=1.0, volume=1000):
    n = len(days)
    return pl.DataFrame(
        {
            "date": days,
            "ticker": [ticker] * n,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [volume] * n,
            "adjustment": [adjustment] * n,
            "split_factor": [split_factor] * n,
            "sp500": [True] * n,
        },
        schema=EXPECTED_SCHEMA,
    )


def new_bars(ticker, days, closes, volume=500):
    """Alpaca's daily bars as traded, as AlpacaClient.daily_bars returns them."""
    n = len(days)
    return pl.DataFrame(
        {
            "date": days,
            "ticker": [ticker] * n,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [volume] * n,
        },
        schema={k: EXPECTED_SCHEMA[k] for k in ("date", "ticker", "open", "high")}
        | {"low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )


def of(prices, ticker, column):
    return prices.filter(pl.col("ticker") == ticker).sort("date")[column].to_list()


def test_stored_bars_stay_exactly_as_they_were_and_new_ones_follow_as_traded():
    stored = pl.concat(
        [
            stored_bars("AAA", NOV[:1], [99.0], adjustment=0.98),
            stored_bars("AAA", NOV[1:3], [100.0, 101.0]),
        ]
    )

    out = continue_with(stored, new_bars("AAA", NOV[3:5], [103.0, 104.0]), [], HISTORY)

    assert out.filter(pl.col("date") <= NOV[2]).equals(stored)
    assert out.filter(pl.col("date") > NOV[2]).rows() == [
        (NOV[3], "AAA", 103.0, 104.0, 102.0, 103.0, 500, 1.0, 1.0, True),
        (NOV[4], "AAA", 104.0, 105.0, 103.0, 104.0, 500, 1.0, 1.0, True),
    ]
    assert dict(out.schema) == EXPECTED_SCHEMA


def test_a_split_after_the_stored_days_reaches_back_over_all_of_them():
    stored = stored_bars("AAA", NOV[:3], [100.0, 100.0, 100.0])
    new = new_bars("AAA", NOV[3:5], [50.0, 51.0])

    out = continue_with(stored, new, [Split("AAA", NOV[3], 2.0)], HISTORY)

    assert of(out, "AAA", "close") == [50.0, 50.0, 50.0, 50.0, 51.0]
    assert of(out, "AAA", "split_factor") == [2.0, 2.0, 2.0, 1.0, 1.0]
    assert of(out, "AAA", "adjustment") == [0.5, 0.5, 0.5, 1.0, 1.0]
    assert of(out, "AAA", "volume") == [2000, 2000, 2000, 500, 500]
    assert corporate_actions(out) == [Split("AAA", NOV[3], 2.0)]


def test_a_dividend_scales_the_bars_before_its_ex_date_by_the_close_before_it():
    stored = stored_bars("AAA", NOV[:3], [100.0, 100.0, 100.0])
    new = new_bars("AAA", NOV[3:6], [120.0, 117.0, 118.0])

    out = continue_with(stored, new, [Dividend("AAA", NOV[4], 3.0)], HISTORY)

    factor = 1 - 3.0 / 120.0  # the traded close of the session before the ex-date
    assert of(out, "AAA", "close") == pytest.approx(
        [100 * factor] * 3 + [120 * factor, 117.0, 118.0]
    )
    assert of(out, "AAA", "split_factor") == [1.0] * 6
    (found,) = corporate_actions(out)
    assert (found.ticker, found.day) == ("AAA", NOV[4])
    assert found.amount == pytest.approx(3.0)


def test_events_by_the_last_stored_day_or_after_the_last_bar_are_left_out():
    stored = stored_bars("AAA", NOV[:3], [100.0, 100.0, 100.0])
    new = new_bars("AAA", NOV[3:5], [100.0, 100.0])
    events = [
        Dividend("AAA", NOV[2], 1.0),  # already in EODHD's adjustment
        Split("AAA", NOV[1], 2.0),
        Dividend("AAA", NOV[5], 1.0),  # its ex-date has no bar yet
        Dividend("BBB", NOV[4], 1.0),  # no such ticker
    ]

    assert continue_with(stored, new, events, HISTORY).equals(
        continue_with(stored, new, [], HISTORY)
    )


def test_a_ticker_with_nothing_stored_comes_whole_and_a_member_from_joining():
    history = IndexHistory((*HISTORY.spans, span("NEW", NOV[3].isoformat())))
    stored = stored_bars("AAA", NOV[:3], [100.0, 100.0, 100.0])
    new = pl.concat(
        [
            new_bars("AAA", NOV[3:5], [100.0, 100.0]),
            new_bars("NEW", NOV[:5], [10.0, 10.0, 10.0, 20.0, 20.0]),
        ]
    )

    out = continue_with(stored, new, [Split("NEW", NOV[3], 0.5)], history)

    assert of(out, "NEW", "close") == [20.0] * 5  # a one-for-two reverse split
    assert of(out, "NEW", "split_factor") == [0.5, 0.5, 0.5, 1.0, 1.0]
    assert of(out, "NEW", "sp500") == [False, False, False, True, True]


def test_a_ticker_without_new_bars_is_left_as_stored():
    stored = pl.concat(
        [
            stored_bars("AAA", NOV[:3], [100.0, 100.0, 100.0]),
            stored_bars("AAA_old", NOV[:3], [7.0, 8.0, 9.0], adjustment=0.5),
        ]
    )
    new = new_bars("AAA", NOV[3:5], [50.0, 50.0])

    out = continue_with(stored, new, [Split("AAA", NOV[3], 2.0)], HISTORY)

    assert out.filter(pl.col("ticker") == "AAA_old").equals(
        stored.filter(pl.col("ticker") == "AAA_old")
    )


def test_stored_factors_are_brought_to_1_on_the_last_stored_day_first():
    # Were EODHD's factors not 1 there, an event after it would count twice.
    stored = stored_bars(
        "AAA",
        NOV[:3],
        [50.0, 50.0, 50.0],
        adjustment=0.5,
        split_factor=2.0,
        volume=2000,
    )
    new = new_bars("AAA", NOV[3:5], [100.0, 101.0])

    out = continue_with(stored, new, [], HISTORY)

    assert of(out, "AAA", "close") == [100.0, 100.0, 100.0, 100.0, 101.0]
    assert of(out, "AAA", "adjustment") == [1.0] * 5
    assert of(out, "AAA", "split_factor") == [1.0] * 5
    assert of(out, "AAA", "volume") == [1000, 1000, 1000, 500, 500]


class FakeAlpaca:
    """Stands in for AlpacaClient: daily bars, by symbol, and corporate actions."""

    def __init__(self, bars, actions=(), spun=None):
        self.bars = bars
        self.spun = spun
        self.actions = list(actions)
        self.calls = []

    def daily_bars(self, symbols, first, end, adjustment="raw"):
        self.calls.append(("bars", tuple(symbols), first, end, adjustment))
        bars = self.bars if adjustment == "raw" else self.spun
        return bars.filter(
            pl.col("ticker").is_in(list(symbols))
            & (pl.col("date") >= first)
            & (pl.col("date") <= end.date())  # a session not yet over comes too
        )

    def corporate_actions(self, symbols, first, last):
        self.calls.append(("actions", tuple(symbols), first, last))
        return [a for a in self.actions if a["symbol"] in symbols]


def at(day, hour, minute):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def test_a_session_is_taken_once_it_has_been_over_for_fifteen_minutes():
    stored = stored_bars("AAA", NOV[:7], [100.0] * 7)  # to Tuesday 10 November
    alpaca = FakeAlpaca(new_bars("AAA", NOV[7:9], [101.0, 102.0]))
    wednesday, thursday = NOV[7], NOV[8]  # the close is at 21:00 UTC in November

    early = continue_prices(stored, alpaca, HISTORY, NOV[-1], at(thursday, 21, 10))
    late = continue_prices(stored, alpaca, HISTORY, NOV[-1], at(thursday, 21, 20))
    told = continue_prices(stored, alpaca, HISTORY, wednesday, at(thursday, 21, 20))

    assert early["date"].max() == wednesday == told["date"].max()
    assert late["date"].max() == thursday
    assert alpaca.calls[0] == (
        "bars",
        ("AAA",),
        wednesday,
        at(thursday, 20, 54),
        "raw",
    )


def test_nothing_is_asked_before_a_new_session_is_over():
    stored = stored_bars("AAA", NOV[:7], [100.0] * 7)
    alpaca = FakeAlpaca(new_bars("AAA", NOV[7:9], [101.0, 102.0]))

    assert (
        continue_prices(stored, alpaca, HISTORY, NOV[-1], at(NOV[7], 21, 10)) is stored
    )
    assert continue_prices(stored, alpaca, HISTORY, NOV[6], at(NOV[9], 22, 0)) is stored
    assert alpaca.calls == []


def test_who_goes_on_who_joins_and_who_is_left_as_stored():
    history = IndexHistory((*HISTORY.spans, span("NEW", NOV[8].isoformat())))
    stored = pl.concat(
        [
            stored_bars("AAA", NOV[:7], [100.0] * 7),
            stored_bars("BRK-B", NOV[:7], [400.0] * 7),
            stored_bars("XYZ_old", NOV[:7], [5.0] * 7),  # an earlier company
            stored_bars("GONE", NOV[:3], [9.0] * 3),  # stopped trading, not a member
        ]
    )
    alpaca = FakeAlpaca(
        pl.concat(
            [
                new_bars("AAA", NOV[7:9], [101.0, 102.0]),
                new_bars("BRK.B", NOV[7:9], [401.0, 402.0]),
                new_bars("XYZ", NOV[7:9], [1.0, 1.0]),
                new_bars("GONE", NOV[7:9], [1.0, 1.0]),
                new_bars("NEW", NOV[:9], [30.0] * 9),
            ]
        )
    )

    out = continue_prices(stored, alpaca, history, NOV[-1], at(NOV[8], 22, 0))

    assert [c[1] for c in alpaca.calls if c[0] == "bars"] == [
        ("AAA", "BRK.B"),
        ("NEW",),
    ]
    assert [c[2] for c in alpaca.calls] == [NOV[7], NOV[7], NOV[0], NOV[0]]
    assert of(out, "BRK-B", "close") == [400.0] * 7 + [401.0, 402.0]
    assert of(out, "BRK-B", "sp500") == [True] * 9  # a member as BRK.B
    assert len(of(out, "NEW", "close")) == 9
    assert of(out, "NEW", "sp500") == [False] * 8 + [True]
    for ticker in ("XYZ_old", "GONE"):
        assert out.filter(pl.col("ticker") == ticker).equals(
            stored.filter(pl.col("ticker") == ticker)
        )


def test_alpaca_s_actions_become_splits_and_dividends_and_a_spin_off_a_split():
    stored = pl.concat(
        [
            stored_bars("AAA", NOV[:7], [100.0] * 7),
            stored_bars("BRK-B", NOV[:7], [400.0] * 7),
        ]
    )
    raw = pl.concat(
        [
            new_bars("AAA", NOV[7:10], [100.0, 81.0, 82.0]),
            new_bars("BRK.B", NOV[7:10], [400.0, 400.0, 199.0]),
        ]
    )
    spun = pl.concat(
        [
            new_bars("AAA", NOV[7:10], [80.0, 81.0, 82.0]),  # a fifth of it spun off
            new_bars("BRK.B", NOV[7:10], [400.0, 400.0, 199.0]),  # not adjusted
        ]
    )
    actions = [
        {"kind": "spin_offs", "symbol": "AAA", "ex_date": NOV[8].isoformat()},
        {"kind": "spin_offs", "symbol": "BRK.B", "ex_date": NOV[8].isoformat()},
        {
            "kind": "forward_splits",
            "symbol": "BRK.B",
            "ex_date": NOV[9].isoformat(),
            "new_rate": 2,
            "old_rate": 1,
        },
        {
            "kind": "cash_dividends",
            "symbol": "BRK.B",
            "ex_date": NOV[8].isoformat(),
            "rate": 4.0,
        },
        {
            "kind": "cash_dividends",
            "symbol": "AAA",
            "ex_date": NOV[12].isoformat(),  # announced, not yet reached
            "rate": 1.0,
        },
    ]
    alpaca = FakeAlpaca(raw, actions, spun)

    out = continue_prices(stored, alpaca, HISTORY, NOV[-1], at(NOV[9], 22, 0))

    found = corporate_actions(out)
    assert [(type(a).__name__, a.ticker, a.day) for a in found] == [
        ("Split", "AAA", NOV[8]),
        ("Dividend", "BRK-B", NOV[8]),
        ("Split", "BRK-B", NOV[9]),
    ]
    assert found[0].ratio == pytest.approx(1.25)
    assert found[1].amount == pytest.approx(4.0)
    assert found[2].ratio == pytest.approx(2.0)
    spin_offs = [c for c in alpaca.calls if c[0] == "bars" and c[4] == "spin-off"]
    assert [c[1] for c in spin_offs] == [("AAA", "BRK.B")]
