from datetime import UTC, date, datetime

import polars as pl

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.data.intraday import (
    backfill_minutes,
    download_year,
    intraday_path,
    matching_share,
    minute_feed,
    minute_sources,
    regular_session,
)
from qfin_alphaguard.data.universe import IndexHistory, Membership
from qfin_alphaguard.events import Bar


def at(day, hour, minute):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def minutes(ticker, starts, closes=None):
    n = len(starts)
    closes = closes or [100.0] * n
    return pl.DataFrame(
        {
            "start": starts,
            "ticker": [ticker] * n,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [10] * n,
        },
        schema=MINUTE_SCHEMA,
    )


def daily(ticker, days, traded_closes, adjustment=1.0):
    n = len(days)
    return pl.DataFrame(
        {
            "date": days,
            "ticker": [ticker] * n,
            "close": [close * adjustment for close in traded_closes],
            "adjustment": [adjustment] * n,
        }
    )


def span(ticker, start, end=None):
    return Membership(
        ticker, date.fromisoformat(start), date.fromisoformat(end) if end else None
    )


class FakeMinuteClient:
    """Stands in for EodhdClient: the given minutes for each code."""

    def __init__(self, by_code):
        self.by_code = by_code
        self.asked = []

    def minute_bars(self, code, first, last):
        self.asked.append(code)
        bars = self.by_code.get(code, minutes(code, []))
        return bars.filter(pl.col("start").dt.date().is_between(first, last))


def test_only_minutes_inside_the_regular_session_are_kept():
    day = date(2026, 10, 6)  # summer time: the session runs 13:30 to 20:00 UTC
    bars = minutes(
        "AAPL", [at(day, 13, 29), at(day, 13, 30), at(day, 19, 59), at(day, 20, 0)]
    )
    kept = regular_session(bars)["start"].to_list()
    assert kept == [at(day, 13, 30), at(day, 19, 59)]


def test_a_half_day_ends_at_1800_utc():
    day = date(2025, 11, 28)
    bars = minutes("AAPL", [at(day, 17, 59), at(day, 18, 0)])
    assert regular_session(bars)["start"].to_list() == [at(day, 17, 59)]


def test_minutes_on_a_holiday_are_dropped():
    assert regular_session(minutes("AAPL", [at(date(2025, 12, 25), 15, 0)])).is_empty()


def test_the_last_minute_must_match_the_daily_close_as_traded():
    day = date(2026, 10, 6)
    bars = minutes("NFLX", [at(day, 19, 58), at(day, 19, 59)], closes=[1100.0, 1112.0])
    # Before its split Netflix's adjusted close was a tenth of the traded one.
    assert matching_share(bars, daily("NFLX", [day], [1112.5], adjustment=0.1)) == 1.0
    assert matching_share(bars, daily("NFLX", [day], [1200.0], adjustment=0.1)) == 0.0


def test_a_current_code_is_its_own_source():
    history = IndexHistory((span("AAPL", "2000-01-03"),))
    assert minute_sources("AAPL", history) == ("AAPL",)


def test_a_renamed_company_also_tries_the_code_it_became():
    # DowDuPont (DWDP) became DD on 3 June 2019.
    history = IndexHistory(
        (
            span("DD", "1996-01-02", "2017-09-01"),
            span("DWDP", "2017-09-01", "2019-06-03"),
            span("DD", "2019-06-03"),
        )
    )
    assert minute_sources("DWDP", history) == ("DWDP", "DD")


def test_a_renamed_company_is_looked_for_under_the_code_that_replaced_it():
    history = IndexHistory(
        (
            span("FB", "2013-12-23", "2022-06-09"),
            span("META", "2022-06-09"),
            span("XOM", "2000-01-03"),
        )
    )
    assert minute_sources("FB_old", history) == ("META",)


def test_an_old_company_nobody_replaced_that_day_has_no_source():
    history = IndexHistory(
        (span("CHK", "2000-01-03", "2020-06-22"), span("XOM", "2000-01-03"))
    )
    assert minute_sources("CHK_old", history) == ()


def test_a_renamed_company_gets_its_minutes_from_the_new_code():
    day = date(2018, 3, 6)  # winter time: the session runs 14:30 to 21:00 UTC
    history = IndexHistory(
        (span("FB", "2013-12-23", "2022-06-09"), span("META", "2022-06-09"))
    )
    client = FakeMinuteClient(
        {"META": minutes("META", [at(day, 15, 0), at(day, 20, 59)], [180.0, 182.0])}
    )

    stored, summary = download_year(
        client,
        "FB_old",
        2018,
        date(2026, 10, 8),
        daily("FB_old", [day], [182.1]),
        history,
    )

    assert summary.source == "META"
    assert (summary.rows, summary.matching) == (2, 1.0)
    assert stored["ticker"].unique().to_list() == ["FB_old"]


def test_a_year_without_matching_minutes_is_tried_again_next_time(tmp_path):
    day = date(2025, 3, 4)
    history = IndexHistory((span("AAA", "2000-01-03"),))
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(day, 20, 59)], [50.0])})
    prices = daily("AAA", [day], [100.0])  # the minutes do not match

    for _ in range(2):
        backfill_minutes(
            client, prices, history, {2025: ("AAA",)}, tmp_path, date(2026, 10, 8)
        )

    assert not intraday_path(tmp_path, "AAA", 2025).exists()
    assert client.asked == ["AAA", "AAA"]


def test_a_failed_download_is_reported_and_the_others_go_on(tmp_path):
    day = date(2025, 3, 4)
    history = IndexHistory((span("AAA", "2000-01-03"), span("BAD", "2000-01-03")))

    class FailsForBad(FakeMinuteClient):
        def minute_bars(self, code, first, last):
            if code == "BAD":
                raise RuntimeError("EODHD request for intraday/BAD.US failed: HTTP 500")
            return super().minute_bars(code, first, last)

    client = FailsForBad({"AAA": minutes("AAA", [at(day, 20, 59)], [100.0])})
    prices = pl.concat([daily("AAA", [day], [100.0]), daily("BAD", [day], [100.0])])

    done = backfill_minutes(
        client, prices, history, {2025: ("AAA", "BAD")}, tmp_path, date(2026, 10, 8)
    )

    by_ticker = {stored.ticker: stored for stored in done}
    assert by_ticker["AAA"].source == "AAA"
    assert "HTTP 500" in by_ticker["BAD"].error
    assert intraday_path(tmp_path, "AAA", 2025).exists()


def test_a_year_stored_while_it_was_running_is_downloaded_again(tmp_path):
    # Stored on 4 March 2025, when that was the latest day; 5 March came later.
    days = [date(2025, 3, 4), date(2025, 3, 5)]
    history = IndexHistory((span("AAA", "2000-01-03"),))
    store(tmp_path, "AAA", 2025, [at(days[0], 20, 59)])
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(d, 20, 59) for d in days])})

    done = backfill_minutes(
        client,
        daily("AAA", days, [100.0, 100.0]),
        history,
        {2025: ("AAA",)},
        tmp_path,
        date(2026, 10, 8),
    )

    assert client.asked == ["AAA"]
    assert (done[0].days, done[0].trading_days) == (2, 2)
    stored = pl.read_parquet(intraday_path(tmp_path, "AAA", 2025))
    assert stored["start"].dt.date().to_list() == days


def test_the_summary_counts_the_trading_days_that_have_minutes():
    # EODHD lacks some days for good: TSLA's from July 2023 to May 2024.
    days = [date(2018, 3, 5), date(2018, 3, 6), date(2018, 3, 7)]
    history = IndexHistory((span("AAA", "2000-01-03"),))
    one_day = minutes("AAA", [at(days[0], 20, 58), at(days[0], 20, 59)])
    client = FakeMinuteClient({"AAA": one_day})

    _, summary = download_year(
        client, "AAA", 2018, date(2026, 10, 8), daily("AAA", days, [100.0] * 3), history
    )

    # Two minutes on one of three days, and that day matches its close.
    assert (summary.rows, summary.days, summary.trading_days) == (2, 1, 3)
    assert summary.matching == 1.0


def test_minutes_that_do_not_match_the_daily_closes_are_not_kept():
    day = date(2018, 3, 6)
    history = IndexHistory((span("AAA", "2000-01-03"),))
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(day, 20, 59)], [50.0])})

    stored, summary = download_year(
        client, "AAA", 2018, date(2026, 10, 8), daily("AAA", [day], [100.0]), history
    )

    assert stored.is_empty()
    assert summary.source is None


def test_backfill_skips_finished_years_and_writes_one_file_per_stock_and_year(
    tmp_path,
):
    day = date(2025, 3, 4)  # winter time
    history = IndexHistory((span("AAA", "2000-01-03"), span("BBB", "2000-01-03")))
    client = FakeMinuteClient(
        {
            "AAA": minutes("AAA", [at(day, 20, 59)], [100.0]),
            "BBB": minutes("BBB", [at(day, 20, 59)], [50.0]),
        }
    )
    prices = pl.concat([daily("AAA", [day], [100.0]), daily("BBB", [day], [50.0])])
    finished = intraday_path(tmp_path, "AAA", 2025)
    finished.parent.mkdir(parents=True)
    minutes("AAA", [at(day, 20, 59)]).write_parquet(finished)

    done = backfill_minutes(
        client, prices, history, {2025: ("AAA", "BBB")}, tmp_path, date(2026, 10, 8)
    )

    assert [stored.ticker for stored in done] == ["BBB"]
    assert client.asked == ["BBB"]
    assert pl.read_parquet(intraday_path(tmp_path, "BBB", 2025)).height == 1


def test_minutes_that_break_the_bar_rules_are_dropped_and_counted():
    day = date(2018, 3, 6)
    history = IndexHistory((span("AAA", "2000-01-03"),))
    bars = minutes("AAA", [at(day, 20, 58), at(day, 20, 59)], [100.0, 100.0])
    broken = bars.with_columns(  # the first minute closes above its high
        pl.when(pl.col("start") == at(day, 20, 58))
        .then(101.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    client = FakeMinuteClient({"AAA": broken})

    stored, summary = download_year(
        client, "AAA", 2018, date(2026, 10, 8), daily("AAA", [day], [100.0]), history
    )

    assert stored["start"].to_list() == [at(day, 20, 59)]
    assert summary.dropped == 1


def store(root, ticker, year, starts, closes=None):
    path = intraday_path(root, ticker, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    minutes(ticker, starts, closes).write_parquet(path)


def test_the_feed_gives_bars_in_time_order_then_ticker_order(tmp_path):
    day = date(2025, 3, 4)
    store(tmp_path, "BBB", 2025, [at(day, 14, 30), at(day, 14, 31)])
    store(tmp_path, "AAA", 2025, [at(day, 14, 31), at(day, 14, 30)])

    bars = list(minute_feed(tmp_path, {2025: ("BBB", "AAA")}, day, day))

    assert [(bar.start.minute, bar.ticker) for bar in bars] == [
        (30, "AAA"),
        (30, "BBB"),
        (31, "AAA"),
        (31, "BBB"),
    ]
    assert all(isinstance(bar, Bar) and bar.seconds == 60 for bar in bars)
    assert bars[0].start == at(day, 14, 30)  # a UTC time, as every event needs


def test_the_feed_keeps_to_the_days_asked_for_and_skips_missing_files(tmp_path):
    days = [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)]
    store(tmp_path, "AAA", 2025, [at(day, 15, 0) for day in days])

    bars = list(minute_feed(tmp_path, {2025: ("AAA", "NONE")}, days[1], days[2]))

    assert [bar.start.date() for bar in bars] == days[1:]


def test_the_feed_takes_each_year_from_that_years_universe(tmp_path):
    store(tmp_path, "OLD", 2024, [at(date(2024, 12, 31), 15, 0)])
    store(tmp_path, "NEW", 2025, [at(date(2025, 1, 2), 15, 0)])
    store(tmp_path, "OLD", 2025, [at(date(2025, 1, 2), 15, 0)])  # out of the 2025 list

    universes = {2024: ("OLD",), 2025: ("NEW",)}
    bars = list(minute_feed(tmp_path, universes, date(2024, 12, 31), date(2025, 1, 2)))

    assert [bar.ticker for bar in bars] == ["OLD", "NEW"]
