from datetime import UTC, date, datetime, timedelta

import polars as pl

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.data.intraday import (
    backfill_minutes,
    by_day,
    download_year,
    intraday_path,
    matching_share,
    minute_feed,
    minute_windows,
    regular_session,
)
from qfin_alphaguard.data.universe import IndexHistory, Membership
from qfin_alphaguard.events import Bar
from qfin_alphaguard.sessions import sessions


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


def whole_session(ticker, day):
    """Every minute of the day's session, closing at 100."""
    (session,) = sessions(day, day)
    starts = [session.open + timedelta(minutes=k) for k in range(session.minutes)]
    return minutes(ticker, starts)


def labelled(frame, source):
    return frame.with_columns(pl.lit(source).alias("source"))


def daily(ticker, days, traded_closes, adjustment=1.0):
    n = len(days)
    return pl.DataFrame(
        {
            "date": days,
            "ticker": [ticker] * n,
            "close": [close * adjustment for close in traded_closes],
            "adjustment": [adjustment] * n,
        },
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "close": pl.Float64,
            "adjustment": pl.Float64,
        },
    )


def span(ticker, start, end=None):
    return Membership(
        ticker, date.fromisoformat(start), date.fromisoformat(end) if end else None
    )


HISTORY = IndexHistory((span("AAA", "2000-01-03"),))


class FakeMinuteClient:
    """Stands in for a provider: the given minutes for each code.

    A stock's minutes are looked for under its own code, or under the codes
    `codes` gives for it.
    """

    def __init__(self, by_code, codes=None, name="EODHD"):
        self.by_code = by_code
        self.codes = codes or {}
        self.name = name
        self.asked = []

    def minute_codes(self, code, history):
        return self.codes.get(code, (code,))

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


def test_each_day_comes_whole_from_the_source_with_more_minutes():
    days = [date(2018, 3, 5), date(2018, 3, 6), date(2018, 3, 7)]
    first = labelled(
        minutes("AAA", [at(days[0], 15, 0), at(days[0], 15, 1), at(days[1], 15, 0)]),
        "EODHD",
    )
    second = labelled(
        minutes(
            "AAA",
            [
                at(days[0], 15, 0),
                at(days[1], 15, 0),
                at(days[1], 15, 1),
                at(days[2], 15, 0),
            ],
        ),
        "Alpaca",
    )

    picked = by_day(first, second)

    # The first has more of 5 March, the second more of 6 March and all of 7 March.
    assert picked.select("start", "source").rows() == [
        (at(days[0], 15, 0), "EODHD"),
        (at(days[0], 15, 1), "EODHD"),
        (at(days[1], 15, 0), "Alpaca"),
        (at(days[1], 15, 1), "Alpaca"),
        (at(days[2], 15, 0), "Alpaca"),
    ]


def test_a_tied_day_comes_from_the_first_source_and_is_not_mixed():
    day = date(2018, 3, 5)
    first = labelled(minutes("AAA", [at(day, 15, 0)], [100.0]), "EODHD")
    second = labelled(minutes("AAA", [at(day, 15, 1)], [101.0]), "Alpaca")

    picked = by_day(first, second)

    assert picked.select("start", "source").rows() == [(at(day, 15, 0), "EODHD")]


def test_the_second_provider_fills_the_days_the_first_lacks():
    days = [date(2018, 3, 5), date(2018, 3, 6)]
    eodhd = FakeMinuteClient({"AAA": minutes("AAA", [at(days[0], 20, 59)])})
    alpaca = FakeMinuteClient(
        {"AAA": minutes("AAA", [at(day, 20, 59) for day in days])}, name="Alpaca"
    )

    stored, summary = download_year(
        (eodhd, alpaca),
        "AAA",
        2018,
        date(2026, 10, 8),
        daily("AAA", days, [100.0, 100.0]),
        HISTORY,
    )

    assert stored.select(pl.col("start").dt.date(), "source").rows() == [
        (days[0], "EODHD"),
        (days[1], "Alpaca"),
    ]
    assert summary.parts == (("EODHD", "AAA", 1), ("Alpaca", "AAA", 1))
    assert (summary.rows, summary.days, summary.trading_days) == (2, 2, 2)


def test_the_second_provider_is_not_asked_when_the_first_has_every_minute():
    day = date(2018, 3, 6)
    eodhd = FakeMinuteClient({"AAA": whole_session("AAA", day)})
    alpaca = FakeMinuteClient({}, name="Alpaca")

    stored, summary = download_year(
        (eodhd, alpaca),
        "AAA",
        2018,
        date(2026, 10, 8),
        daily("AAA", [day], [100.0]),
        HISTORY,
    )

    assert alpaca.asked == []
    assert stored.height == 390
    assert summary.parts == (("EODHD", "AAA", 1),)


def test_a_day_the_first_provider_has_only_in_part_comes_from_the_second():
    day = date(2018, 3, 6)
    eodhd = FakeMinuteClient(
        {"AAA": minutes("AAA", [at(day, 20, 58), at(day, 20, 59)])}
    )
    alpaca = FakeMinuteClient(
        {"AAA": minutes("AAA", [at(day, 20, 57), at(day, 20, 58), at(day, 20, 59)])},
        name="Alpaca",
    )

    stored, summary = download_year(
        (eodhd, alpaca),
        "AAA",
        2018,
        date(2026, 10, 8),
        daily("AAA", [day], [100.0]),
        HISTORY,
    )

    assert stored["source"].to_list() == ["Alpaca"] * 3
    assert summary.parts == (("EODHD", "AAA", 0), ("Alpaca", "AAA", 1))


def test_a_year_without_daily_bars_asks_no_provider():
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(date(2018, 3, 6), 20, 59)])})

    stored, summary = download_year(
        (client,), "AAA", 2018, date(2026, 10, 8), daily("AAA", [], []), HISTORY
    )

    assert client.asked == []
    assert stored.is_empty()
    assert summary.parts == ()


def test_broken_minutes_are_counted_from_every_provider():
    day = date(2018, 3, 6)
    bars = minutes("AAA", [at(day, 20, 58), at(day, 20, 59)])
    broken = bars.with_columns(  # the first minute closes above its high
        pl.when(pl.col("start") == at(day, 20, 58))
        .then(101.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    eodhd = FakeMinuteClient({"AAA": broken})
    alpaca = FakeMinuteClient({"AAA": broken}, name="Alpaca")

    _, summary = download_year(
        (eodhd, alpaca),
        "AAA",
        2018,
        date(2026, 10, 8),
        daily("AAA", [day], [100.0]),
        HISTORY,
    )

    assert summary.dropped == 2


def test_a_stock_the_first_provider_lacks_wholly_comes_from_the_second():
    # EODHD has no minutes for the old Chesapeake; Alpaca has them under CHK.
    day = date(2017, 3, 1)
    eodhd = FakeMinuteClient({})
    alpaca = FakeMinuteClient(
        {"CHK": minutes("CHK", [at(day, 20, 59)], [5.57])},
        codes={"CHK_old": ("CHK",)},
        name="Alpaca",
    )

    stored, summary = download_year(
        (eodhd, alpaca),
        "CHK_old",
        2017,
        date(2026, 10, 8),
        daily("CHK_old", [day], [5.57]),
        HISTORY,
    )

    assert summary.parts == (("Alpaca", "CHK", 1),)
    assert stored.select("ticker", "source").rows() == [("CHK_old", "Alpaca")]


def test_a_renamed_company_gets_its_minutes_from_the_code_that_matches():
    day = date(2018, 3, 6)  # winter time: the session runs 14:30 to 21:00 UTC
    history = IndexHistory(
        (span("FB", "2013-12-23", "2022-06-09"), span("META", "2022-06-09"))
    )
    client = FakeMinuteClient(
        {
            "ANF": minutes("ANF", [at(day, 20, 59)], [30.0]),  # another company
            "META": minutes("META", [at(day, 15, 0), at(day, 20, 59)], [180.0, 182.0]),
        },
        codes={"FB_old": ("ANF", "META")},
    )

    stored, summary = download_year(
        (client,),
        "FB_old",
        2018,
        date(2026, 10, 8),
        daily("FB_old", [day], [182.1]),
        history,
    )

    assert summary.parts == (("EODHD", "META", 1),)
    assert (summary.rows, summary.matching) == (2, 1.0)
    assert stored["ticker"].unique().to_list() == ["FB_old"]


def test_an_etf_outside_the_index_is_stored_under_its_own_code(tmp_path):
    # QQQ's minutes come from Alpaca first (decision 0011); the index never had it.
    day = date(2026, 10, 9)
    history = IndexHistory((span("AAPL", "2000-01-03"),))
    client = FakeMinuteClient(
        {"QQQ": minutes("QQQ", [at(day, 19, 59)], [600.0])}, name="Alpaca"
    )

    done = backfill_minutes(
        (client,),
        daily("QQQ", [day], [600.0]),
        history,
        {2026: ("QQQ",)},
        tmp_path,
        day,
    )

    assert [(stored.ticker, stored.parts) for stored in done] == [
        ("QQQ", (("Alpaca", "QQQ", 1),))
    ]
    stored = pl.read_parquet(intraday_path(tmp_path, "QQQ", 2026))
    assert stored["start"].to_list() == [at(day, 19, 59)]


def test_a_year_without_matching_minutes_is_tried_again_next_time(tmp_path):
    day = date(2025, 3, 4)
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(day, 20, 59)], [50.0])})
    prices = daily("AAA", [day], [100.0])  # the minutes do not match

    for _ in range(2):
        backfill_minutes(
            (client,), prices, HISTORY, {2025: ("AAA",)}, tmp_path, date(2026, 10, 8)
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
        (client,), prices, history, {2025: ("AAA", "BAD")}, tmp_path, date(2026, 10, 8)
    )

    by_ticker = {stored.ticker: stored for stored in done}
    assert by_ticker["AAA"].parts == (("EODHD", "AAA", 1),)
    assert "HTTP 500" in by_ticker["BAD"].error
    assert intraday_path(tmp_path, "AAA", 2025).exists()


def test_a_failing_second_provider_fails_the_year_and_keeps_its_old_file(tmp_path):
    days = [date(2025, 3, 4), date(2025, 3, 5)]
    store(tmp_path, "AAA", 2025, [at(days[0], 20, 59)])  # stored on 4 March

    class Failing(FakeMinuteClient):
        def minute_bars(self, code, first, last):
            raise RuntimeError("Alpaca request for AAA failed: HTTP 500")

    eodhd = FakeMinuteClient({"AAA": minutes("AAA", [at(day, 20, 59) for day in days])})

    done = backfill_minutes(
        (eodhd, Failing({}, name="Alpaca")),
        daily("AAA", days, [100.0, 100.0]),
        HISTORY,
        {2025: ("AAA",)},
        tmp_path,
        date(2026, 10, 8),
    )

    assert "HTTP 500" in done[0].error
    stored = pl.read_parquet(intraday_path(tmp_path, "AAA", 2025))
    assert stored["start"].to_list() == [at(days[0], 20, 59)]


def test_a_year_stored_while_it_was_running_is_downloaded_again(tmp_path):
    # Stored on 4 March 2025, when that was the latest day; 5 March came later.
    days = [date(2025, 3, 4), date(2025, 3, 5)]
    store(tmp_path, "AAA", 2025, [at(days[0], 20, 59)])
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(d, 20, 59) for d in days])})

    done = backfill_minutes(
        (client,),
        daily("AAA", days, [100.0, 100.0]),
        HISTORY,
        {2025: ("AAA",)},
        tmp_path,
        date(2026, 10, 8),
    )

    assert client.asked == ["AAA"]
    assert (done[0].days, done[0].trading_days) == (2, 2)
    stored = pl.read_parquet(intraday_path(tmp_path, "AAA", 2025))
    assert stored["start"].dt.date().to_list() == days


def test_minutes_stored_before_the_source_column_are_downloaded_again(tmp_path):
    day = date(2025, 3, 4)
    store(tmp_path, "AAA", 2025, [at(day, 20, 59)], source=None)  # up to date
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(day, 20, 59)])})

    backfill_minutes(
        (client,),
        daily("AAA", [day], [100.0]),
        HISTORY,
        {2025: ("AAA",)},
        tmp_path,
        date(2026, 10, 8),
    )

    assert client.asked == ["AAA"]
    stored = pl.read_parquet(intraday_path(tmp_path, "AAA", 2025))
    assert stored["source"].to_list() == ["EODHD"]


def test_a_year_needs_its_universe_all_year_and_the_stocks_that_left_until_january():
    windows = minute_windows({2024: ("A", "B"), 2025: ("B", "C")})

    assert windows == {
        2024: {"A": date(2024, 12, 31), "B": date(2024, 12, 31)},
        2025: {
            "B": date(2025, 12, 31),
            "C": date(2025, 12, 31),
            "A": date(2025, 1, 31),
        },
    }


def test_a_stock_that_left_the_universe_gets_its_minutes_for_january(tmp_path):
    days = [date(2025, 1, 2), date(2025, 2, 3)]
    history = IndexHistory((span("OLD", "2000-01-03"), span("NEW", "2000-01-03")))
    client = FakeMinuteClient(
        {
            "OLD": minutes("OLD", [at(day, 20, 59) for day in days]),
            "NEW": minutes("NEW", [at(day, 20, 59) for day in days]),
        }
    )
    prices = pl.concat(
        [daily("OLD", days, [100.0, 100.0]), daily("NEW", days, [100.0, 100.0])]
    )

    backfill_minutes(
        (client,),
        prices,
        history,
        {2024: ("OLD",), 2025: ("NEW",)},
        tmp_path,
        date(2026, 10, 8),
    )

    old = pl.read_parquet(intraday_path(tmp_path, "OLD", 2025))
    new = pl.read_parquet(intraday_path(tmp_path, "NEW", 2025))
    assert old["start"].dt.date().to_list() == days[:1]  # January only
    assert new["start"].dt.date().to_list() == days


def test_the_summary_counts_the_trading_days_that_have_minutes():
    # A provider can lack days: EODHD lacks TSLA's from July 2023 to May 2024.
    days = [date(2018, 3, 5), date(2018, 3, 6), date(2018, 3, 7)]
    one_day = minutes("AAA", [at(days[0], 20, 58), at(days[0], 20, 59)])
    client = FakeMinuteClient({"AAA": one_day})

    _, summary = download_year(
        (client,),
        "AAA",
        2018,
        date(2026, 10, 8),
        daily("AAA", days, [100.0] * 3),
        HISTORY,
    )

    # Two minutes on one of three days, and that day matches its close.
    assert (summary.rows, summary.days, summary.trading_days) == (2, 1, 3)
    assert summary.matching == 1.0


def test_the_summary_gives_the_share_of_stored_days_that_match():
    # Two weeks of February 2018, winter time: the session ends at 21:00 UTC.
    days = [date(2018, 2, 5) + timedelta(d) for d in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]
    closes = [100.0] * 9 + [150.0]  # the last day ends far from its daily close
    client = FakeMinuteClient(
        {"AAA": minutes("AAA", [at(d, 20, 59) for d in days], closes)}
    )

    _, summary = download_year(
        (client,),
        "AAA",
        2018,
        date(2026, 10, 8),
        daily("AAA", days, [100.0] * 10),
        HISTORY,
    )

    assert summary.matching == 0.9  # still enough to keep them


def test_minutes_that_do_not_match_the_daily_closes_are_not_kept():
    day = date(2018, 3, 6)
    client = FakeMinuteClient({"AAA": minutes("AAA", [at(day, 20, 59)], [50.0])})

    stored, summary = download_year(
        (client,), "AAA", 2018, date(2026, 10, 8), daily("AAA", [day], [100.0]), HISTORY
    )

    assert stored.is_empty()
    assert summary.parts == ()


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
    store(tmp_path, "AAA", 2025, [at(day, 20, 59)])

    done = backfill_minutes(
        (client,), prices, history, {2025: ("AAA", "BBB")}, tmp_path, date(2026, 10, 8)
    )

    assert [stored.ticker for stored in done] == ["BBB"]
    assert client.asked == ["BBB"]
    assert pl.read_parquet(intraday_path(tmp_path, "BBB", 2025)).height == 1


def test_minutes_that_break_the_bar_rules_are_dropped_and_counted():
    day = date(2018, 3, 6)
    bars = minutes("AAA", [at(day, 20, 58), at(day, 20, 59)], [100.0, 100.0])
    broken = bars.with_columns(  # the first minute closes above its high
        pl.when(pl.col("start") == at(day, 20, 58))
        .then(101.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    client = FakeMinuteClient({"AAA": broken})

    stored, summary = download_year(
        (client,), "AAA", 2018, date(2026, 10, 8), daily("AAA", [day], [100.0]), HISTORY
    )

    assert stored["start"].to_list() == [at(day, 20, 59)]
    assert summary.dropped == 1


def store(root, ticker, year, starts, closes=None, source="EODHD"):
    """Stored minutes; source=None stores them as before the source column."""
    path = intraday_path(root, ticker, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = minutes(ticker, starts, closes)
    (labelled(frame, source) if source else frame).write_parquet(path)


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


def test_the_feed_reads_files_stored_before_and_after_the_source_column(tmp_path):
    day = date(2025, 3, 4)
    store(tmp_path, "AAA", 2025, [at(day, 14, 30)], source=None)
    store(tmp_path, "BBB", 2025, [at(day, 14, 30)], source="Alpaca")

    bars = list(minute_feed(tmp_path, {2025: ("AAA", "BBB")}, day, day))

    assert [bar.ticker for bar in bars] == ["AAA", "BBB"]


def test_the_feed_keeps_to_the_days_asked_for_and_skips_missing_files(tmp_path):
    days = [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)]
    store(tmp_path, "AAA", 2025, [at(day, 15, 0) for day in days])

    bars = list(minute_feed(tmp_path, {2025: ("AAA", "NONE")}, days[1], days[2]))

    assert [bar.start.date() for bar in bars] == days[1:]


def test_the_feed_gives_a_stock_that_left_the_universe_until_january_ends(tmp_path):
    store(tmp_path, "OLD", 2024, [at(date(2024, 12, 31), 15, 0)])
    store(tmp_path, "NEW", 2025, [at(date(2025, 1, 2), 15, 0)])
    # OLD left the universe at the new year; a file may hold more than January.
    store(
        tmp_path,
        "OLD",
        2025,
        [at(date(2025, 1, 2), 15, 0), at(date(2025, 2, 3), 15, 0)],
    )

    universes = {2024: ("OLD",), 2025: ("NEW",)}
    bars = list(minute_feed(tmp_path, universes, date(2024, 12, 31), date(2025, 2, 3)))

    assert [(bar.start.date(), bar.ticker) for bar in bars] == [
        (date(2024, 12, 31), "OLD"),
        (date(2025, 1, 2), "NEW"),
        (date(2025, 1, 2), "OLD"),
    ]
