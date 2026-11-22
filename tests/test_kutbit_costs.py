"""What a round trip costs in KUTBIT's instruments (research/kutbit/costs.py)."""

import importlib.util
import io
import json
import sys
import urllib.error
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_costs", HERE / "costs.py")
costs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(costs)
import gym

START = datetime(2016, 3, 15, 15, 0, tzinfo=UTC)


def around(spread: float, mid: float = 100.0) -> tuple[float, float]:
    """A bid and an ask `spread` bps apart about `mid`."""
    half = mid * spread / 2e4
    return mid - half, mid + half


def quote(seconds: float, bid: float, ask: float, sizes=(5, 3), start=START) -> dict:
    """A quote as Alpaca gives it, `seconds` after `start`, to the nanosecond."""
    at = start + timedelta(seconds=seconds)
    return {
        "t": at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z",
        "bp": bid,
        "ap": ask,
        "bs": sizes[0],
        "as": sizes[1],
    }


def fake_alpaca(quotes_of: dict, page_size: int = 1000, asked: list | None = None):
    """Answers as Alpaca's quotes do, from made-up quotes, `page_size` quotes a page."""

    def get(params: dict) -> dict:
        if asked is not None:
            asked.append(dict(params))
        begin = np.datetime64(params["start"].rstrip("Z"), "ns")
        end = np.datetime64(params["end"].rstrip("Z"), "ns")
        rows = [
            (ticker, q)
            for ticker in params["symbols"].split(",")
            for q in quotes_of.get(ticker, [])
            if begin <= np.datetime64(q["t"].rstrip("Z"), "ns") <= end
        ]
        first = int(params.get("page_token", 0))
        page = {"quotes": {}}
        for ticker, q in rows[first : first + page_size]:
            page["quotes"].setdefault(ticker, []).append(q)
        if first + page_size < len(rows):
            page["next_page_token"] = str(first + page_size)
        return page

    return get


# --- one moment's quotes ------------------------------------------------------------------------


def test_a_spread_is_in_bps_of_the_mid():
    assert costs.spread_bps(99.99, 100.01) == pytest.approx(2.0)
    assert costs.spread_bps(13.41, 13.42) == pytest.approx(7.4543, abs=1e-4)
    assert costs.spread_bps(*around(3.7, 412.0)) == pytest.approx(3.7)


def test_each_quote_weighs_as_long_as_it_stood_within_the_window():
    quotes = [
        quote(1.5, *around(6)),
        quote(-0.5, *around(50)),  # before the moment
        quote(0.0, *around(2)),
        quote(2.5, *around(50)),  # after the window
    ]
    found = costs.measure(quotes, START, 2)
    assert found["spread"] == pytest.approx((2 * 1.5 + 6 * 0.5) / 2)
    assert found["quotes"] == 2 and found["invalid"] == 0


def test_locked_and_crossed_quotes_are_counted_and_their_time_left_out():
    quotes = [
        quote(0.0, *around(2)),
        quote(0.5, 100.0, 100.0),  # locked
        quote(1.0, *around(4)),
        quote(1.5, 100.01, 100.0),  # crossed
    ]
    found = costs.measure(quotes, START, 2)
    assert found["spread"] == pytest.approx((2 * 0.5 + 4 * 0.5) / 1.0)
    assert found["invalid"] == 2 and found["quotes"] == 4
    assert (
        costs.measure([quote(0.2, 0.0, 100.0), quote(0.4, 100.0, 99.0)], START, 2)
        is None
    )
    assert costs.measure([quote(3.0, *around(2))], START, 2) is None
    assert costs.measure([], START, 2) is None


def test_the_first_valid_quote_gives_the_prices_and_the_sizes():
    quotes = [quote(0.1, 0.0, 100.0, (9, 9)), quote(0.3, 104.90, 104.91, (5, 2))]
    found = costs.measure(quotes, START, 2)
    assert (found["bid"], found["ask"]) == (104.90, 104.91)
    assert (found["bid_size"], found["ask_size"]) == (5, 2)


def test_every_page_is_read():
    quotes = {"AAA": [quote(0.4 * i, *around(2 + i)) for i in range(5)]}
    whole = costs.at_moment(["AAA"], START, fake_alpaca(quotes))
    asked = []
    paged = costs.at_moment(
        ["AAA"], START, fake_alpaca(quotes, page_size=2, asked=asked)
    )
    assert len(asked) == 3
    assert whole["AAA"]["spread"] == pytest.approx(4.0)
    assert paged["AAA"]["spread"] == pytest.approx(4.0) and paged["AAA"]["quotes"] == 5


def test_a_quiet_instrument_is_asked_for_again_over_the_wider_window():
    quotes = {"AAA": [quote(0.5, *around(2))], "BBB": [quote(10.0, *around(8))]}
    asked = []
    found = costs.at_moment(
        ["AAA", "BBB", "CCC"], START, fake_alpaca(quotes, asked=asked)
    )
    assert (
        found["AAA"]["window"] == costs.WINDOW and found["BBB"]["window"] == costs.WIDER
    )
    assert found["BBB"]["spread"] == pytest.approx(8.0)
    assert "CCC" not in found
    assert [a["symbols"] for a in asked] == ["AAA,BBB,CCC", "BBB,CCC"]


def test_the_requests_ask_for_the_tape_s_quotes_in_utc_a_hundred_tickers_at_most(
    monkeypatch,
):
    asked = []
    new_york = START.astimezone(ZoneInfo("America/New_York"))
    costs.at_moment(["AAA"], new_york, fake_alpaca({}, asked=asked))
    assert asked[0]["feed"] == "sip" and asked[0]["sort"] == "asc"
    assert (asked[0]["start"], asked[0]["end"]) == (
        "2016-03-15T15:00:00Z",
        "2016-03-15T15:00:02Z",
    )
    assert asked[1]["end"] == "2016-03-15T15:00:30Z"
    monkeypatch.setattr(costs, "A_REQUEST", 2)
    asked.clear()
    quotes = {t: [quote(0.5, *around(2))] for t in ("AAA", "BBB", "CCC")}
    found = costs.at_moment(
        ["AAA", "BBB", "CCC"], START, fake_alpaca(quotes, asked=asked)
    )
    assert [a["symbols"] for a in asked] == ["AAA,BBB", "CCC"] and len(found) == 3


def test_an_instrument_is_found_under_its_alpaca_ticker_or_its_earlier_one():
    assert costs.symbols("BRK-B") == ("BRK.B",)
    assert costs.symbols("CHK_old") == ("CHK",)
    assert costs.symbols("FB_old") == ("FB",)
    assert costs.symbols("BKNG") == ("BKNG", "PCLN")
    assert costs.symbols("AABA") == ("AABA", "YHOO")
    quotes = {"PCLN": [quote(0.5, *around(3))], "AAPL": [quote(0.5, *around(1))]}
    asked = []
    found = costs.at_moment(["AAPL", "BKNG"], START, fake_alpaca(quotes, asked=asked))
    assert (found["AAPL"]["symbol"], found["BKNG"]["symbol"]) == ("AAPL", "PCLN")
    assert found["BKNG"]["spread"] == pytest.approx(3.0)
    assert asked[0]["symbols"] == "AAPL,BKNG,PCLN"
    both = {"BKNG": [quote(0.5, *around(2))], "PCLN": [quote(0.5, *around(9))]}
    assert (
        costs.at_moment(["BKNG"], START, fake_alpaca(both))["BKNG"]["symbol"] == "BKNG"
    )


def test_alpaca_is_asked_again_while_busy_and_a_refusal_stops_without_the_keys(
    monkeypatch,
):
    calls = []

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def busy_once(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "busy", {}, None)
        return Answer(json.dumps({"quotes": {}}).encode())

    def refused(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(costs.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(costs.urllib.request, "urlopen", busy_once)
    get = costs.alpaca(("the-key-id", "the-secret"))
    assert get({"start": "2016-03-15T15:00:00Z"}) == {"quotes": {}}
    assert len(calls) == 2 and calls[0].get_header("Apca-api-key-id") == "the-key-id"
    monkeypatch.setattr(costs.urllib.request, "urlopen", refused)
    with pytest.raises(RuntimeError, match="HTTP 403") as caught:
        get({"start": "2016-03-15T15:00:00Z"})
    assert "the-secret" not in str(caught.value) and "the-key-id" not in str(
        caught.value
    )
    counted = costs.Counted(fake_alpaca({}))
    costs.at_moment(["AAA"], START, counted)
    assert counted.requests == 2


def test_however_many_threads_ask_the_requests_start_at_most_rate_a_minute(monkeypatch):
    clock, starts = [100.0], []

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def answer(request, timeout):
        starts.append(clock[0])
        return Answer(b'{"quotes": {}}')

    monkeypatch.setattr(costs.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        costs.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)
    )
    monkeypatch.setattr(costs.urllib.request, "urlopen", answer)
    get = costs.alpaca(("id", "secret"), rate=120)
    for _ in range(3):
        get({"start": "2016-03-15T15:00:00Z"})
    assert starts == pytest.approx([100.0, 100.5, 101.0])
    clock[0] += 10  # a pause in the asking does not bank turns
    get({"start": "2016-03-15T15:00:00Z"})
    get({"start": "2016-03-15T15:00:00Z"})
    assert starts[-2:] == pytest.approx([111.0, 111.5])


def test_the_market_is_quiet_only_well_away_from_a_session():
    def at(day, hour, minute):
        return datetime(*day, hour, minute, tzinfo=UTC)

    # 15 March 2016: open 13:30 UTC, close 20:00 UTC
    assert not costs.market_is_quiet(at((2016, 3, 15), 15, 0))
    assert not costs.market_is_quiet(at((2016, 3, 15), 12, 50))
    assert costs.market_is_quiet(at((2016, 3, 15), 12, 40))
    assert not costs.market_is_quiet(at((2016, 3, 15), 20, 40))
    assert costs.market_is_quiet(at((2016, 3, 15), 20, 50))
    assert costs.market_is_quiet(at((2026, 11, 22), 15, 0))  # a Sunday


# --- what is measured, and when ------------------------------------------------------------------


def test_two_full_sessions_are_drawn_from_every_month_the_same_way_each_time():
    chosen = costs.drawn_sessions()
    days = [s.day for s in chosen]
    assert len(chosen) == 144 and all(s.minutes == 390 for s in chosen)
    assert all(
        s.minutes == 390 for s in costs.drawn_sessions(a_month=15)
    )  # no half day
    months = Counter((d.year, d.month) for d in days)
    assert len(months) == 72 and set(months.values()) == {2}
    assert days == sorted(days) and min(days).year == 2016 and max(days).year == 2021
    assert days == [s.day for s in costs.drawn_sessions()]
    assert days != [s.day for s in costs.drawn_sessions(seed=1)]
    with pytest.raises(ValueError, match="2022-2026"):
        costs.drawn_sessions(years=(2021, 2022))


def test_the_moments_run_every_ten_minutes_from_09_31_and_at_15_59():
    assert len(costs.MOMENTS) == 40 and len(set(costs.MOMENTS)) == 40
    assert costs.MOMENTS[:3] == (1, 11, 21) and costs.MOMENTS[-2:] == (381, 389)
    assert all(b - a == 10 for a, b in zip(costs.MOMENTS[:-2], costs.MOMENTS[1:-1]))


def test_the_agent_enters_at_the_next_minute_s_open_and_leaves_after_its_hold():
    s = costs.sessions(date(2016, 3, 15), date(2016, 3, 15))[0]
    enter, leave = costs.entry_and_exit(s, 10)
    assert enter == s.open + timedelta(minutes=11)
    assert leave == enter + timedelta(minutes=costs.HOLD)
    assert costs.entry_and_exit(s, 370)[1] == s.close - timedelta(minutes=1)
    enter, leave = costs.entry_and_exit(s, 388)
    assert enter == leave == s.close - timedelta(minutes=1)


def made_up_gym(folder: Path, years=range(2016, 2022)) -> None:
    """Gym events of the first sessions of each January: three of kind 0 and one of every
    other kind in each, of two stocks."""
    rng = np.random.default_rng(3)
    day, minute, kind = [], [], []
    for year in years:
        for s in costs.sessions(date(year, 1, 1), date(year, 1, 31))[:4]:
            for k in range(len(gym.KINDS)):
                for _ in range(3 if k == 0 else 1):
                    day.append(s.day.toordinal())
                    minute.append(int(rng.integers(0, 389)))
                    kind.append(k)
    n = len(day)
    gym.save(
        folder,
        {
            "state": np.zeros((n, len(gym.STATE)), np.float32),
            "own": np.zeros((n, len(gym.HOLDS))),
            "qqq": np.zeros((n, len(gym.HOLDS))),
            "beta": np.ones(n),
            "kind": np.array(kind, np.uint8),
            "day": np.array(day, np.int32),
            "minute": np.array(minute, np.int16),
            "known": np.minimum(np.array(minute) + 30, 389).astype(np.int16),
            "ticker": (np.arange(n) % 2).astype(np.int16),
            "chart": np.zeros((n, 2 * gym.CHART + 1), np.float32),
        },
        ["AAPL", "BKNG"],
    )


def test_events_are_drawn_up_to_per_kind_of_each_kind_and_never_after_2021(tmp_path):
    made_up_gym(tmp_path / "kutbit" / "agent" / "real", years=range(2016, 2023))
    drawn = costs.drawn_events(tmp_path, per_kind=30)
    counts = Counter(drawn["kind"].to_list())
    assert counts[0] == 30 and all(counts[k] == 24 for k in range(1, len(gym.KINDS)))
    assert drawn["day"].is_sorted() and drawn["day"].max().year == 2021
    assert set(drawn["ticker"]) == {"AAPL", "BKNG"}
    assert drawn.equals(costs.drawn_events(tmp_path, per_kind=30))
    assert costs.drawn_events(tmp_path, per_kind=100).height == 6 * 4 * 10


# --- the downloads -----------------------------------------------------------------------------


SPREADS = {"AAPL": 1.0, "PCLN": 3.0, "XLF": 4.5}


def made_up_days(data: Path, chosen: list) -> None:
    """days.parquet for the chosen sessions: AAPL and BKNG on each, XLF from the second on,
    and the first not usable for the features."""
    rows = [
        {"day": s.day, "ticker": code, "usable": i > 0, "etf": code == "XLF"}
        for i, s in enumerate(chosen)
        for code in ("AAPL", "BKNG", "XLF")
        if code != "XLF" or i > 0
    ]
    (data / "kutbit").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(data / "kutbit" / "days.parquet")


def test_each_session_s_moments_are_written_once_and_a_stopped_download_goes_on(
    tmp_path,
):
    chosen = costs.drawn_sessions()[:2]
    made_up_days(tmp_path, chosen)
    quotes = {
        ticker: [
            quote(0.5, *around(spread), start=s.open + timedelta(minutes=m))
            for s in chosen
            for m in costs.MOMENTS
        ]
        for ticker, spread in SPREADS.items()
    }
    asked = []
    folder = costs.download_moments(
        tmp_path, chosen[:1], fake_alpaca(quotes, asked=asked), log=lambda s: None
    )
    assert [p.name for p in sorted(folder.glob("*"))] == [f"{chosen[0].day}.parquet"]
    starts = sorted({a["start"] for a in asked})
    s = chosen[0]
    assert starts[0] == (s.open + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert starts[-1] == (s.close - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert len(starts) == len(costs.MOMENTS)
    asked.clear()
    costs.download_moments(
        tmp_path, chosen, fake_alpaca(quotes, asked=asked), log=lambda s: None
    )
    assert {a["start"][:10] for a in asked} == {chosen[1].day.isoformat()}
    rows = costs.load(folder, costs.MOMENT_SCHEMA)
    assert rows.height == len(costs.MOMENTS) * (2 + 3)
    means = dict(rows.group_by("ticker").agg(pl.col("spread").mean()).iter_rows())
    assert means == pytest.approx({"AAPL": 1.0, "BKNG": 3.0, "XLF": 4.5})
    assert set(rows.filter(pl.col("ticker") == "BKNG")["symbol"]) == {"PCLN"}
    assert not list(folder.glob("*.part"))


def test_a_download_stops_before_a_session_once_the_market_is_not_quiet(tmp_path):
    chosen = costs.drawn_sessions()[:2]
    made_up_days(tmp_path, chosen)
    asked, answers = [], iter([True, False])
    folder = costs.download_moments(
        tmp_path,
        chosen,
        fake_alpaca({}, asked=asked),
        quiet=lambda: next(answers),
        log=lambda s: None,
    )
    assert [p.name for p in folder.glob("*")] == [f"{chosen[0].day}.parquet"]
    assert {a["start"][:10] for a in asked} == {chosen[0].day.isoformat()}
    events = pl.DataFrame(
        {"day": [chosen[0].day], "minute": [10], "ticker": ["AAPL"], "kind": [0]},
        schema={
            "day": pl.Date,
            "minute": pl.Int64,
            "ticker": pl.String,
            "kind": pl.Int64,
        },
    )
    asked.clear()
    folder = costs.download_events(
        tmp_path,
        events,
        fake_alpaca({}, asked=asked),
        quiet=lambda: False,
        log=lambda s: None,
    )
    assert asked == [] and not list(folder.glob("*"))


def test_each_event_s_entry_and_exit_are_measured_one_file_a_year(tmp_path):
    days = [
        costs.sessions(date(y, 6, 1), date(y, 6, 30))[0] for y in (2016, 2016, 2017)
    ]
    events = pl.DataFrame(
        {
            "day": [s.day for s in days],
            "minute": [10, 370, 100],
            "ticker": ["AAPL", "BKNG", "AAPL"],
            "kind": [0, 4, 1],
        },
        schema={
            "day": pl.Date,
            "minute": pl.Int64,
            "ticker": pl.String,
            "kind": pl.Int64,
        },
    )
    quotes = defaultdict(list)
    for s, minute, code in zip(days, events["minute"], events["ticker"]):
        enter, leave = costs.entry_and_exit(s, minute)
        ticker = "PCLN" if code == "BKNG" else code
        quotes[ticker] += [
            quote(0.5, *around(4), start=enter),
            quote(0.5, *around(2), start=leave),
        ]
    folder = costs.download_events(
        tmp_path, events, fake_alpaca(quotes), log=lambda s: None
    )
    assert sorted(p.name for p in folder.glob("*")) == ["2016.parquet", "2017.parquet"]
    measured = costs.load(folder, costs.EVENT_SCHEMA)
    assert measured["enter_spread"].to_list() == pytest.approx([4.0] * 3)
    assert measured["leave_spread"].to_list() == pytest.approx([2.0] * 3)
    assert measured.filter(pl.col("ticker") == "BKNG")["enter_symbol"].to_list() == [
        "PCLN"
    ]
    asked = []
    costs.download_events(
        tmp_path, events, fake_alpaca(quotes, asked=asked), log=lambda s: None
    )
    assert asked == []


# --- what it shows --------------------------------------------------------------------------------


def moment_row(ticker, minute, prices, sizes=(5, 3), day=date(2016, 3, 15)) -> dict:
    """A measured moment as the download writes it, at the bid and ask `prices`, or one
    without a valid quote if `prices` is None."""
    if prices is None:
        return {"day": day, "minute": minute, "ticker": ticker, **costs.EMPTY}
    bid, ask = prices
    return {
        "day": day,
        "minute": minute,
        "ticker": ticker,
        "spread": costs.spread_bps(bid, ask),
        "bid": bid,
        "ask": ask,
        "bid_size": sizes[0],
        "ask_size": sizes[1],
        "quotes": 4,
        "invalid": 1,
        "symbol": ticker,
        "window": costs.WINDOW,
    }


def made_up_moments() -> pl.DataFrame:
    rows = [
        moment_row(
            "AAPL", 1, (99.985, 100.015), (5, 3)
        ),  # 3 bps; 30,000 USD fits on both sides
        moment_row("AAPL", 101, (99.995, 100.005), (20, 13)),  # 1 bp; 120,000 fits too
        moment_row(
            "AAPL", 381, (99.9875, 100.0125), (5, 2)
        ),  # 2.5 bps; 20,000 on the ask
        moment_row("AAPL", 201, None),
        moment_row("MSFT", 101, (300.0, 300.005), (100, 7)),  # under a cent
        moment_row("XLF", 1, (22.20, 22.21), (3947, 2199)),
        moment_row("XLF", 101, (22.20, 22.21), (3947, 2199), day=date(2017, 3, 15)),
    ]
    return pl.DataFrame(rows, schema=costs.MOMENT_SCHEMA)


def made_up_trips() -> pl.DataFrame:
    rows = []
    for kind, enter, leave in (
        (0, 4.0, 2.0),
        (0, 8.0, 6.0),
        (4, 9.0, 5.0),
        (1, None, 3.0),
    ):
        row = {"day": date(2016, 3, 15), "minute": 50, "ticker": "AAPL", "kind": kind}
        for side, spread in (("enter", enter), ("leave", leave)):
            measured = moment_row(
                "AAPL", 51, None if spread is None else around(spread)
            )
            row |= {f"{side}_{k}": measured[k] for k in costs.COLUMNS}
        rows.append(row)
    return pl.DataFrame(rows, schema=costs.EVENT_SCHEMA)


AAPL_MEAN = (3.0 + 1.0 + 2.5) / 3
XLF_SPREAD = 0.01 / 22.205 * 1e4
MSFT_SPREAD = 0.005 / 300.0025 * 1e4


def test_the_costs_are_summed_up_by_instrument_part_of_day_year_and_event():
    found = costs.summary(made_up_moments(), made_up_trips(), {"XLF"})
    aapl = found["by_instrument"].filter(pl.col("ticker") == "AAPL").row(0, named=True)
    assert aapl["mean"] == pytest.approx(AAPL_MEAN) and aapl["moments"] == 3
    assert aapl["median"] == pytest.approx(2.5) and not aapl["etf"]
    assert [aapl[name] for name, _, _ in costs.PARTS] == pytest.approx([3.0, 1.0, 2.5])
    assert aapl["fits 30000"] == pytest.approx(2 / 3)
    assert aapl["fits 120000"] == pytest.approx(1 / 3)
    xlf = found["by_instrument"].filter(pl.col("ticker") == "XLF").row(0, named=True)
    assert (
        xlf["etf"]
        and xlf["price"] == pytest.approx(22.205)
        and xlf["fits 120000"] == 1.0
    )
    assert found["by_instrument"]["ticker"].to_list() == ["MSFT", "AAPL", "XLF"]
    years = {
        (r["etf"], r["year"]): r["mean"] for r in found["by_year"].iter_rows(named=True)
    }
    assert years == pytest.approx(
        {
            (False, 2016): (3.0 + 1.0 + 2.5 + MSFT_SPREAD) / 4,
            (True, 2016): XLF_SPREAD,
            (True, 2017): XLF_SPREAD,
        }
    )
    parts = {
        (r["etf"], r["part"]): r["mean"] for r in found["by_part"].iter_rows(named=True)
    }
    assert parts[False, "the day between"] == pytest.approx((1.0 + MSFT_SPREAD) / 2)
    kinds = {r["kind"]: r for r in found["by_kind"].iter_rows(named=True)}
    assert set(kinds) == {0, 4}
    assert kinds[0]["mean"] == pytest.approx(5.0) and kinds[0][
        "above"
    ] == pytest.approx(0.5)
    assert kinds[4]["mean"] == pytest.approx(7.0) and kinds[4][
        "usual"
    ] == pytest.approx(AAPL_MEAN)
    checks = found["checks"]
    assert checks["with a valid quote"] == pytest.approx(6 / 7)
    assert checks["quotes locked or crossed"] == pytest.approx(6 / 24)
    assert checks["found only in the wider window"] == 0.0
    assert checks["spreads under a cent above 1 USD"] == pytest.approx(1 / 6)
    assert checks["sizes that are a multiple of LOT"] == pytest.approx(1 / 12)
    assert checks["event trips measured"] == pytest.approx(3 / 4)


def test_the_page_shows_every_part_and_the_verdict():
    text = costs.page(
        costs.summary(made_up_moments(), made_up_trips(), {"XLF"}), "a test"
    )
    for part in (
        "<h1>a test</h1>",
        "the median stock's round trip costs 1.17 bps on average and the median ETF's 4.50",
        "Of 3 instruments, 1 cost less than 1.89 bps, 1 less than 2 bps and 3 less than 5 bps",
        "A round trip by instrument",
        "XLF (ETF)",
        "stocks, the first 30 minutes",
        "By year",
        "At the first agent's extreme minutes",
        "at the open, QQQ quiet, no headline",
        "every kind",
        "How often an order fits",
        "Checks",
    ):
        assert part in text
    empty = costs.summary(
        pl.DataFrame(schema=costs.MOMENT_SCHEMA),
        pl.DataFrame(schema=costs.EVENT_SCHEMA),
        set(),
    )
    assert "Checks" in costs.page(empty, "nothing yet")


def test_the_page_is_written_from_the_downloads(tmp_path):
    chosen = costs.drawn_sessions()[:1]
    made_up_days(tmp_path, chosen * 2)
    base = tmp_path / "kutbit" / "costs" / "trial"
    (base / "moments").mkdir(parents=True)
    made_up_moments().write_parquet(base / "moments" / "2016-03-15.parquet")
    path = costs.write_page(tmp_path, "trial", "the trial")
    assert path == base / "costs.html" and "XLF (ETF)" in path.read_text()
