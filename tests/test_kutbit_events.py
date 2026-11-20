"""KUTBIT's event study of extreme minutes (research/kutbit/events.py)."""

import importlib.util
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import Day

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_events", HERE / "events.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
from skill import clustered_mean

SWING = 1e-3


def a_day(steps, opens=None, **given) -> Day:
    """A session whose minute log moves are `steps`, each bar opening at the close before
    unless `opens` says otherwise."""
    n = len(steps)
    close = 100 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[100.0], close[:-1]])
    for t, value in (opens or {}).items():
        open_[t] = value
    fields = {
        "open": open_,
        "high": np.maximum(open_, close) * (1 + 1e-5),
        "low": np.minimum(open_, close) * (1 - 1e-5),
        "close": close,
        "volume": np.full(n, 1000.0),
        "seen": np.ones(n, dtype=bool),
        "profile": np.full(n, 1000.0),
        "news": np.zeros(n),
        "swing": np.full(n, SWING),
        "habit": np.zeros(n),
        "beta": 0.5,
        "minutes": n,
        "prev_close": 100.0,
        "vol": 0.02,
        "vix": 20.0,
        "news_before": 0,
        "etf": False,
    }
    fields.update(given)
    return Day(**fields)


# --- the events -------------------------------------------------------------------------


def test_a_fresh_extreme_minute_is_big_traded_and_alone():
    apart = np.zeros(100)
    apart[[10, 25, 60, 99]] = [5.0, -6.0, 4.5, 9.0]
    apart[80] = 3.9
    seen = np.ones(100, dtype=bool)
    # 25 falls in 10's cooldown and restarts it; 3.9 is too small; the last bar never counts
    np.testing.assert_array_equal(ev.fresh(apart, seen, 4, 30), [10, 60])
    after_a_gap = seen.copy()
    after_a_gap[59] = False  # 60's move would sum the minutes of a gap
    np.testing.assert_array_equal(ev.fresh(apart, after_a_gap, 4, 30), [10])
    untraded = seen.copy()
    untraded[10] = False
    np.testing.assert_array_equal(ev.fresh(apart, untraded, 4, 30), [25, 60])
    edge = np.zeros(200)
    edge[[0, 30, 61]] = 5.0  # 30 is 30 minutes after 0; 61 is 31 after 30
    np.testing.assert_array_equal(ev.fresh(edge, np.ones(200, bool), 4, 30), [0, 61])
    chain = np.zeros(200)
    chain[[0, 25, 50, 81]] = [
        4.0,
        5.0,
        5.0,
        -4.0,
    ]  # each of 25 and 50 restarts the wait
    np.testing.assert_array_equal(ev.fresh(chain, np.ones(200, bool), 4, 30), [0, 81])


def test_each_minute_has_its_part_of_the_day():
    minutes = np.array([0, 29, 30, 119, 120, 299, 300, 389])
    assert list(ev.part_of(minutes)) == [
        "open",
        "open",
        "morning",
        "morning",
        "midday",
        "midday",
        "late",
        "late",
    ]


def test_an_event_s_context_and_what_followed_it():
    steps = np.zeros(90)
    steps[40] = 0.006  # six usual moves up
    steps[41:46] = 0.001  # then five minutes more the same way
    close_40 = 100 * math.exp(0.006)
    news = np.zeros(90)
    news[[10, 11, 41]] = [1, 2, 5]  # only minute 11 is within the 30 minutes up to 40
    volume = np.full(90, 1000.0)
    volume[40] = 9999.0
    profile = np.full(90, 1000.0)
    profile[40] = 999.0
    day = a_day(
        steps,
        opens={41: close_40 * math.exp(-0.002)},
        news=news,
        volume=volume,
        profile=profile,
    )
    market = a_day(np.zeros(90), beta=1.0, etf=True)
    found = ev.session_events(day, market)
    np.testing.assert_array_equal(found["minute"], [40])
    assert found["size"][0] == pytest.approx(6.0) and found["sign"][0] == 1
    assert found["bounce"][0] == pytest.approx(-20.0)
    assert found["news"][0] == 2 and found["volume"][0] == pytest.approx(10.0)
    assert found["qqq"][0] == 0
    ret = (
        math.exp(0.007) - 1
    )  # from the next open, 20 bps below the close, five more up
    assert found["apart_5"][0] == pytest.approx(ret * 1e4)
    assert found["own_5"][0] == pytest.approx(ret * 1e4)
    assert found["z_5"][0] == pytest.approx(ret / (math.sqrt(5) * SWING))
    assert found["adverse_5"][0] == pytest.approx(-0.1, abs=1e-6)
    assert math.isfinite(found["apart_30"][0]) and math.isnan(found["apart_60"][0])


def test_a_fall_is_signed_its_way_and_qqq_s_part_is_taken_out():
    steps = np.zeros(90)
    steps[40] = -0.007
    steps[41:46] = -0.001
    market_steps = np.zeros(90)
    market_steps[40] = 0.004  # QQQ jumps four of its usual moves in the same minute
    market_steps[43] = 0.002
    close_40 = 100 * math.exp(-0.007)
    day = a_day(steps, opens={41: close_40 * math.exp(0.002)})  # it bounces 20 bps up
    market = a_day(market_steps, beta=1.0, etf=True)
    found = ev.session_events(day, market)
    assert found["sign"][0] == -1 and found["size"][0] == pytest.approx(9.0)
    assert found["qqq"][0] == pytest.approx(4.0)
    assert found["bounce"][0] == pytest.approx(-20.0)  # against the fall, so negative
    own = math.exp(-0.007) - 1  # from the bounced open, five minutes down
    of_market = math.exp(0.002) - 1
    assert found["own_5"][0] == pytest.approx(-own * 1e4)
    assert found["apart_5"][0] == pytest.approx(-(own - 0.5 * of_market) * 1e4)
    assert found["adverse_5"][0] == pytest.approx(
        -0.1, abs=1e-6
    )  # a short's worst is the high


def test_a_quiet_session_has_no_events():
    found = ev.session_events(a_day(np.zeros(60)), a_day(np.zeros(60), etf=True))
    assert set(found) == set(ev.KEYS) and all(len(v) == 0 for v in found.values())


# --- a small data folder ----------------------------------------------------------------

STOCKS = tuple(f"S{i:02d}" for i in range(6))
MINUTES = 90
FIELDS = ("open", "high", "low", "close", "volume", "profile", "swing", "habit")


def store(tmp_path, days: int, after: float, seed: int = 4, unusable=()):
    """2016's first `days` sessions of QQQ, six stocks and an ETF. Each stock's session, and
    the ETF's, has one extreme minute of eight usual moves, up or down, between minutes 10
    and 50, followed for 15 minutes by `after` usual moves a minute its own way (a reversal
    when negative). `unusable` names (ticker, day number) pairs."""
    rng = np.random.default_rng(seed)
    rows, fields, row = [], {f: [] for f in (*FIELDS, "seen", "news")}, 0
    for i in range(days):
        day = date(2016, 1, 4) + timedelta(days=i)
        for ticker in ("QQQ", *STOCKS, "XLF"):
            steps = rng.normal(0, 2e-4, MINUTES)
            if ticker != "QQQ":
                t, side = int(rng.integers(10, 51)), rng.choice([-1.0, 1.0])
                steps[t] = side * 8 * SWING
                steps[t + 1 : t + 16] += side * after * SWING
            d = a_day(steps, beta=0.0)
            for f in FIELDS:
                fields[f].append(
                    np.pad(getattr(d, f), (0, 390 - MINUTES), mode="edge").astype(
                        np.float32
                    )
                )
            fields["seen"].append(np.ones(390, dtype=bool))
            fields["news"].append(np.zeros(390, dtype=np.uint16))
            rows.append(
                {
                    "day": day,
                    "year": 2016,
                    "row": row,
                    "ticker": ticker,
                    "etf": ticker in ("QQQ", "XLF"),
                    "minutes": MINUTES,
                    "prev_close": 100.0,
                    "vol": 0.02,
                    "beta": 0.0,
                    "vix": 20.0,
                    "news_before": 0,
                    "split": 1.0,
                    "usable": (ticker, i) not in unusable,
                }
            )
            row += 1
    folder = tmp_path / "kutbit"
    folder.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(folder / "days.parquet")
    np.savez_compressed(
        folder / "2016.npz", **{f: np.stack(v) for f, v in fields.items()}
    )
    return tmp_path


def test_only_stocks_on_days_qqq_is_usable_too_give_events(tmp_path):
    data = store(tmp_path, 3, 0.0, unusable=(("S02", 1), ("QQQ", 2)))
    days, arrays = ev.load(data, (2016,))
    frame = ev.collect(days, arrays, (2016,))
    assert set(frame["ticker"]) == set(STOCKS)  # never QQQ or the ETF
    first, second = (date(2016, 1, 4) + timedelta(days=i) for i in (0, 1))
    assert frame["day"].to_list() == [first.toordinal()] * 6 + [second.toordinal()] * 5
    assert "S02" not in frame.filter(pl.col("day") == second.toordinal())["ticker"]
    assert frame["size"].min() >= 4 and frame["year"].unique().to_list() == [2016]
    assert set(frame["part"]) <= {"open", "morning"}
    with pytest.raises(ValueError, match="2016-2018 only"):
        ev.collect(days, arrays, (2018, 2019))
    assert ev.collect(days, arrays, (2016, 2018)).equals(frame)  # 2018 may be read


@pytest.mark.parametrize("after, sign", [(0.6, 1), (-0.6, -1)])
def test_a_planted_continuation_or_reversal_is_found(tmp_path, after, sign):
    found = ev.study(store(tmp_path, 40, after), (2016,), log=lambda s: None)
    fifteen = found["apart"]["all"]["all"][15]
    assert sign * fifteen["mean"] > 0 and sign * fifteen["t"] > 5
    assert found["events"] == 40 * 6


def test_without_a_pattern_nothing_is_found(tmp_path):
    found = ev.study(store(tmp_path, 40, 0.0, seed=8), (2016,), log=lambda s: None)
    for h in ev.HORIZONS:
        assert abs(found["apart"]["all"]["all"][h]["t"]) < 3


# --- the measures -----------------------------------------------------------------------


def a_frame():
    nan = float("nan")
    columns = {
        "day": [1, 1, 2, 3, 3],
        "minute": [5, 40, 150, 320, 330],
        "part": ["open", "morning", "midday", "late", "late"],
        "size": [4.5, 6.0, 9.0, 13.0, 5.0],  # 6 sits on a bin's edge
        "sign": [1.0, -1.0, 1.0, -1.0, 1.0],
        "volume": [1.0, 3.0, 6.0, 1.5, 2.0],
        "news": [0.0, 1.0, 0.0, 2.0, 0.0],
        "qqq": [0.5, -3.5, 1.0, 3.0, 0.0],
    }
    for h in ev.HORIZONS:
        columns[f"apart_{h}"] = [10.0, -4.0, 6.0, 2.0, nan if h == 60 else -8.0]
        columns[f"own_{h}"] = [12.0, -6.0, 6.0, 5.0, nan if h == 60 else -9.0]
        columns[f"adverse_{h}"] = [-1.0, -2.0, -3.0, -4.0, -50.0]
    return pl.DataFrame(columns)


def test_events_are_measured_by_group_with_a_t_clustered_by_day():
    frame = a_frame()
    found = ev.measure(frame)
    day = frame["day"].to_numpy()
    values = frame["apart_5"].to_numpy()
    expected = clustered_mean(values, day)
    assert found["all"]["all"][5]["mean"] == pytest.approx(expected["mean"])
    assert found["all"]["all"][5]["t"] == pytest.approx(expected["t"])
    assert found["all"]["all"][5]["continued"] == pytest.approx(3 / 5)
    assert found["all"]["all"][60]["n"] == 4  # the one cut by the close is left out
    assert list(found["size"]) == ["4-6", "6-8", "8-12", "12+"]
    assert found["size"]["4-6"]["events"] == 2
    assert found["size"]["4-6"][5]["mean"] == pytest.approx(1.0)
    assert list(found["volume"]) == ["under 2x", "2-5x", "5x+"]
    assert [found["volume"][g]["events"] for g in found["volume"]] == [2, 2, 1]
    assert found["headline"]["a headline"]["events"] == 2
    assert found["QQQ"]["moved too"]["events"] == 2  # |qqq| of 3 and more
    assert list(found["part"]) == list(ev.PARTS)
    assert found["direction"]["down"][5]["mean"] == pytest.approx(-1.0)


def test_how_far_it_went_against_a_follower():
    found = ev.adverse(a_frame())
    assert found["4-6"][5]["median"] == pytest.approx(-25.5)
    assert found["4-6"][5]["worst tenth"] == pytest.approx(
        np.quantile([-1.0, -50.0], 0.1)
    )
    assert found["12+"][1]["median"] == pytest.approx(-4.0)


def test_the_news_rule_follows_with_a_headline_and_bets_against_without():
    frame = a_frame()
    found = ev.follow_the_news(frame)
    side = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
    expected = clustered_mean(
        side * frame["apart_5"].to_numpy(), frame["day"].to_numpy()
    )
    assert found[5]["mean"] == pytest.approx(expected["mean"])


def test_what_the_hedge_takes_out_by_kind_of_event():
    found = ev.hedge_effect(a_frame())
    own = np.array([12.0, -6.0, 6.0, 5.0, -9.0])
    apart = np.array([10.0, -4.0, 6.0, 2.0, -8.0])
    assert found["all"][5]["own"] == pytest.approx(own.std())
    assert found["all"][5]["removed"] == pytest.approx(1 - apart.var() / own.var())
    assert found["all"][60]["own"] == pytest.approx(own[:4].std())  # the cut one is out
    moved = found["QQQ moved too"][5]  # the second and fourth events
    assert moved["own"] == pytest.approx(5.5) and moved["apart"] == pytest.approx(3.0)
    assert moved["removed"] == pytest.approx(1 - 9 / 30.25)
    assert found["at the open"] == {}  # one event is too few for a spread


def test_betting_against_events_without_a_headline():
    frame = a_frame()
    found = ev.fades(frame)
    day = frame["day"].to_numpy()
    calm = np.array([True, False, True, False, True])  # no headline
    expected = clustered_mean(-frame["apart_5"].to_numpy()[calm], day[calm])
    assert found["no headline"][5]["mean"] == pytest.approx(expected["mean"])
    assert found["no headline"][5]["mean"] == pytest.approx(-8 / 3)
    assert found["at the open, no headline"][5]["n"] == 1
    assert found["QQQ moved, no headline"][5]["n"] == 0  # both had a headline
    either = found["either, no headline"][5]  # only the first event, at the open
    assert either["n"] == 1 and math.isnan(either["mean"])  # one event gives no t


def test_the_report_shows_every_table(tmp_path):
    text = ev.report(ev.study(store(tmp_path, 4, 0.0), (2016,), log=lambda s: None))
    assert "2016-2016; 2019-2026 not read" in text
    for title in (
        "All events: the move apart from QQQ",
        "By size, in usual moves",
        "By a headline naming the stock in the 30 minutes up to it",
        "By QQQ's own move that minute",
        "How far it went against a position that followed",
        "A rule to beat",
        "What the hedge takes out",
        "Betting against it, apart from QQQ",
    ):
        assert title in text
