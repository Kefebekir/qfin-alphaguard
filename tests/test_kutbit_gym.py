"""KUTBIT's event gym for the first agent (research/kutbit/gym.py)."""

import importlib.util
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, Day, channels, scalars

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_gym", HERE / "gym.py")
gym = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gym)
import events as ev

SWING = 1e-3
Action = gym.Action


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


# --- the actions ------------------------------------------------------------------------


def test_the_actions_are_every_side_size_hold_and_hedge_within_the_cap():
    assert len(gym.ACTIONS) == 1 + 2 * 5 * 3 * 3 == 91
    assert gym.ACTIONS[0] == Action(0.0, 0.0, 0, 0.0)
    assert len(set(gym.ACTIONS)) == 91
    two = gym.actions(2.0)
    assert len(two) == 1 + 2 * 4 * 3 * 3 and max(a.size for a in two) == 2.0
    assert gym.index_of(Action(-1.0, 1.0, 15, 1.0)) > 0
    for bad in (
        lambda: Action(0.5, 1.0, 15, 1.0),
        lambda: Action(1.0, 3.0, 15, 1.0),
        lambda: Action(1.0, 1.0, 10, 1.0),
        lambda: Action(1.0, 1.0, 15, 0.25),
        lambda: Action(0.0, 1.0, 0, 0.0),
    ):
        with pytest.raises(ValueError):
            bad()


# --- the events -------------------------------------------------------------------------


def an_event(side: float):
    """A session of 90 minutes with an extreme minute at 40, `side` up or down, QQQ moving
    two of its usual moves the other way, headlines before it and a gap at the open."""
    steps = np.zeros(90)
    steps[40] = side * 0.006
    steps[41:46] = side * 0.001
    market_steps = np.zeros(90)
    market_steps[40] = -side * 0.002
    market_steps[42] = 0.0015  # QQQ rises during the hold, whichever way the stock went
    news = np.zeros(90)
    news[[10, 11, 30]] = [5, 1, 2]  # minute 10 is 30 minutes before 40, so outside
    habit = np.full(90, 2e-4)
    day = a_day(steps, news=news, habit=habit, prev_close=99.0, beta=1.2)
    market = a_day(market_steps, beta=1.0, etf=True)
    return day, market


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_what_the_agent_sees_is_turned_to_the_move_s_direction(side):
    day, market = an_event(side)
    found = gym.session(day, market)
    np.testing.assert_array_equal(found["minute"], [40])
    rows, values = channels(day, market), scalars(day)

    def channel(name):
        return rows[40, CHANNELS.index(name)]

    def scalar(name):
        return values[40, SCALARS.index(name)]

    place = scalar("place")
    expected = [
        abs(channel("apart")),
        side,
        channel("range"),
        channel("volume"),
        side * channel("market"),
        math.log1p(3),  # the headlines of minutes 11 and 30
        side * channel("vwap"),
        scalar("time"),
        side * scalar("gap"),
        side * scalar("since_open"),
        place if side > 0 else 1 - place,
        scalar("day_vol"),
        scalar("vix"),
        1.2,
        scalar("news_before"),
        side * scalar("habit"),
    ]
    np.testing.assert_allclose(found["state"][0], expected, rtol=1e-6, atol=1e-6)
    state = dict(zip(gym.STATE, found["state"][0], strict=True))
    assert state["since_open"] > 0 and state["place"] == pytest.approx(1.0, abs=0.01)
    assert state["qqq"] == pytest.approx(-2.0)  # QQQ went the other way
    assert state["size"] == pytest.approx(6 + 1.2 * 2)  # apart from QQQ's opposite move


def test_every_hold_s_outcome_is_signed_by_the_move():
    day, market = an_event(-1.0)
    found = gym.session(day, market)
    for i, h in enumerate(gym.HOLDS):
        own = day.close[40 + h] / day.open[41] - 1
        qqq = market.close[40 + h] / market.open[41] - 1
        assert found["own"][0, i] == pytest.approx(-own)
        assert found["qqq"][0, i] == pytest.approx(-qqq)
    assert found["own"][0, 0] > 0  # the fall went on for five minutes
    assert found["qqq"][0, 0] < 0  # QQQ rose, against a fall's direction


def test_an_outcome_is_known_once_the_longest_hold_or_the_session_has_ended():
    day, market = an_event(1.0)
    assert list(gym.session(day, market)["known"]) == [40 + max(gym.HOLDS)]
    steps = np.zeros(50)
    steps[40] = 0.006
    late = gym.session(a_day(steps), a_day(np.zeros(50), etf=True))
    assert list(late["known"]) == [49]  # the session's last bar


def test_a_hold_past_the_close_ends_at_the_last_bar():
    steps = np.zeros(50)
    steps[40] = 0.006
    steps[41:] = 0.001
    day, market = a_day(steps), a_day(np.zeros(50), etf=True)
    found = gym.session(day, market)
    last = day.close[49] / day.open[41] - 1
    assert found["own"][0, 1] == pytest.approx(last)  # 15 minutes would pass the close
    assert found["own"][0, 2] == pytest.approx(last)


def test_the_chart_around_an_event():
    day, market = an_event(1.0)
    chart = gym.session(day, market)["chart"][0]
    assert chart[gym.CHART] == 0
    assert chart[gym.CHART + 5] == pytest.approx(
        math.log(day.close[45] / day.close[40])
    )
    assert chart[0] == pytest.approx(math.log(day.close[10] / day.close[40]))
    steps = np.zeros(60)
    steps[10] = 0.006
    early = gym.session(a_day(steps), a_day(np.zeros(60), etf=True))["chart"][0]
    assert (
        np.isnan(early[: gym.CHART - 10]).all()
        and np.isfinite(early[gym.CHART - 10 :]).all()
    )


def test_each_event_has_its_kind():
    kind = gym.kind_of(
        np.array([5, 100, 29, 30, 200]),
        np.array([-3.5, 0.0, 3.0, 2.9, 0.0]),
        np.array([2.0, 0.0, 0.0, 0.0, 1.0]),
    )
    names = [gym.KINDS[k] for k in kind]
    assert names == [
        "at the open, QQQ moved, a headline",
        "later, QQQ quiet, no headline",
        "at the open, QQQ moved, no headline",
        "later, QQQ quiet, no headline",
        "later, QQQ quiet, a headline",
    ]


# --- outcomes, choices and their worth ----------------------------------------------------


def one_event(beta=1.2):
    return {
        "own": np.array([[0.01, 0.02, -0.005]]),
        "qqq": np.array([[0.004, -0.002, 0.003]]),
        "beta": np.array([beta]),
    }


def test_every_action_s_return_by_hand():
    data = one_event()
    r = gym.returns(data)[0]
    assert r[0] == 0
    j = gym.index_of(Action(1.0, 2.0, 15, 0.5))
    assert r[j] == pytest.approx(
        2 * (0.02 + 0.5 * 1.2 * 0.002) - (2 * 5e-4 + 0.5 * 1.2 * 2 * 0.5e-4)
    )
    j = gym.index_of(Action(-1.0, 0.25, 5, 0.0))
    assert r[j] == pytest.approx(-0.25 * 0.01 - 0.25 * 5e-4)
    j = gym.index_of(Action(-1.0, 4.0, 30, 1.0))
    assert r[j] == pytest.approx(
        4 * (0.005 + 1.2 * 0.003) - (4 * 5e-4 + 1.2 * 4 * 0.5e-4)
    )
    cheap = gym.returns(data, cost=2e-4)[0]
    j = gym.index_of(Action(-1.0, 0.25, 5, 0.0))
    assert cheap[j] == pytest.approx(-0.25 * 0.01 - 0.25 * 2e-4)


def test_the_market_part_left_unhedged_is_limited_by_m():
    for beta in (1.2, -1.2):
        ok = dict(zip(gym.ACTIONS, gym.allowed(one_event(beta))[0], strict=True))
        assert ok[gym.ACTIONS[0]]
        assert not ok[Action(1.0, 4.0, 15, 0.0)]  # 4.8x of QQQ's luck
        assert not ok[Action(-1.0, 4.0, 15, 0.5)]  # 2.4x
        assert ok[Action(1.0, 4.0, 15, 1.0)]
        assert not ok[Action(1.0, 2.0, 5, 0.0)]  # 2.4x
        assert ok[Action(1.0, 2.0, 5, 0.5)] and ok[Action(-1.0, 1.0, 30, 0.0)]
    small = gym.allowed(one_event(), cap=2.0)[0]
    assert not small[gym.index_of(Action(1.0, 4.0, 15, 1.0))]


def test_utility_and_its_certainty_equivalent():
    assert gym.utility(0.1) == pytest.approx(1 - 1 / 1.1)
    assert gym.utility(0.1, gamma=1) == pytest.approx(math.log(1.1))
    assert gym.utility(-1.0) == -np.inf and gym.utility(-1.5, gamma=1) == -np.inf
    r = np.full(4, 0.01)
    assert gym.certainty(gym.utility(r).mean()) == pytest.approx(0.01)
    assert gym.certainty(gym.utility(r, 1).mean(), 1) == pytest.approx(0.01)
    swing = np.array([0.1, -0.1])
    assert gym.certainty(gym.utility(swing).mean()) == pytest.approx(0.99 - 1)
    assert gym.certainty(-np.inf) == -1.0


# --- the bars -----------------------------------------------------------------------------


def test_the_fixed_bars():
    data = {"kind": np.array([0, 1, 4, 5], dtype=np.uint8)}
    permitted = np.ones((4, len(gym.ACTIONS)), dtype=bool)
    j = gym.index_of(gym.AGAINST)
    found = gym.bars(data, permitted)
    assert list(found["nothing"]) == [0, 0, 0, 0]
    assert list(found["against every event"]) == [j, j, j, j]
    assert list(found["against, no headline"]) == [j, 0, j, 0]
    assert list(found["against, no headline, at the open"]) == [0, 0, j, 0]
    permitted[2, j] = False
    assert list(gym.bars(data, permitted)["against, no headline"]) == [j, 0, 0, 0]


def test_the_best_simple_policy_takes_the_best_action_of_each_kind():
    values = np.array(
        [
            [0.0, 0.1, 0.3],
            [0.0, 0.2, 0.1],
            [0.0, 0.5, -0.4],
            [0.0, -0.1, -0.2],
        ]
    )
    kind = np.array([0, 0, 3, 3])
    permitted = np.ones((4, 3), dtype=bool)
    policy = gym.best_simple(values, permitted, kind, least=2)
    assert policy[0] == 2 and policy[3] == 1 and policy[1] == 0  # kind 1 has no events
    assert list(gym.best_simple(values, permitted, kind, least=3)) == [0] * 8  # too few
    assert list(gym.best_simple(values, permitted, kind)) == [0] * 8  # under MIN_KIND
    permitted[0, 2] = False  # counted as letting the move be: 0 + 0.1 against 0.1 + 0.2
    policy = gym.best_simple(values, permitted, kind, least=2)
    assert policy[0] == 1
    chosen = gym.follow(np.array([2, 0, 0, 1, 0, 0, 0, 0]), kind, permitted)
    assert list(chosen) == [
        0,
        2,
        1,
        1,
    ]  # the first cannot take 2, so it lets the move be


# --- the shuffled bar -----------------------------------------------------------------------


def test_shuffled_outcomes_stay_in_their_part_of_the_day_and_leave_their_stock():
    n = 40
    minute = np.where(np.arange(n) < 20, 10, 200).astype(np.int16)
    data = {
        "state": np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 16), np.float32),
        "own": np.arange(n, dtype=float)[:, None] * np.ones((1, 3)),
        "qqq": -np.arange(n, dtype=float)[:, None] * np.ones((1, 3)),
        "beta": np.ones(n),
        "minute": minute,
        "ticker": (np.arange(n) % 5).astype(np.int16),
    }
    out = gym.shuffled(data, np.random.default_rng(3))
    source = out["own"][:, 0].astype(int)
    assert (
        out["qqq"][:, 0] == -source
    ).all()  # each event's two outcomes travel together
    assert (minute[source] == minute).all()
    assert (data["ticker"][source] != data["ticker"]).all()
    assert sorted(source) == list(range(n)) and (source != np.arange(n)).any()
    np.testing.assert_array_equal(out["state"], data["state"])
    alone = {**data, "ticker": np.zeros(n, np.int16)}
    with pytest.raises(ValueError, match="own stock"):
        gym.shuffled(alone, np.random.default_rng(3))


# --- a small data folder ----------------------------------------------------------------

STOCKS = tuple(f"S{i:02d}" for i in range(12))
MINUTES = 100
FIELDS = ("open", "high", "low", "close", "volume", "profile", "swing", "habit")


def store(tmp_path, days: int = 3, seed: int = 4, unusable=(("S02", 1), ("QQQ", 2))):
    """2016's first sessions of QQQ, twelve stocks and an ETF; each stock and the ETF have
    one extreme minute of eight usual moves between minutes 10 and 50, followed for 15
    minutes by a third of a usual move a minute the other way."""
    rng = np.random.default_rng(seed)
    rows, fields, row = [], {f: [] for f in (*FIELDS, "seen", "news")}, 0
    for i in range(days):
        day = date(2016, 1, 4) + timedelta(days=i)
        for ticker in ("QQQ", *STOCKS, "XLF"):
            steps = rng.normal(0, 2e-4, MINUTES)
            if ticker != "QQQ":
                t, side = int(rng.integers(10, 51)), rng.choice([-1.0, 1.0])
                steps[t] = side * 8 * SWING
                steps[t + 1 : t + 16] -= side * SWING / 3
            d = a_day(steps, beta=0.9)
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
                    "beta": 0.9,
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


def test_the_gym_finds_the_event_study_s_events(tmp_path):
    days, arrays = gym.load(store(tmp_path), (2016,))
    data = gym.build(days, arrays, (2016,))
    frame = ev.collect(days, arrays, (2016,))
    tickers = sorted(days["ticker"].unique())
    mine = sorted(
        zip(
            data["day"].tolist(),
            data["minute"].tolist(),
            [tickers[t] for t in data["ticker"]],
            strict=True,
        )
    )
    theirs = sorted(
        zip(
            frame["day"].to_list(),
            frame["minute"].to_list(),
            frame["ticker"].to_list(),
            strict=True,
        )
    )
    assert mine == theirs and len(mine) == 12 + 11  # no ETF, no day without QQQ
    order = np.lexsort((data["ticker"], data["minute"], data["day"]))
    sizes = data["state"][order, gym.STATE.index("size")]
    np.testing.assert_allclose(
        sizes, frame.sort("day", "minute", "ticker")["size"], rtol=1e-6
    )
    whole = np.isfinite(frame.sort("day", "minute", "ticker")["own_15"].to_numpy())
    np.testing.assert_allclose(
        data["own"][order, 1][whole] * 1e4,
        frame.sort("day", "minute", "ticker")["own_15"].to_numpy()[whole],
    )
    assert (data["own"][:, 1] < 0).mean() > 0.9  # the planted moves reverse
    with pytest.raises(ValueError, match="2022-2026"):
        gym.build(days, arrays, (2021, 2022))


def test_random_walk_events_come_from_the_skill_test_s_walk(tmp_path):
    days, arrays = gym.load(store(tmp_path, days=6, unusable=()), (2016,))
    walk = gym.build(days, arrays, (2016,), "random")
    again = gym.build(days, arrays, (2016,), "random")
    real = gym.build(days, arrays, (2016,))
    assert len(walk["day"]) > 0
    for k in gym.FIELDS:
        np.testing.assert_array_equal(walk[k], again[k])
    assert (walk["own"][:, 1] < 0).mean() < 0.8  # nothing planted survives the walk
    assert len(real["day"]) == 6 * 12
    with pytest.raises(ValueError, match="source"):
        gym.build(days, arrays, (2016,), "shuffled")


def test_events_are_saved_a_year_a_file_and_loaded_back(tmp_path):
    days, arrays = gym.load(store(tmp_path), (2016,))
    data = gym.build(days, arrays, (2016,))
    folder = tmp_path / "agent" / "real"
    gym.save(folder, data, sorted(days["ticker"].unique()))
    assert (folder / "2016.npz").exists() and (folder / "tickers.json").exists()
    back = gym.load_events(folder, (2016,))
    for k in gym.FIELDS:
        np.testing.assert_array_equal(back[k], data[k])
    with pytest.raises(ValueError, match="2022-2026"):
        gym.load_events(folder, (2022,))


def test_the_learning_report(tmp_path):
    days, arrays = gym.load(store(tmp_path), (2016,))
    data = gym.build(days, arrays, (2016,))
    found = gym.summary(data)
    assert found["bars"]["nothing"] == (0.0, 0)
    assert found["bars"]["against every event"][1] == len(data["day"])
    assert found["bars"]["against every event"][0] > 0  # the planted reversals pay
    text = gym.report(data, 1.0)
    assert "2016-2016; 2019-2026 not read" in text
    assert (
        "At 5 bps a round trip" in text
        and "the best simple policy's action by kind" in text
    )
