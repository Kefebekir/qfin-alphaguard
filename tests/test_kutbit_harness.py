"""The harness of KUTBIT's first agent (research/kutbit/harness.py)."""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_harness", HERE / "harness.py")
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)
import gym
from skill import clustered_mean

A = len(gym.ACTIONS)


def synthetic(
    seed: int = 1, edge: float = 0.0, days_a_month: int = 3, events_a_day: int = 5
):
    """Gym-like events of 2016-2021: in each month a few days, in each day a few events of six
    stocks. With `edge`, a move over 6 usual moves reverses by `edge` over every hold and a
    smaller one goes on by half as much, so only an agent that reads the size can tell."""
    rng = np.random.default_rng(seed)
    day, minute, ticker = [], [], []
    for year in range(2016, 2022):
        for month in range(1, 13):
            for d in range(days_a_month):
                ordinal = date(year, month, 1 + 4 * d).toordinal()
                for e in range(events_a_day):
                    day.append(ordinal)
                    minute.append(int(rng.integers(0, 380)))
                    ticker.append(e % 6)
    n = len(day)
    state = rng.normal(0, 1, (n, len(gym.STATE))).astype(np.float32)
    state[:, 0] = 4 + rng.exponential(2.5, n)
    big = state[:, 0] > 6
    pattern = np.where(big, -edge, edge / 2)
    own = rng.normal(0, 0.003, (n, len(gym.HOLDS))) + pattern[:, None]
    minute = np.array(minute, dtype=np.int16)
    return {
        "state": state,
        "own": own,
        "qqq": rng.normal(0, 0.002, (n, len(gym.HOLDS))),
        "beta": np.ones(n),
        "kind": rng.integers(0, 8, n).astype(np.uint8),
        "day": np.array(day, dtype=np.int32),
        "minute": minute,
        "known": np.minimum(minute + 30, 389).astype(np.int16),
        "ticker": np.array(ticker, dtype=np.int16),
        "chart": np.tile(np.linspace(-0.01, 0.01, 2 * gym.CHART + 1), (n, 1)).astype(
            np.float32
        ),
    }


class Nothing:
    def fit(self, history):
        pass

    def update(self, known):
        pass

    def act(self, events):
        return np.zeros(len(events["state"]), dtype=int)


class Spy(Nothing):
    """Records what it is given and when, and every breach of the order of time."""

    def __init__(self):
        self.now, self.seen, self.learned, self.breaches, self.fits = 0, [], [], [], []

    def fit(self, history):
        self.fits.append(history["day"].max() if len(history["day"]) else 0)

    def update(self, known):
        self.learned += list(known["day"].astype(np.int64) * 1000 + known["known"])

    def act(self, events):
        if "utility" in events or "known" in events or "own" in events:
            self.breaches.append("act was given an outcome")
        stamps = events["day"].astype(np.int64) * 1000 + events["minute"]
        if len(set(stamps)) != 1:
            self.breaches.append("one call held events of different minutes")
        self.now = int(stamps[0])
        if self.learned and max(self.learned) > self.now:
            self.breaches.append("an outcome known after this minute was given")
        self.seen += list(stamps)
        return super().act(events)


class Rogue(Nothing):
    def __init__(self, action, length=None):
        self.action, self.length = action, length

    def act(self, events):
        n = len(events["state"]) if self.length is None else self.length
        return np.full(n, self.action)


class TableAgent:
    """A test's agent, not KUTBIT's: the mean utility of each action, for small and big moves
    when it uses the size, from everything it has been told."""

    def __init__(self, setting):
        self.use_size = setting["use_size"]
        self.sums, self.counts = np.zeros((2, A)), np.zeros(2)

    def group(self, state):
        return (
            (state[:, 0] > 6).astype(int)
            if self.use_size
            else np.zeros(len(state), int)
        )

    def add(self, events):
        g = self.group(events["state"])
        worth = np.where(events["allowed"], events["utility"], 0.0)
        for k in (0, 1):
            self.sums[k] += worth[g == k].sum(axis=0)
            self.counts[k] += (g == k).sum()

    def fit(self, history):
        self.sums, self.counts = np.zeros((2, A)), np.zeros(2)
        self.add(history)

    def update(self, known):
        self.add(known)

    def forecast(self, events):
        g = self.group(events["state"])
        return self.sums[g] / np.maximum(self.counts[g], 1)[:, None]

    def act(self, events):
        scored = np.where(events["allowed"], self.forecast(events), -np.inf)
        best = np.argmax(scored, axis=1)
        return np.where(scored[np.arange(len(best)), best] > 0, best, 0)


SETTINGS = [{"use_size": False}, {"use_size": True}]


def ready(data):
    return gym.utility(gym.returns(data)), gym.allowed(data)


# --- the run ----------------------------------------------------------------------------------


def test_an_outcome_is_never_given_before_it_is_known():
    data = synthetic()
    values, permitted = ready(data)
    months = hs.months_of([2019])
    spy = Spy()
    found = hs.run(spy, data, values, permitted, months)
    assert spy.breaches == []
    inside = np.isin(hs.month_key(data["day"]), months)
    assert sorted(spy.learned) == sorted(
        (data["day"][inside].astype(np.int64) * 1000 + data["known"][inside]).tolist()
    )  # every event of the months, each once
    assert spy.seen == sorted(spy.seen)  # decisions in the order of time
    assert (found["choice"][inside] == 0).all() and (
        found["choice"][~inside] == -1
    ).all()
    starts = [date(2019, m, 1).toordinal() for m in range(1, 13)]
    assert all(last < start for last, start in zip(spy.fits, starts, strict=True))


class Clock(Nothing):
    """Records, at each decision, the minutes at whose close the outcomes it has had became
    known."""

    def __init__(self):
        self.known, self.at = [], {}

    def update(self, known):
        self.known += [int(k) for k in known["known"]]

    def act(self, events):
        self.at[int(events["minute"][0])] = sorted(self.known)
        return super().act(events)


def test_an_outcome_known_at_a_minute_s_close_comes_before_that_minute_s_decision():
    data = synthetic()
    day = date(2019, 3, 5).toordinal()
    data["day"][:3] = day
    data["minute"][:3] = [10, 39, 40]
    data["known"][:3] = [40, 69, 70]
    keep = (data["day"] != day) | (np.arange(len(data["day"])) < 3)
    data = gym.subset(data, keep)
    values, permitted = ready(data)
    clock = Clock()
    hs.run(clock, data, values, permitted, [2019 * 12 + 2])
    assert 40 not in clock.at[39]  # the first event's outcome is not known at 39
    assert 40 in clock.at[40]  # but it is at the close of 40, before that decision


class Recorder(Nothing):
    """Keeps every outcome it learns from."""

    def __init__(self):
        self.own, self.qqq, self.utility = [], [], []

    def update(self, known):
        self.own.append(known["own"])
        self.qqq.append(known["qqq"])
        self.utility.append(known["utility"])


def test_what_followed_is_given_to_learn_from_and_the_dealt_bar_learns_the_dealt():
    data = synthetic(days_a_month=6, events_a_day=8)
    values, permitted = ready(data)
    months = hs.months_of([2019])
    inside = np.flatnonzero(np.isin(hs.month_key(data["day"]), months))
    order = inside[
        np.lexsort((data["ticker"][inside], data["known"][inside], data["day"][inside]))
    ]
    mine = Recorder()
    hs.run(mine, data, values, permitted, months)
    np.testing.assert_array_equal(np.concatenate(mine.own), data["own"][order])
    np.testing.assert_array_equal(np.concatenate(mine.qqq), data["qqq"][order])
    dealt = hs.within_months(data, np.random.default_rng(4))
    teach = gym.utility(gym.returns(dealt))
    theirs = Recorder()
    hs.run(theirs, data, teach, permitted, months, outcomes=dealt)
    np.testing.assert_array_equal(np.concatenate(theirs.own), dealt["own"][order])
    np.testing.assert_array_equal(np.concatenate(theirs.utility), teach[order])
    assert not np.array_equal(np.concatenate(theirs.own), data["own"][order])


def test_an_action_the_limits_forbid_or_that_does_not_exist_is_refused():
    data = synthetic()
    values, permitted = ready(data)
    months = hs.months_of([2019])
    forbidden = gym.index_of(gym.Action(1.0, 4.0, 15, 0.0))  # 4x of QQQ's luck, M is 2x
    with pytest.raises(ValueError, match="forbid"):
        hs.run(Rogue(forbidden), data, values, permitted, months)
    with pytest.raises(ValueError, match="does not exist"):
        hs.run(Rogue(A), data, values, permitted, months)
    with pytest.raises(ValueError, match="one action for each"):
        hs.run(Rogue(0, length=99), data, values, permitted, months)
    allowed = gym.index_of(gym.Action(-1.0, 1.0, 15, 1.0))
    found = hs.run(Rogue(allowed), data, values, permitted, months)
    inside = np.isin(hs.month_key(data["day"]), months)
    assert (found["choice"][inside] == allowed).all()


def test_dealt_outcomes_stay_in_their_month_and_part_and_leave_their_stock():
    data = synthetic(days_a_month=6, events_a_day=8)
    n = len(data["day"])
    data["own"] = np.repeat(np.arange(n, dtype=float)[:, None], 3, axis=1)
    data["minute"] = np.where(np.arange(n) % 2 == 0, 10, 200).astype(np.int16)
    dealt = hs.within_months(data, np.random.default_rng(2))
    source = dealt["own"][:, 0].astype(int)
    key = hs.month_key(data["day"])
    assert (key[source] == key).all() and (
        data["minute"][source] == data["minute"]
    ).all()
    assert (data["ticker"][source] != data["ticker"]).all()
    assert sorted(source) == list(range(n))
    np.testing.assert_array_equal(dealt["state"], data["state"])


# --- the bars and the test ------------------------------------------------------------------


def test_the_simple_policy_is_chosen_afresh_each_month_from_the_months_before():
    data = synthetic(days_a_month=6, events_a_day=8, edge=0.01)
    data["kind"][:] = 0
    values, permitted = ready(data)
    months = hs.months_of([2019])
    choice = hs.simple_policy(data, values, permitted, months)
    key = hs.month_key(data["day"])
    for month in months:
        before = key < month
        mine = key == month
        policy = gym.best_simple(
            values[before], permitted[before], data["kind"][before]
        )
        assert (
            choice[mine] == gym.follow(policy, data["kind"][mine], permitted[mine])
        ).all()
    assert (choice[~np.isin(key, months)] == -1).all()
    found = hs.bars(data, values, permitted, months)
    assert set(found) == {*gym.bars(data, permitted), "best simple policy"}


def test_the_test_of_decision_0026():
    rng = np.random.default_rng(3)
    day = np.repeat(np.arange(200), 5)
    mine, theirs = rng.normal(0, 1e-3, (2, len(day)))
    bars = {"nothing": np.zeros(len(day)), "a rule": theirs + 2e-4}
    found = hs.gate(mine + 1e-3, bars, day)
    assert (
        found["passed"] and found["best bar"] == "a rule" and found["margin"]["t"] > 2
    )
    assert not hs.gate(mine + 1e-4, bars, day)["passed"]  # behind the rule
    close = hs.gate(mine + 2.3e-4, bars, day)
    assert all(close["beats"].values()) and close["margin"]["t"] < 2
    assert not close["passed"]  # ahead of every bar, but with a t under 2
    losing = {"nothing": np.zeros(len(day)), "a rule": theirs - 3e-3}
    found = hs.gate(mine - 1e-3, losing, day)
    assert not found["passed"] and found["margin"]["t"] < 0  # behind doing nothing
    below = {"nothing": np.full(len(day), -2e-3)}
    found = hs.gate(mine - 1e-3, below, day)
    assert found["margin"]["t"] > 2 and not found["passed"]  # beats it, but loses money


# --- the three steps --------------------------------------------------------------------------


def test_the_setting_that_did_best_on_2019_is_kept():
    data = synthetic(edge=0.01)
    values, permitted = ready(data)
    found = hs.choose(TableAgent, SETTINGS, data, values, permitted, log=lambda s: None)
    assert found["index"] == 1 and found["settings"] == {"use_size": True}
    assert found["scores"][1] > found["scores"][0] and len(found["scores"]) == 2
    inside = np.isin(hs.month_key(data["day"]), hs.months_of([2019]))
    assert (found["run"]["choice"][~inside] == -1).all()


def test_the_placebo_passes_on_noise_and_fails_where_something_can_be_learned():
    quiet = hs.placebo(TableAgent, SETTINGS, synthetic(seed=4), log=lambda s: None)
    assert quiet["passed"]
    loud = hs.placebo(
        TableAgent, SETTINGS, synthetic(seed=4, edge=0.01), log=lambda s: None
    )
    assert not loud["passed"] and loud["agent"]["t"] >= hs.PLACEBO_T


def test_the_measurement_runs_the_agent_beside_every_bar():
    found = hs.measure(
        TableAgent, {"use_size": True}, synthetic(edge=0.01), log=lambda s: None
    )
    assert found["gate"]["passed"]
    expected = {
        "the agent",
        "nothing",
        "against every event",
        "against, no headline",
        "against, no headline, at the open",
        "best simple policy",
        "itself on dealt outcomes",
    }
    assert set(found["table"]) == expected
    assert set(found["table"]["the agent"]) == {"2 bps", "5 bps", "10 bps"}
    cheap, dear = (
        found["table"]["the agent"]["2 bps"],
        found["table"]["the agent"]["10 bps"],
    )
    assert cheap["certainty"] > dear["certainty"] and cheap["trades"] == dear["trades"]
    traded = found["choice"][found["idx"]] > 0
    assert np.isfinite(found["expected"][found["idx"]][traded]).all()
    dealt = found["bars"]["itself on dealt outcomes"][found["idx"]]
    assert (
        dealt != found["choice"][found["idx"]]
    ).mean() > 0.2  # it cannot read the size
    flat = hs.measure(TableAgent, {"use_size": True}, synthetic(), log=lambda s: None)
    assert not flat["gate"]["passed"]


def test_the_bar_on_dealt_outcomes_learns_from_the_dealt_moves():
    data = synthetic(edge=0.01)
    made = []

    def make(setting):
        made.append(Recorder())
        return made[-1]

    hs.measure(make, {}, data, log=lambda s: None)
    inside = np.flatnonzero(
        np.isin(hs.month_key(data["day"]), hs.months_of(hs.MEASURE_YEARS))
    )
    order = inside[
        np.lexsort((data["ticker"][inside], data["known"][inside], data["day"][inside]))
    ]
    dealt = hs.within_months(data, np.random.default_rng(hs.SEED))
    np.testing.assert_array_equal(np.concatenate(made[0].own), data["own"][order])
    np.testing.assert_array_equal(np.concatenate(made[1].own), dealt["own"][order])
    np.testing.assert_array_equal(np.concatenate(made[1].qqq), dealt["qqq"][order])


def test_the_page_shows_what_the_agent_did():
    found = hs.measure(
        TableAgent, {"use_size": True}, synthetic(edge=0.01), log=lambda s: None
    )
    text = hs.page(
        "a test",
        synthetic(edge=0.01),
        [f"S{i}" for i in range(6)],
        found["idx"],
        found["choice"],
        found["expected"],
        found["values"],
        found["bars"],
        found["gate"],
    )
    for part in (
        "The test: <b>passed</b>",
        "Its worth against the bars",
        "What it did, by kind of event",
        "Where it hedged",
        "Its forecasts against what followed",
        "Where it left the best simple policy",
        "Sample events",
        "<svg",
    ):
        assert part in text


def test_the_commands_run_in_order_and_measure_waits(tmp_path, monkeypatch):
    folder = tmp_path / "kutbit" / "agent"
    tickers = [f"S{i}" for i in range(6)]
    gym.save(folder / "real", synthetic(seed=5, edge=0.01), tickers)
    gym.save(folder / "random", synthetic(seed=6), tickers)
    monkeypatch.setattr(
        hs, "efe_agent", lambda: SimpleNamespace(make=TableAgent, SETTINGS=SETTINGS)
    )
    with pytest.raises(SystemExit, match="placebo first"):
        hs.main(["measure", str(tmp_path)])
    hs.main(["placebo", str(tmp_path)])
    assert json.loads((folder / "placebo.json").read_text())["passed"]
    with pytest.raises(SystemExit, match="choose"):
        hs.main(["measure", str(tmp_path)])
    hs.main(["choose", str(tmp_path)])
    chosen = json.loads((folder / "choice.json").read_text())
    assert (
        chosen["settings"] == {"use_size": True} and (folder / "choice.html").exists()
    )
    hs.main(["measure", str(tmp_path)])
    measured = json.loads((folder / "measure.json").read_text())
    assert measured["gate"]["passed"] and (folder / "agent.html").exists()
    with pytest.raises(SystemExit, match="usage"):
        hs.main(["train", str(tmp_path)])


def test_the_harness_never_reads_past_2021(tmp_path):
    with pytest.raises(ValueError, match="2022-2026"):
        gym.load_events(tmp_path, (2021, 2022))
    assert max(hs.MEASURE_YEARS) == 2021 and hs.CHOOSE_YEAR == 2019
    # the clustered mean the test relies on
    found = clustered_mean(np.array([1.0, 1.0, 3.0]), np.array([0, 0, 1]))
    assert found["n"] == 3
