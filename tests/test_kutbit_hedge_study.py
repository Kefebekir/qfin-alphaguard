"""KUTBIT's hedge study (research/kutbit/hedge_study.py)."""

import importlib.util
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import Day

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location(
    "kutbit_hedge_study", HERE / "hedge_study.py"
)
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)
from labels import targets


def a_day(rng, n=100, **given) -> Day:
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 2e-4, n))
    fields = {
        "open": open_,
        "high": np.maximum(open_, close) * 1.0002,
        "low": np.minimum(open_, close) * 0.9998,
        "close": close,
        "volume": np.full(n, 1000.0),
        "seen": np.ones(n, dtype=bool),
        "profile": np.full(n, 1000.0),
        "news": np.zeros(n),
        "swing": np.full(n, 1e-3),
        "habit": np.zeros(n),
        "beta": 0.8,
        "minutes": n,
        "prev_close": 100.0,
        "vol": 0.02,
        "vix": 20.0,
        "news_before": 0,
        "etf": False,
    }
    fields.update(given)
    return Day(**fields)


# --- the windows ------------------------------------------------------------------------


def test_a_window_is_the_target_over_its_bars():
    rng = np.random.default_rng(1)
    market, day = a_day(rng, beta=1.0), a_day(rng, beta=0.7)
    found = hs.windows(day, market)
    np.testing.assert_array_equal(found["start"], [0, 30, 60])
    expected = targets(day, market, (30,))[30]
    np.testing.assert_allclose(found["z"], expected["z"][[0, 30, 60]])
    np.testing.assert_allclose(found["scale"], math.sqrt(30) * 1e-3)
    of_market = market.close[[30, 60, 90]] / market.open[[1, 31, 61]] - 1
    np.testing.assert_allclose(found["market"], of_market)
    own = day.close[30] / day.open[1] - 1 - 0.7 * of_market[0]
    assert found["z"][0] == pytest.approx(own / (math.sqrt(30) * 1e-3))


def test_only_whole_windows_before_the_close_are_kept():
    rng = np.random.default_rng(2)
    found = hs.windows(a_day(rng, 61), a_day(rng, 61))
    np.testing.assert_array_equal(found["start"], [0, 30])  # 30 + 30 is the last bar
    found = hs.windows(a_day(rng, 60), a_day(rng, 60))
    np.testing.assert_array_equal(found["start"], [0])


def test_each_start_has_its_part_of_the_day():
    assert list(hs.part_of(np.array(hs.STARTS))) == [
        "open",
        *["morning"] * 3,
        *["midday"] * 6,
        *["late"] * 2,
    ]


# --- a small data folder ----------------------------------------------------------------

STOCKS = tuple(f"S{i:02d}" for i in range(12))
MINUTES = 100
DAYS = (date(2016, 3, 1), date(2016, 3, 2), date(2016, 3, 3))
FIELDS = ("open", "high", "low", "close", "volume", "profile", "swing", "habit")


def store(tmp_path, seed: int = 4):
    """2016's first three days of QQQ, twelve stocks and an ETF: on the second day one stock's
    session is unusable, on the third QQQ's."""
    rng = np.random.default_rng(seed)
    rows, fields, row = [], {f: [] for f in (*FIELDS, "seen", "news")}, 0
    for i, day in enumerate(DAYS):
        for ticker in ("QQQ", *STOCKS, "XLF"):
            d = a_day(rng, MINUTES, beta=0.5 + 0.05 * (row % 7))
            for f in FIELDS:
                values = getattr(d, f)
                fields[f].append(
                    np.pad(values, (0, 390 - MINUTES), mode="edge").astype(np.float32)
                )
            fields["seen"].append(np.ones(390, dtype=bool))
            fields["news"].append(np.zeros(390, dtype=np.uint16))
            usable = not (i == 1 and ticker == "S03") and not (
                i == 2 and ticker == "QQQ"
            )
            rows.append(
                {
                    "day": day,
                    "year": 2016,
                    "row": row,
                    "ticker": ticker,
                    "etf": ticker in ("QQQ", "XLF"),
                    "minutes": MINUTES,
                    "prev_close": 100.0,
                    "vol": 0.01 * (i + 1) if ticker == "QQQ" else 0.05,
                    "beta": d.beta,
                    "vix": 20.0,
                    "news_before": 0,
                    "split": 1.0,
                    "usable": usable,
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


def test_only_stocks_on_days_qqq_is_usable_too_give_windows(tmp_path):
    data = store(tmp_path)
    days, arrays = hs.load(data, (2016,))
    frame = hs.collect(days, arrays, (2016,))
    assert set(frame["ticker"]) == set(STOCKS)  # neither QQQ nor the ETF
    by_day = frame.group_by("day").len().sort("day")
    assert by_day["day"].to_list() == [DAYS[0].toordinal(), DAYS[1].toordinal()]
    assert by_day["len"].to_list() == [12 * 3, 11 * 3]  # S03 unusable on the second day
    assert "S03" not in frame.filter(pl.col("day") == DAYS[1].toordinal())["ticker"]
    first = days.filter((pl.col("day") == DAYS[0]) & (pl.col("ticker") == "S05")).row(
        0, named=True
    )
    mine = frame.filter(
        (pl.col("day") == DAYS[0].toordinal()) & (pl.col("ticker") == "S05")
    )
    assert mine["beta"].to_list() == [first["beta"]] * 3
    assert mine["qqq_vol"].to_list() == [0.01] * 3  # QQQ's volatility, not the stock's
    assert mine["start"].to_list() == [0, 30, 60]
    with pytest.raises(ValueError, match="development years"):
        hs.collect(days, arrays, (2021, 2022))
    assert hs.collect(days, arrays, (2016, 2021)).equals(frame)  # 2021 may be read


def test_the_ground_takes_out_each_part_s_mean_once_a_window():
    frame = pl.DataFrame(
        {
            "day": [1, 1, 1, 2, 2],
            "ticker": ["A", "B", "C", "A", "B"],
            "start": [0, 0, 0, 0, 120],
            "beta": [1.0] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 10.0],
            "scale": [0.01, 0.01, 0.02, 0.01, 0.01],
            "market": [0.003, 0.003, 0.003, 0.009, 0.004],
            "qqq_vol": [0.01, 0.01, 0.01, 0.02, 0.02],
            "part": ["open", "open", "open", "open", "midday"],
        }
    )
    found = hs.centred(frame).sort("day", "start", "ticker")
    np.testing.assert_allclose(found["own"], [-0.015, -0.005, 0.01, 0.015, 0.0])
    # the open's QQQ mean is over its two windows, 0.006, not its four rows, 0.0045
    np.testing.assert_allclose(found["market"], [-0.003, -0.003, -0.003, 0.003, 0.0])
    assert found["regime"].to_list() == ["calm"] * 3 + ["jumpy"] * 2


# --- the books --------------------------------------------------------------------------


def window_frame():
    """Two windows: the first of four stocks, the second of three."""
    return pl.DataFrame(
        {
            "day": [1] * 7,
            "start": [0] * 4 + [30] * 3,
            "ticker": ["A", "B", "C", "D", "A", "B", "C"],
            "own": [0.01] * 4 + [1.0] * 3,
            "scale": [0.01, 0.02, 0.03, 0.04, 0.5, 0.5, 0.5],
            "beta": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "market": [0.002] * 4 + [0.005] * 3,
            "part": ["open"] * 4 + ["morning"] * 3,
            "regime": ["calm"] * 7,
        }
    )


def test_a_book_holds_stocks_of_one_window_and_adds_them_by_side():
    rng = np.random.default_rng(5)
    found = hs.books(window_frame(), 3, 3, 2, rng)
    assert sorted(found["own"]) == pytest.approx([0.01, 0.01, 1.0, 1.0])
    assert sorted(found["market"]) == pytest.approx([0.002, 0.002, 0.005, 0.005])
    found = hs.books(window_frame(), 4, 4, 3, rng)  # the second window is too small
    np.testing.assert_allclose(found["own"], 0.01)
    np.testing.assert_allclose(found["unit"], 0.025)
    np.testing.assert_allclose(found["beta"], 1.0)
    assert list(found["part"]) == ["open"] * 3
    assert list(found["day"]) == [1] * 3 and list(found["draw"]) == [0, 1, 2]
    found = hs.books(window_frame(), 4, 2, 3, rng)  # two long, two short
    np.testing.assert_allclose(found["own"], 0.0, atol=1e-15)
    np.testing.assert_allclose(found["beta"], 0.0, atol=1e-15)
    np.testing.assert_allclose(found["unit"], 0.025)


def test_a_book_draws_different_stocks():
    frame = window_frame().with_columns(
        pl.Series("own", [0.01, 0.02, 0.03, 0.04, 1.0, 1.0, 1.0])
    )
    found = hs.books(frame.head(4), 2, 2, 50, np.random.default_rng(6))
    assert set(np.round(found["own"], 4)) <= {0.015, 0.02, 0.025, 0.03, 0.035}
    assert len(set(np.round(found["own"], 4))) > 2


# --- the choice -------------------------------------------------------------------------


def a_book(own=0.0, unit=0.01, beta=0.5, market=0.02):
    return {
        k: np.array([v])
        for k, v in {"own": own, "unit": unit, "beta": beta, "market": market}.items()
    }


def test_both_legs_count_towards_the_cap_and_pay_their_costs():
    book = a_book()
    exposure = 10 / 1.5
    assert hs.account(book, 0.3, 10, 1.0, 1.0)[0] == pytest.approx(
        exposure * 0.003 - exposure * 2e-4 - 0.5 * exposure * 0.5e-4
    )
    assert hs.account(book, 0.3, 10, 1.0, 0.0)[0] == pytest.approx(
        10 * (0.003 + 0.5 * 0.02) - 10 * 2e-4
    )
    assert hs.account(book, 0.3, 10, 0.5, 0.5)[0] == pytest.approx(
        4 * (0.003 + 0.5 * 0.5 * 0.02) - 4 * 2e-4 - 1 * 0.5e-4
    )
    short_book = a_book(beta=-0.5)  # its hedge buys QQQ, at the same cost
    assert hs.account(short_book, 0.3, 10, 1.0, 1.0)[0] == pytest.approx(
        exposure * 0.003 - exposure * 2e-4 - 0.5 * exposure * 0.5e-4
    )


def test_the_certainty_equivalent():
    assert hs.certainty(np.full(5, 0.01), 1) == pytest.approx(0.01)
    assert hs.certainty(np.full(5, 0.01), 2) == pytest.approx(0.01)
    swing = np.array([0.1, -0.1])
    assert hs.certainty(swing, 1) == pytest.approx(math.sqrt(1.1 * 0.9) - 1)
    assert hs.certainty(swing, 2) == pytest.approx(1 / ((1 / 1.1 + 1 / 0.9) / 2) - 1)
    assert hs.certainty(swing, 2) < hs.certainty(swing, 1) < 0
    assert hs.certainty(np.array([0.5, -1.0]), 1) == -1.0
    assert hs.certainty(np.array([0.5, -1.2]), 2) == -1.0


def synthetic(n=200_000, seed=3, size=0.03, beta=1.0):
    rng = np.random.default_rng(seed)
    own, market = rng.normal(0, size, n), rng.normal(0, size, n)
    return {
        "own": own - own.mean(),
        "unit": np.full(n, size),
        "beta": np.full(n, beta),
        "market": market - market.mean(),
    }


def test_without_an_edge_nothing_is_traded():
    found = hs.choose(synthetic(), 0.0, 4, 1)
    assert found["hedge"] is None and found["share"] == 0 and found["ce"] == 0
    assert "risk" not in found


def test_a_weak_edge_is_hedged_in_full_and_a_strong_one_against_the_cap_less():
    book = synthetic()
    weak = hs.choose(book, 0.05, 4, 1)  # its best size leaves room under the cap
    assert weak["hedge"] == 1.0 and 0 < weak["share"] < 1
    strong = hs.choose(book, 0.3, 4, 1)  # it would take more than the cap allows
    assert strong["hedge"] < 1.0 and strong["share"] == 1.0
    assert strong["ce"] > strong["full"]["ce"] > 0
    careful = hs.choose(book, 0.3, 4, 2)  # less appetite for risk, more of a hedge
    assert strong["hedge"] < careful["hedge"]
    assert careful["ce"] < strong["ce"]


def test_a_choice_that_would_wipe_the_account_out_is_never_taken():
    book = synthetic(n=10_000)
    calm = hs.choose(book, 0.3, 10, 1)
    book["own"][0] = -0.3  # one window in which the stocks fall 30% on their own
    found = hs.choose(book, 0.3, 10, 1)
    assert found["risk"]["worst"] > -1
    assert hs.account(book, 0.3, 10, found["share"], found["hedge"])[0] > -1

    def exposure(f):
        return f["share"] * 10 / (1 + f["hedge"])

    assert exposure(found) < exposure(calm)  # the crash held the stocks back


def test_an_empty_book_trades_nothing():
    empty = {k: np.array([]) for k in ("own", "unit", "beta", "market")}
    found = hs.choose(empty, 0.3, 4, 1)
    assert found["hedge"] is None and found["full"]["ce"] == 0


# --- a day --------------------------------------------------------------------------------


def test_a_day_compounds_its_windows_one_draw_at_a_time():
    book = {
        "own": np.array([0.01, 0.02, -0.01, 0.03, -0.5]),
        "unit": np.zeros(5),
        "beta": np.zeros(5),
        "market": np.zeros(5),
        "day": np.array([7, 7, 7, 8, 8]),
        "draw": np.array([0, 0, 1, 0, 0]),
    }
    found = hs.daily(book, 0.0, 4, 0.25, 1.0)  # an exposure of one
    returns = book["own"] - hs.COST_STOCK
    expected = [
        (1 + returns[0]) * (1 + returns[1]) - 1,
        returns[2],
        (1 + returns[3]) * (1 + returns[4]) - 1,
    ]
    np.testing.assert_allclose(found, expected)
    wiped = hs.daily(book, 0.0, 10, 1.0, 0.0)  # at 10x the last window takes everything
    assert wiped[2] == -1.0 and wiped[0] > -1


def test_a_day_s_risk_counts_the_days_below_each_loss():
    found = hs.day_risk(np.array([0.02, -0.04, -0.05, -0.12, 0.01]))
    assert found["worst"] == pytest.approx(-0.12)
    assert found["below 0.05"] == pytest.approx(2 / 5)
    assert found["below 0.1"] == pytest.approx(1 / 5)
    assert found["mean"] == pytest.approx(-0.036)


# --- a whole run --------------------------------------------------------------------------


def test_the_study_runs_on_a_small_folder_and_reports_every_case(tmp_path):
    data = store(tmp_path)
    found = hs.study(data, (2016,), log=lambda s: None)
    assert found["windows"] == 12 * 3 + 11 * 3
    assert found["books"] == {
        "one stock": 69,
        "ten long": 6 * hs.DRAWS,
        "five long, five short": 6 * hs.DRAWS,
    }
    assert len(found["results"]) == 3 * len(hs.CAPS) * len(hs.GAMMAS) * len(hs.EDGES)
    assert len(found["conditions"]) == len(hs.CONDITIONS) * len(hs.PARTS) * len(
        hs.REGIMES
    )
    assert len(found["daily"]) == 2 * len(hs.CAPS) * len(hs.DAILY_HEDGES)
    text = hs.report(found)
    assert "one stock, cap 4x: the best choice" in text
    assert "five long, five short, cap 10x: hedged in full | not at all" in text
    name, cap, gamma, edge = hs.CONDITIONS[0]
    assert (
        f"When to hedge: {name}, cap {cap:g}x, gamma {gamma:g}, edge {edge:g}" in text
    )
    assert "A day of such books in every window" in text
    assert "2016-2016; 2022-2026 not read" in text
