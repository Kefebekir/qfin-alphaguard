"""KUTBIT's skill test (research/kutbit/skill.py)."""

import importlib.util
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW, Day, inputs

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_skill", HERE / "skill.py")
ks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ks)

FIELDS = ("open", "high", "low", "close", "volume", "profile", "swing", "habit")


def moments(x, z):
    k, h = x.shape[1], z.shape[1]
    m = ks.Moments(0, np.zeros(k), np.zeros((k, k)), np.zeros((k, h)), np.zeros(h))
    m.add(x.astype(np.float32), z)
    return m


def test_the_ridge_is_fitted_on_standardised_inputs():
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, [1.0, 3.0, 0.5, 2.0], (2000, 4))
    z = np.stack([x @ [0.5, -0.2, 0.0, 0.1], x[:, 0]], axis=1) + rng.normal(
        0, 1, (2000, 2)
    )
    model = ks.fit(moments(x, z), 0.3)
    x32 = x.astype(np.float32).astype(float)
    mean, sd = x32.mean(axis=0), x32.std(axis=0)
    standard = (x32 - mean) / sd
    b = np.linalg.solve(
        standard.T @ standard / len(x) + 0.3 * np.eye(4),
        standard.T @ (z - z.mean(axis=0)) / len(x),
    )
    np.testing.assert_allclose(model.beta, b / sd[:, None], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        model(x),
        z.mean(axis=0) + (x32 - mean) @ (b / sd[:, None]),
        rtol=1e-4,
        atol=1e-5,  # X'X is summed in float32
    )


def test_without_a_penalty_the_ridge_is_least_squares():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (3000, 3))
    z = (1.5 + x @ [0.4, -1.0, 0.2])[:, None] + rng.normal(0, 0.1, (3000, 1))
    model = ks.fit(moments(x, z), 1e-12)
    design = np.column_stack([np.ones(3000), x.astype(np.float32)])
    exact = np.linalg.lstsq(design, z, rcond=None)[0]
    np.testing.assert_allclose(model.intercept, exact[0], rtol=1e-5)
    np.testing.assert_allclose(model.beta, exact[1:], rtol=1e-4)


def test_an_input_that_never_varies_gets_no_weight():
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.normal(0, 1, 500), np.full(500, 7.0)])
    z = x[:, :1] * 2
    model = ks.fit(moments(x, z), 0.01)
    assert model.beta[1, 0] == 0.0 and model.beta[0, 0] > 1.9


def test_moments_add_up_over_days():
    rng = np.random.default_rng(4)
    x, z = rng.normal(0, 1, (100, 5)), rng.normal(0, 1, (100, 2))
    whole = moments(x, z)
    halves = moments(x[:40], z[:40]) + moments(x[40:], z[40:])
    for name in ("n", "s", "q", "r", "t"):
        np.testing.assert_allclose(
            getattr(halves, name), getattr(whole, name), rtol=1e-5
        )


def test_correlations_by_day_and_across_instruments_by_hand():
    rng = np.random.default_rng(5)
    day = np.repeat([10, 11, 12], 40)
    minute = np.tile(np.repeat([0, 1], 20), 3)
    p, z = rng.normal(0.3, 1, 120), rng.normal(-0.2, 1, 120)
    found = ks.daily_correlation(p, z, day)
    dp, dz = (
        p - p.mean(),
        z - z.mean(),
    )  # about the means of all the rows, not each day's
    per_day = [
        (dp[day == d] @ dz[day == d])
        / math.sqrt((dp[day == d] @ dp[day == d]) * (dz[day == d] @ dz[day == d]))
        for d in (10, 11, 12)
    ]
    assert found["mean"] == pytest.approx(np.mean(per_day))
    assert found["t"] == pytest.approx(
        np.mean(per_day) / np.std(per_day, ddof=1) * math.sqrt(3)
    )
    across = ks.across_correlation(p, z, day, minute)
    each = [
        np.mean(
            [
                np.corrcoef(
                    p[(day == d) & (minute == m)], z[(day == d) & (minute == m)]
                )[0, 1]
                for m in (0, 1)
            ]
        )
        for d in (10, 11, 12)
    ]
    assert across["mean"] == pytest.approx(np.mean(each)) and across["n"] == 3


def test_a_path_every_instrument_of_a_day_shares_does_not_look_like_a_reversal():
    # Each day, ten instruments: a path they all share, as QQQ's part, and one of their own.
    # The forecast is the last 30 minutes reversed; the target, the next 30 minutes. Nothing
    # can be forecast, but about each day's own means the past and the future of the shared
    # path move against each other, as in the placebo's first run on 17 November 2026.
    rng = np.random.default_rng(17)
    days, instruments, minutes, back, ahead = 300, 10, 390, 30, 30
    shared = np.cumsum(rng.normal(0, 1, (days, 1, minutes + 1)), axis=2)
    paths = shared + np.cumsum(
        rng.normal(0, 1, (days, instruments, minutes + 1)), axis=2
    )
    t = np.arange(back, minutes + 1 - ahead)
    p = -(paths[:, :, t] - paths[:, :, t - back]).ravel()
    z = ((paths[:, :, t + ahead] - paths[:, :, t]) / math.sqrt(ahead)).ravel()
    day = np.repeat(np.arange(days), instruments * len(t))
    minute = np.tile(t, days * instruments)
    _, about_each_day = ks.grouped_correlations(p, z, day)
    assert ks.mean_and_t(about_each_day)["t"] > 4  # the bias the measure must not have
    assert abs(ks.daily_correlation(p, z, day)["t"]) < ks.PLACEBO_T
    assert abs(ks.across_correlation(p, z, day, minute)["t"]) < ks.PLACEBO_T


def test_a_minute_with_too_few_instruments_is_left_out():
    day, minute = (
        np.zeros(ks.MIN_ACROSS - 1, dtype=int),
        np.zeros(ks.MIN_ACROSS - 1, dtype=int),
    )
    rng = np.random.default_rng(6)
    found = ks.across_correlation(
        rng.normal(0, 1, len(day)), rng.normal(0, 1, len(day)), day, minute
    )
    assert found["n"] == 0


def test_a_mean_s_t_allows_one_day_s_values_to_move_together():
    values = np.array([1.0, 1.0, 1.0, 3.0, 3.0, -1.0])
    groups = np.array([1, 1, 1, 2, 2, 3])
    found = ks.clustered_mean(values, groups)
    mean = values.mean()
    sums = [sum(values[groups == g] - mean) for g in (1, 2, 3)]
    assert found["mean"] == pytest.approx(mean)
    assert found["t"] == pytest.approx(mean / (math.sqrt(sum(s * s for s in sums)) / 6))


def test_the_strongest_forecasts_trade_in_their_direction_after_the_cost():
    p = np.array([0.5, -0.6, 0.1, -0.05, 0.7, -0.9])
    apart_bps = np.array([10.0, -4.0, 50.0, 50.0, -2.0, 6.0])
    day = np.array([1, 1, 1, 2, 2, 2])
    found = ks.strongest(p, apart_bps, day, threshold=0.5)
    gross = np.array(
        [10.0, 4.0, -2.0, -6.0]
    )  # the forecast's sign times the hedged move
    assert found["gross"] == pytest.approx(gross.mean())
    assert found["net"] == pytest.approx(gross.mean() - ks.COST)
    assert found["signals_a_day"] == pytest.approx(2.0)
    assert ks.thresholds_of(np.arange(-50, 51.0))[0.1] == pytest.approx(45.0)


def a_day(rng, n=60, **given) -> Day:
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
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
        "habit": rng.normal(0, 1e-3, n),
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


def test_a_random_walk_keeps_all_but_the_prices_and_moves_by_the_usual_move():
    rng = np.random.default_rng(7)
    swing = np.where(np.arange(390) < 30, 4e-3, 1e-3)
    day = a_day(rng, 390, swing=swing)
    walk = ks.random_walk(day, np.random.default_rng(1))
    again = ks.random_walk(day, np.random.default_rng(1))
    np.testing.assert_array_equal(walk.close, again.close)
    assert walk.open[0] == day.open[0] and np.array_equal(walk.volume, day.volume)
    assert walk.habit is day.habit and walk.beta == day.beta
    steps = np.log(walk.close / walk.open)
    assert np.std(steps[:30]) == pytest.approx(4e-3, rel=0.35)
    assert np.std(steps[30:]) == pytest.approx(1e-3, rel=0.15)
    np.testing.assert_allclose(walk.open[1:], walk.close[:-1])


def test_an_instrument_s_rows_are_its_inputs_and_what_followed():
    rng = np.random.default_rng(8)
    market = a_day(rng, beta=1.0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 60)))
    close[40:] *= 1.5  # a jump far past the clip
    open_ = np.concatenate([[100.0], close[:-1]])
    day = a_day(
        rng,
        open=open_,
        close=close,
        high=np.maximum(open_, close) * 1.0002,
        low=np.minimum(open_, close) * 0.9998,
    )
    found = ks.instrument_rows(day, market)
    windows, scalars = inputs(day, market)
    x = np.concatenate([windows.reshape(60, -1), scalars], axis=1)
    assert found["x"].shape == (59, ks.INPUTS) and found["x"].dtype == np.float32
    np.testing.assert_allclose(found["x"], np.clip(x[:-1], -10, 10), rtol=1e-6)
    assert found["x"].max() == 10.0
    assert (
        found["z"].shape == (59, len(ks.HORIZONS)) and np.abs(found["z"]).max() == 5.0
    )
    assert np.abs(found["apart"]).max() > 0.3  # the hedged return itself is not clipped
    np.testing.assert_allclose(
        found["patterns"][:, 0], scalars[:-1, SCALARS.index("habit")]
    )
    apart = windows[:-1, -30:, CHANNELS.index("apart")].sum(axis=1)
    np.testing.assert_allclose(found["patterns"][:, 1], -apart)
    assert WINDOW * len(CHANNELS) + len(SCALARS) == ks.INPUTS == 971


# --- a whole run on a small data folder ----------------------------------------------------

TICKERS = ("QQQ", *(f"S{i:02d}" for i in range(12)), "XLF")
MINUTES = 60


def store(
    tmp_path,
    signal: float,
    seed: int = 9,
    measured_scale: float = 1.0,
    signal_years=range(2016, 2022),
):
    """A data folder with each development year's first 6 days of 14 instruments. In
    `signal_years`, each stock's minute moves repeat `signal` times the move before, which a
    model can learn."""
    rng = np.random.default_rng(seed)
    rows, arrays = [], {}
    for year in range(2016, 2022):
        fields = {f: [] for f in (*FIELDS, "seen", "news")}
        row = 0
        for d in range(6):
            day = date(year, 3, 1 + d)
            for ticker in TICKERS:
                steps = np.zeros(MINUTES)
                noise = rng.normal(0, 1e-3, MINUTES)
                for m in range(MINUTES):
                    planted = signal if year in signal_years and ticker != "QQQ" else 0
                    steps[m] = noise[m] + (planted * steps[m - 1] if m else 0)
                if year in (2020, 2021):
                    steps *= measured_scale
                close = 100 * np.exp(np.cumsum(steps))
                open_ = np.concatenate([[100.0], close[:-1]])
                values = {
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.0002,
                    "low": np.minimum(open_, close) * 0.9998,
                    "close": close,
                    "volume": np.full(MINUTES, 1000.0),
                    "profile": np.full(MINUTES, 1000.0),
                    "swing": np.full(MINUTES, 1e-3),
                    "habit": rng.normal(0, 1e-3, MINUTES),
                }
                for f in FIELDS:
                    fields[f].append(
                        np.pad(values[f], (0, 390 - MINUTES), mode="edge").astype(
                            np.float32
                        )
                    )
                fields["seen"].append(np.ones(390, dtype=bool))
                fields["news"].append(np.zeros(390, dtype=np.uint16))
                rows.append(
                    {
                        "day": day,
                        "year": year,
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
                        "usable": True,
                    }
                )
                row += 1
        arrays[year] = {f: np.stack(v) for f, v in fields.items()}
    folder = tmp_path / "kutbit"
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(folder / "days.parquet")
    for year, fields in arrays.items():
        np.savez_compressed(folder / f"{year}.npz", **fields)
    return tmp_path


def test_a_planted_signal_is_found_and_the_random_walk_finds_nothing(tmp_path):
    data = store(tmp_path, signal=0.4)
    real = ks.run(data, "real", log=lambda s: None)
    one_minute = real["results"]["ridge"]["1"]
    assert (
        one_minute["all"]["daily"]["mean"] > 0.2 and one_minute["all"]["daily"]["t"] > 5
    )
    assert one_minute["all"]["across"]["t"] > 5
    assert (
        one_minute["all"]["top0.1"]["gross"] > 0
        and one_minute["all"]["top0.1"]["t"] > 3
    )
    assert one_minute["stocks"]["daily"]["mean"] > 0.2  # the ETF here has no signal
    assert real["checks"] is None
    assert real["fitted_rows"] == 4 * 6 * 13 * (
        MINUTES - 1
    )  # all but QQQ, all but the last
    assert real["measured_rows"] == 2 * 6 * 13 * (MINUTES - 1)
    walk = ks.run(data, "random", log=lambda s: None)
    assert walk["checks"] == {"passed": True, "failed": []}
    assert abs(walk["results"]["ridge"]["1"]["all"]["daily"]["mean"]) < 0.05


def test_each_day_keeps_a_year_s_first_days_when_limited(tmp_path):
    data = store(tmp_path, signal=0.0)
    days, arrays = ks.load(data, [2016, 2017])
    tickers = {t: i for i, t in enumerate(sorted(days["ticker"].unique()))}
    seen = [d for d, _ in ks.each_day(days, arrays, [2016, 2017], tickers, "real", 2)]
    assert seen == [
        date(2016, 3, 1),
        date(2016, 3, 2),
        date(2017, 3, 1),
        date(2017, 3, 2),
    ]


def test_only_usable_sessions_of_instruments_other_than_the_market_are_rows(tmp_path):
    data = store(tmp_path, signal=0.0)
    days = pl.read_parquet(data / "kutbit" / "days.parquet")
    days = days.with_columns(
        usable=~(
            ((pl.col("ticker") == "S00") & (pl.col("day") == date(2016, 3, 2)))
            | ((pl.col("ticker") == "QQQ") & (pl.col("day") == date(2016, 3, 3)))
        )
    )
    kept = ks.eligible(days, [2016])
    assert "QQQ" not in kept["ticker"].to_list()
    assert kept.filter(pl.col("day") == date(2016, 3, 2)).height == 12  # S00 left out
    assert (
        kept.filter(pl.col("day") == date(2016, 3, 3)).height == 0
    )  # no usable market
    assert set(kept["year"]) == {2016}


def test_the_choice_is_made_before_the_measured_years_are_read(tmp_path):
    first = ks.run(store(tmp_path / "a", 0.4), "real", log=lambda s: None)
    other = ks.run(
        store(tmp_path / "b", 0.4, measured_scale=3.0), "real", log=lambda s: None
    )
    assert first["choice"] == other["choice"]
    assert first["results"] != other["results"]


def test_a_random_walk_that_shows_something_fails_the_rule():
    found = {
        "all": {
            "daily": {"t": 0.5},
            "across": {"t": 0.1},
            "top0.1": {"t": -3.2},
            "top0.01": {"t": float("nan")},
        }
    }
    assert ks.checks({"ridge": {1: found}}) == {
        "passed": False,
        "failed": ["ridge, 1 min, top0.1: t -3.2"],
    }
    found["all"]["top0.1"]["t"] = 2.9
    assert ks.checks({"habit": {60: found}})["passed"]


def test_the_real_run_waits_for_the_random_walk_placebo(tmp_path):
    with pytest.raises(SystemExit, match="random-walk placebo first"):
        ks.main(["real", str(tmp_path)])
    folder = tmp_path / "kutbit" / "skill"
    folder.mkdir(parents=True)
    (folder / "random.json").write_text(json.dumps({"checks": {"passed": False}}))
    with pytest.raises(SystemExit, match="random-walk placebo first"):
        ks.main(["real", str(tmp_path)])


def test_the_page_shows_each_run(tmp_path):
    data = store(tmp_path, signal=0.4)
    path = ks.write(data, ks.run(data, "random", log=lambda s: None))
    path = ks.write(data, ks.run(data, "real", log=lambda s: None))
    text = path.read_text()
    assert "Real prices" in text and "Placebo: random-walk prices" in text
    assert text.count("The placebo rule holds") == 1
    assert "Run after the random-walk placebo passed." in text
    saved = json.loads((data / "kutbit" / "skill" / "real.json").read_text())
    assert saved["source"] == "real"
    assert set(saved["results"]) == {"ridge", "habit", "reversal"}


def test_the_final_model_is_fitted_on_the_chosen_year_too(tmp_path):
    # a signal that starts in 2019: only a model that saw 2019 can find it in 2020-2021
    data = store(tmp_path, signal=0.4, signal_years=(2019, 2020, 2021))
    found = ks.run(data, "real", log=lambda s: None)["results"]["ridge"]["1"]["all"]
    assert found["daily"]["mean"] > 0.1


def test_the_penalty_is_the_one_that_forecast_best_and_sets_the_thresholds():
    rng = np.random.default_rng(10)
    n, h, k = 4000, len(ks.HORIZONS), len(ks.PENALTIES)
    z = rng.normal(0, 1, (n, h))
    early = rng.normal(0, 1, (n, h, k))
    best = [3, 0, 4, 2, 1]
    for j, i in enumerate(best):
        early[:, j, i] += z[:, j]  # the penalty that forecasts best, horizon by horizon
    final = rng.normal(0, 1, (n, h, k)) * np.arange(
        1, k + 1
    )  # each penalty its own size
    patterns = rng.normal(0, [1.0, 2.0], (n, 2))
    choice, thresholds = ks.choose(early, final, patterns, z)
    for j, horizon in enumerate(ks.HORIZONS):
        assert choice[horizon]["penalty"] == ks.PENALTIES[best[j]]
        assert thresholds[horizon] == ks.thresholds_of(final[:, j, best[j]])
    assert thresholds["reversal"] == ks.thresholds_of(patterns[:, 1])


def test_each_horizon_takes_its_own_penalty_s_model():
    models = [
        ks.Ridge(
            np.full(len(ks.HORIZONS), float(i)),
            np.full((3, len(ks.HORIZONS)), i * 10.0),
        )
        for i in range(len(ks.PENALTIES))
    ]
    choice = {
        h: {"penalty": ks.PENALTIES[(j + 2) % 5]} for j, h in enumerate(ks.HORIZONS)
    }
    model = ks.ridge_of(models, choice)
    for j in range(len(ks.HORIZONS)):
        assert model.intercept[j] == (j + 2) % 5
        assert (model.beta[:, j] == ((j + 2) % 5) * 10.0).all()


def test_the_measures_count_the_hedged_return_in_bps():
    rng = np.random.default_rng(11)
    n = 600
    p = rng.normal(0, 1, n)
    apart = np.sign(p) * 0.001  # every forecast right, by 10 bps
    day = np.repeat(np.arange(6), 100)
    found = ks.measures(
        p,
        rng.normal(0, 1, n),
        apart,
        day,
        np.tile(np.arange(100), 6),
        {0.1: 1.0, 0.01: 2.5},
    )
    assert found["top0.1"]["gross"] == pytest.approx(10.0)
    assert found["top0.01"]["net"] == pytest.approx(10.0 - ks.COST)
