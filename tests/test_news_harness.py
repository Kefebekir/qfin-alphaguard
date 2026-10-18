"""The news reader's harness (research/news_reader/harness.py): the parts that decide a test."""

import importlib.util
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_spec = importlib.util.spec_from_file_location(
    "harness", Path(__file__).parents[1] / "research" / "news_reader" / "harness.py"
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def test_a_test_year_reads_the_vintage_of_the_year_before_and_none_after_2024():
    assert [harness.vintage_for(y) for y in (2018, 2021, 2025, 2026)] == [
        2017,
        2020,
        2024,
        2024,
    ]


def test_the_size_follows_the_third_of_the_training_forecasts():
    train = np.arange(1.0, 10.0)  # thirds at 3.67 and 6.33
    sizes = harness.n1_sizes(train, np.array([3.0, 5.0, 7.0, 6.33, 6.34]))
    assert sizes.tolist() == [0.5, 1.0, 1.5, 1.0, 1.5]


def year(train_rule, test_rule):
    return harness.Year(
        2018,
        np.zeros((len(train_rule), 1)),
        np.zeros(len(train_rule)),
        np.zeros((len(test_rule), 1)),
        pl.DataFrame({"rule": train_rule}),
        pl.DataFrame({"rule": test_rule}),
    )


def test_an_uninformative_forecast_leaves_the_rule_as_it_is():
    data = year([0.01, -0.02, 0.005, 0.0], [0.03, -0.01])
    flat = np.zeros(4)
    assert harness.n1_score(data, flat, np.zeros(2)).tolist() == [0.0, 0.0]


def test_the_sized_rule_is_compared_at_the_rule_s_risk():
    train_rule = np.array([0.01, -0.02, 0.005, 0.0, 0.015, -0.01])
    train_forecasts = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )  # sizes 0.5 0.5 1 1 1.5 1.5
    data = year(train_rule.tolist(), [0.02, 0.02, 0.02])
    sized = np.array([0.5, 0.5, 1, 1, 1.5, 1.5]) * train_rule
    scale = train_rule.std(ddof=1) / sized.std(ddof=1)
    got = harness.n1_score(data, train_forecasts, np.array([0.0, 3.5, 9.0]))
    want = scale * np.array([0.5, 1.0, 1.5]) * 0.02 - 0.02
    assert got == pytest.approx(want)


def test_the_long_short_takes_the_tenths_less_costs_and_skips_thin_days():
    days = [date(2022, 1, 3)] * 20 + [date(2022, 1, 4)] * 19
    returns = list(np.arange(20) / 1000) + [0.01] * 19
    forecasts = np.concatenate([np.arange(20.0), np.zeros(19)])
    data = harness.Year(
        2022,
        np.zeros((0, 1)),
        np.zeros(0),
        np.zeros((39, 1)),
        pl.DataFrame(),
        pl.DataFrame({"day": days, "next_return": returns}),
    )
    scores = harness.n2_score(data, forecasts)
    # 20 members: long the top 2 (0.018, 0.019), short the bottom 2 (0, 0.001), less 2 bps,
    # 1 bp a round trip on each side (amendment 2)
    assert scores.tolist() == pytest.approx(
        [0.0185 - 0.0005 - 0.0002]
    )  # the 19 are skipped


def test_each_test_passes_at_its_own_threshold():
    # N1 had half of the news tests' 2.5% (Bonferroni over N1 and N2); N2 and N3 share the
    # other half (amendment 3).
    assert harness.THRESHOLDS == {"N1": 2.24, "N2": 2.50, "N3": 2.50}
    assert harness.COST == 0.0001


def test_the_t_statistic():
    v = np.array([0.01, 0.02, 0.03])
    assert harness.t_stat(v) == pytest.approx(0.02 / 0.01 * math.sqrt(3))


def test_training_years_end_before_the_test_year_and_the_holdout_on_its_last_day():
    rows = pl.DataFrame(
        {
            "day": [
                date(2015, 6, 1),
                date(2016, 6, 1),
                date(2021, 6, 1),
                date(2026, 10, 9),
                date(2026, 10, 12),
            ],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    x = np.arange(5.0)[:, None]
    data = harness._split(rows, x, 2026)
    assert data.train_y.tolist() == [2.0, 3.0]  # 2016-2025; not 2015
    assert data.test_x.ravel().tolist() == [3.0]  # 2026 up to 9 October


def test_the_next_day_return_spans_a_weekend_and_needs_both_closes(
    tmp_path, monkeypatch
):
    prices = pl.DataFrame(
        {
            "date": [date(2022, 1, 7), date(2022, 1, 10), date(2022, 1, 7)],
            "ticker": ["AAA", "AAA", "BBB"],
            "close": [100.0, 102.0, 50.0],
        }
    )
    prices.write_parquet(tmp_path / "prices.parquet")
    monkeypatch.setattr(harness, "PRICES", tmp_path / "prices.parquet")
    got = harness.next_day_returns()
    assert got.rows() == [(date(2022, 1, 7), "AAA", pytest.approx(0.02))]


def test_the_holdout_needs_a_chosen_variant_and_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "VARIANTS", tmp_path / "variants")
    monkeypatch.setattr(harness, "LOCKS", tmp_path)
    with pytest.raises(RuntimeError, match="commit"):
        harness.run_holdout("N1", object)
    (tmp_path / "holdout_N1.json").write_text("{}")
    with pytest.raises(RuntimeError, match="has been run"):
        harness.run_holdout("N1", object)


def test_a_split_other_than_development_or_holdout_is_refused():
    with pytest.raises(ValueError, match="split"):
        harness.run("N1", object, split="test")


def n3_files(tmp_path, monkeypatch):
    rows = pl.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA"],
            "day": [
                date(2015, 12, 31),
                date(2016, 3, 1),
                date(2017, 3, 1),
                date(2018, 3, 1),
                date(2019, 3, 1),
            ],
            "count": [1, 3, 1, 2, 7],
        }
    )
    rows.write_parquet(tmp_path / "n3_rows.parquet")
    embedding = np.arange(10, dtype=np.float16).reshape(5, 2)  # row r is [2r, 2r + 1]
    np.save(tmp_path / "n3_ft-mean.npy", embedding)
    monkeypatch.setattr(harness, "PIT_EMB", tmp_path)
    returns = rows.select(
        "day", "ticker", next_return=pl.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    )
    return returns


def test_n3_reads_its_feature_set_and_the_count_up_to_the_test_year(
    tmp_path, monkeypatch
):
    returns = n3_files(tmp_path, monkeypatch)
    data = harness.n3_year(2018, returns, "ft-mean")
    # Trained on 2016-2017 (not 2015); tested on 2018; 2019 is not read.
    assert data.train_x.tolist() == [
        [2.0, 3.0, pytest.approx(math.log(4))],
        [4.0, 5.0, pytest.approx(math.log(2))],
    ]
    assert data.test_x.tolist() == [[6.0, 7.0, pytest.approx(math.log(3))]]
    assert data.train_y.tolist() == [0.0, 0.0]  # one member a day: none beats the mean
    counted = harness.n3_year(2018, returns, None, count_only=True)
    assert counted.test_x.tolist() == [[pytest.approx(math.log(3))]]
    with pytest.raises(ValueError, match="one of"):
        harness.n3_year(2018, returns, "base-max")


def test_n3_needs_a_feature_set_and_the_others_take_none():
    with pytest.raises(ValueError, match="N3's features"):
        harness.run("N3", object)
    with pytest.raises(ValueError, match="N3's features"):
        harness.run("N3", object, features="mean")
    with pytest.raises(ValueError, match="one feature set"):
        harness.run("N2", object, features="base-last")


def test_n3_s_holdout_takes_the_feature_set_from_the_chosen_variant(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(harness, "VARIANTS", tmp_path)
    monkeypatch.setattr(harness, "LOCKS", tmp_path)
    (tmp_path / "N3_chosen.json").write_text('{"components": 4, "penalty": 10.0}')
    with pytest.raises(RuntimeError, match="feature set"):
        harness.run_holdout("N3", object)
