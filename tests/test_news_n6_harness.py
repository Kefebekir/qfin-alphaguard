"""N6's harness (research/news_reader/n6_harness.py)."""

import importlib.util
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.sessions import sessions

HERE = Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nh = _load("n6_harness")
harness = _load("harness")

DAYS = [s.day for s in sessions(date(2017, 12, 1), date(2018, 1, 31))]


def table(per_day: int = 25, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """n6_data's two tables for DAYS, `per_day` member-days a day, 1 to 3 headlines each."""
    g = np.random.default_rng(seed)
    rows, pairs, start = [], [], 0
    for day in DAYS:
        for k in range(per_day):
            n = int(g.integers(1, 4))
            rows.append(
                {
                    "ticker": f"T{k:02}",
                    "day": day,
                    "count": n,
                    "start": start,
                    "stop": start + n,
                    "next_return": float(g.normal(0, 0.02)),
                }
            )
            for j in range(n):
                pairs.append(
                    {
                        "id": len(pairs),
                        "row": len(pairs),
                        "age_hours": float(j + 1),
                        "edited": j == 0,
                        "tags": 1 + j % 2,
                    }
                )
            start += n
    days = pl.DataFrame(rows)
    n = days.height
    days = days.with_columns(
        **{c: pl.Series(g.normal(size=n)) for c in nh.NUMBERS if c != "vol20"},
        **{
            c: pl.Series(np.repeat(g.normal(size=len(DAYS)), per_day))
            for c in nh.MARKET
        },
        vol20=pl.Series(g.uniform(0.01, 0.03, n)),
        sector=pl.Series(g.integers(0, 9, n)),
        weekday=pl.col("day").dt.weekday() - 1,
        fomc=pl.lit(False),
        cpi=pl.lit(False),
        nfp=pl.lit(True),
    )
    return days, pl.DataFrame(pairs)


def test_the_target_is_n4_s_scaled_one():
    days, _ = table()
    days = days.with_columns(
        next_return=pl.when(pl.int_range(pl.len()) == 3)
        .then(None)
        .otherwise(pl.col("next_return")),
        vol20=pl.when(pl.int_range(pl.len()) == 5)
        .then(None)
        .otherwise(pl.col("vol20")),
    )
    y = nh.target(days)
    first = days.filter(pl.col("day") == DAYS[0])
    r, v = first["next_return"].to_numpy(), first["vol20"].to_numpy()
    v[5] = np.nanmedian(v)
    expected = (r - np.nanmean(r)) / v
    np.testing.assert_allclose(y[:25], expected, rtol=1e-6)
    assert np.isnan(y[3])


def test_the_member_s_numbers_are_scaled_within_each_day():
    days, _ = table()
    days = days.with_columns(
        r1=pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("r1")),
        r5=pl.lit(1.0),  # no spread: 0
    )
    scaled = nh.within_day(days, ("r1", "r5", "r20"))
    first = days.filter(pl.col("day") == DAYS[0])["r20"].to_numpy()
    np.testing.assert_allclose(
        scaled[:25, 2],
        np.clip((first - first.mean()) / first.std(ddof=1), -5, 5),
        rtol=1e-5,
    )
    assert scaled[0, 0] == 0.0 and (scaled[:, 1] == 0).all()
    assert np.abs(scaled).max() <= nh.CLIP
    # with n rows a day, one value is at most (n - 1) / sqrt(n) spreads out: 60 can pass 5
    outlier = table(per_day=60)[0].with_columns(
        r20=pl.when(pl.int_range(pl.len()) == 7).then(1e6).otherwise(pl.col("r20"))
    )
    assert nh.within_day(outlier, ("r20",))[7, 0] == nh.CLIP


def test_the_market_is_scaled_on_the_training_days_one_value_a_day():
    day = np.repeat(np.arange(4), [1, 3, 2, 5])  # days with 1, 3, 2 and 5 member-days
    market = np.repeat(
        np.array([[1.0], [3.0], [100.0], [np.nan]]), [1, 3, 2, 5], axis=0
    )
    train = day < 2
    out = nh.market_scaled(market.astype(np.float32), day, train)
    # the training days' values are 1 and 3, once each: mean 2, spread 1
    assert out[0, 0] == pytest.approx(-1.0) and out[1, 0] == pytest.approx(1.0)
    assert out[4, 0] == pytest.approx(nh.CLIP)  # 98 spreads away, clipped
    assert (out[6:, 0] == 0).all()  # missing


def test_a_member_day_keeps_its_last_headlines():
    days = pl.DataFrame({"start": [0, 2], "stop": [2, 7]})
    pairs = pl.DataFrame(
        {
            "row": [10, 11, 20, 21, 22, 23, 24],
            "age_hours": [3.0, 1.0, 9.0, 7.0, 5.0, 3.0, 1.0],
            "edited": [False, True, False, False, False, False, True],
            "tags": [1, 2, 1, 1, 1, 1, 2],
        }
    )
    heads, info = nh.headlines(days, pairs, most=3)
    assert heads.tolist() == [[10, 11, -1], [22, 23, 24]]
    np.testing.assert_allclose(info[1, :, 0], np.log1p([5.0, 3.0, 1.0]))
    assert info[0, 1].tolist() == [pytest.approx(np.log1p(1.0)), 1.0, 1.0]
    assert (info[0, 2] == 0).all()


def test_the_arrays_hold_every_member_day():
    days, pairs = table()
    a = nh.build(days.with_columns(sector=pl.lit(None, pl.Int64)), pairs)
    assert a.day.tolist() == np.repeat(np.arange(len(DAYS)), 25).tolist()
    assert (a.sector == nh.NO_SECTOR).all()
    np.testing.assert_allclose(a.count, np.log1p(days["count"].to_numpy()))
    assert a.member.shape == (days.height, len(nh.NUMBERS))
    assert a.market.shape == (days.height, len(nh.MARKET))
    assert (a.events[:, -1] == 1).all() and a.weekday.max() <= 4
    assert a.days.columns == ["day", "ticker", "next_return", "vol", "sector"]


def test_a_development_year_trains_on_the_days_before_less_the_gap():
    days, pairs = table()
    a = nh.build(days, pairs)
    trained, validated = nh.fold(a, 2018)
    when = a.days["day"].to_numpy()
    assert (when[validated] >= np.datetime64(date(2018, 1, 1))).all()
    assert validated.sum() == 25 * sum(d.year == 2018 for d in DAYS)
    assert not np.isnan(a.y[trained]).any() and not np.isnan(a.y[validated]).any()
    last_trained = when[trained].max()
    december = [d for d in DAYS if d.year == 2017]
    assert last_trained == np.datetime64(december[-1 - nh.GAP])


def test_the_placebo_deals_the_headlines_within_each_day():
    days, pairs = table()
    a = nh.build(days, pairs)
    b = nh.deal(a, seed=1)
    for d in range(len(DAYS)):
        rows = a.day == d
        assert sorted(map(tuple, a.heads[rows])) == sorted(map(tuple, b.heads[rows]))
    assert not np.array_equal(a.heads, b.heads)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.member, b.member)
    moved = {tuple(h): c for h, c in zip(b.heads, b.count, strict=True)}
    assert all(moved[tuple(h)] == c for h, c in zip(a.heads, a.count, strict=True))
    quiet = nh.silent(a)
    assert (quiet.heads == -1).all() and (quiet.count == 0).all()


def test_the_score_is_n4_s_portfolio():
    days, pairs = table()
    a = nh.build(days, pairs)
    _, validated = nh.fold(a, 2018)
    forecasts = np.random.default_rng(3).normal(size=validated.sum())
    got = nh.score(a, validated, forecasts)
    test = a.days.filter(pl.Series(validated))
    assert got == harness.n4_days(
        types.SimpleNamespace(test=test), forecasts, nh.LEVERS
    )
    s = nh.summary(got)
    scores = np.array([d["score"] for d in got])
    assert s["days"] == len(got) and s["bps"] == pytest.approx(scores.mean() * 1e4)
    assert s["t"] == pytest.approx(harness.t_stat(scores))


def test_a_development_year_stops_at_the_last_day(monkeypatch):
    days, pairs = table()
    a = nh.build(days, pairs)
    monkeypatch.setattr(nh, "LAST_DAY", date(2018, 1, 10))
    _, validated = nh.fold(a, 2018)
    assert a.days["day"].filter(pl.Series(validated)).max() == date(2018, 1, 10)
