"""N6's member-days and their headlines (research/news_reader/n6_data.py)."""

import importlib.util
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n6_data = _load("n6_data")
members = _load("members")


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_the_reader_is_n3_s_checkpoint_by_new_york_s_year():
    moments = [
        utc(2016, 3, 1, 12),
        utc(2017, 1, 1, 3),  # 22:00 on 31 December 2016 in New York
        utc(2017, 1, 1, 6),
        utc(2020, 7, 1),
        utc(2025, 2, 1),
        utc(2026, 10, 1),
    ]
    got = (
        pl.DataFrame({"known": moments}, schema={"known": pl.Datetime("us", "UTC")})
        .select(n6_data.reader(pl.col("known")))["known"]
        .to_list()
    )
    expected = [n6_data.CHECKPOINTS.index(members.checkpoint(m)) for m in moments]
    assert got == expected == [0, 0, 1, 4, 9, 9]


def test_15_30_in_new_york_is_19_30_or_20_30_utc():
    assert n6_data.decision(date(2026, 10, 19)) == utc(2026, 10, 19, 19, 30)
    assert n6_data.decision(date(2026, 11, 9)) == utc(2026, 11, 9, 20, 30)


def world():
    """Two member-days on Monday 19 October 2026 (15:30 is 19:30 UTC) and their pairs."""
    rows = pl.DataFrame(
        {"ticker": ["AAA", "BBB"], "day": [date(2026, 10, 19)] * 2, "count": [2, 3]},
        schema_overrides={"count": pl.UInt32},
    )
    groups = pl.DataFrame({"id": [11, 12, 21, 22, 22], "group": [0, 0, 1, 1, 1]})
    index = pl.DataFrame(
        {
            "id": [11, 12, 21, 22, 99],
            "created": [
                utc(2026, 10, 19, 19, 0),
                utc(2026, 10, 16, 21, 0),
                utc(2026, 10, 19, 13, 30),
                utc(2026, 10, 1, 12, 0),
                utc(2026, 10, 1),
            ],
            "known": [
                utc(2026, 10, 19, 19, 0),
                utc(2026, 10, 16, 21, 0),
                utc(2026, 10, 19, 13, 30),
                utc(2026, 10, 19, 7, 30),  # counts from its edit
                utc(2026, 10, 1),
            ],
            "symbols": [["AAA"], ["AAA"], ["BBB", "XYZ"], ["BBB", "BBB.A"], ["Z"]],
        },
        schema_overrides={
            "created": pl.Datetime("us", "UTC"),
            "known": pl.Datetime("us", "UTC"),
        },
    )
    store_ids = np.array([99, 22, 21, 12, 11])
    return groups, rows, index, store_ids


def test_each_member_day_s_headlines_come_in_the_order_they_became_known():
    pairs, days = n6_data.arrange(*world())
    assert pairs["id"].to_list() == [12, 11, 22, 22, 21]
    assert pairs["row"].to_list() == [3, 4, 1, 1, 2]
    assert pairs["age_hours"].to_list() == pytest.approx([70.5, 0.5, 12.0, 12.0, 6.0])
    assert pairs["edited"].to_list() == [False, False, True, True, False]
    assert pairs["tags"].to_list() == [1, 1, 2, 2, 2]
    assert days["start"].to_list() == [0, 2] and days["stop"].to_list() == [2, 5]
    assert days.select("ticker", "day", "count").equals(world()[1])


def test_a_headline_missing_from_the_store_or_known_after_15_30_is_refused():
    groups, rows, index, store_ids = world()
    with pytest.raises(ValueError, match="not in the store"):
        n6_data.arrange(groups, rows, index, store_ids[store_ids != 21])
    late = index.with_columns(
        known=pl.when(pl.col("id") == 11)
        .then(pl.lit(utc(2026, 10, 19, 19, 31)))
        .otherwise(pl.col("known"))
    )
    with pytest.raises(ValueError, match="after 15:30"):
        n6_data.arrange(groups, rows, late, store_ids)
    wrong = rows.with_columns(count=pl.Series([2, 2], dtype=pl.UInt32))
    with pytest.raises(ValueError, match="count is not"):
        n6_data.arrange(groups, wrong, index, store_ids)
