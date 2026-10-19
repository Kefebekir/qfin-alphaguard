"""N3's article rules and features (research/news_reader/members.py, features_n3.py)."""

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


members = _load("members")
features_n3 = _load("features_n3")


@pytest.mark.parametrize(
    ("known", "expected"),
    [
        (datetime(2016, 5, 1, 14, tzinfo=UTC), "201511"),
        (datetime(2017, 1, 3, 14, tzinfo=UTC), "201612"),
        # 23:30 in New York on 31 December 2019 is 2020 in UTC: still 2019's article.
        (datetime(2020, 1, 1, 4, 30, tzinfo=UTC), "201812"),
        (datetime(2020, 1, 1, 5, 30, tzinfo=UTC), "201912"),
        (datetime(2025, 3, 1, 14, tzinfo=UTC), "202412"),
        (datetime(2026, 10, 9, 14, tzinfo=UTC), "202412"),
    ],
)
def test_an_article_is_read_by_the_checkpoint_of_the_year_before(known, expected):
    assert members.checkpoint(known) == expected


def test_member_articles_follow_n2_s_rule():
    # Tuesday 6 October 2026's window: 16:00 New York on Monday (20:00 UTC) to 15:30 on
    # Tuesday (19:30 UTC); Wednesday's opens at 16:00 on Tuesday, leaving 15:30-16:00 to none.
    known = [
        (1, datetime(2026, 10, 5, 19, 59, tzinfo=UTC), ["AAPL"]),  # before the window
        (2, datetime(2026, 10, 5, 20, 1, tzinfo=UTC), ["BRK/B"]),
        (3, datetime(2026, 10, 6, 13, 0, tzinfo=UTC), ["AAPL", "MSFT", "NVDA"]),
        (4, datetime(2026, 10, 6, 14, 0, tzinfo=UTC), ["XYZ"]),
        (5, datetime(2026, 10, 6, 15, 0, tzinfo=UTC), ["CHK"]),
        (6, datetime(2026, 10, 6, 19, 31, tzinfo=UTC), ["AAPL"]),  # 15:31: no window
        (7, datetime(2026, 10, 6, 20, 1, tzinfo=UTC), ["AAPL"]),  # 16:01: Wednesday's
        # Sessions run to the last article's day, as in N2: this one keeps Wednesday in.
        (8, datetime(2026, 10, 7, 14, 0, tzinfo=UTC), ["XYZ"]),
        (9, datetime(2026, 10, 6, 16, 0, tzinfo=UTC), ["DUP"]),  # fits two members
    ]
    index = pl.DataFrame(known, schema=["id", "known", "symbols"], orient="row")
    index = index.sort("known")  # the index's order: as articles became known
    prices = pl.DataFrame(
        [
            (date(2026, 10, 5), "AAPL"),
            (date(2026, 10, 6), "AAPL"),
            (date(2026, 10, 6), "BRK-B"),
            (date(2026, 10, 6), "CHK_old"),
            (date(2026, 10, 6), "DUP"),
            (date(2026, 10, 6), "DUP_old"),
            (date(2026, 10, 7), "AAPL"),
        ],
        schema=["date", "ticker"],
        orient="row",
    ).with_columns(sp500=True)
    got = members.member_articles(index, prices).sort("id")
    assert got.rows() == [
        (2, "BRK-B", date(2026, 10, 6)),  # BRK/B read as BRK-B
        (5, "CHK_old", date(2026, 10, 6)),  # the member that day under its _old name
        (7, "AAPL", date(2026, 10, 7)),
    ]
    # 3 tags three stocks; 4 and 8 tag no member; 9 fits two.


def pairs_of(rows):
    return pl.DataFrame(
        rows, schema={"id": pl.Int64, "ticker": pl.String, "day": pl.Date}, orient="row"
    )


def test_a_member_day_counts_its_articles_and_an_article_counts_for_each_member():
    d1, d2 = date(2022, 1, 3), date(2022, 1, 4)
    rows, pairs = features_n3.member_days(
        pairs_of(
            [
                (10, "BBB", d1),
                (11, "AAA", d1),
                (10, "AAA", d1),
                (12, "AAA", d2),
                (12, "AAA", d2),  # 12 tags AAA twice, as BRK.B and BRK/B would
            ]
        )
    )
    # 10 counts for both members; 12 counts twice, as in N2 (94 such articles).
    assert rows.rows() == [("AAA", d1, 2), ("BBB", d1, 1), ("AAA", d2, 2)]
    assert sorted(pairs.rows()) == [(10, 0), (10, 1), (11, 0), (12, 2), (12, 2)]


def test_a_member_day_s_features_are_the_mean_of_its_articles():
    d1, d2 = date(2022, 1, 3), date(2022, 1, 4)
    rows, pairs = features_n3.member_days(
        pairs_of([(7, "AAA", d1), (3, "AAA", d1), (5, "BBB", d2)])
    )
    ids = np.array([5, 3, 7, 9])  # as the checkpoints wrote them, not sorted
    vectors = np.array(
        [[1.0, 2.0, 3.0], [10.0, 0.0, -2.0], [20.0, 4.0, 2.0], [99.0, 99.0, 99.0]],
        dtype=np.float16,
    )
    features_n3.COLUMNS = 2  # several column blocks, as with 4,096 columns
    got = features_n3.means(pairs, rows.height, ids, vectors)
    assert got.dtype == np.float16
    assert got.tolist() == [[15.0, 2.0, 0.0], [1.0, 2.0, 3.0]]


def test_an_article_without_an_embedding_stops_the_features():
    _, pairs = features_n3.member_days(pairs_of([(4, "AAA", date(2022, 1, 3))]))
    with pytest.raises(ValueError, match="no embedding"):
        features_n3.means(pairs, 1, np.array([5]), np.zeros((1, 2), np.float16))


def test_an_article_read_by_two_checkpoints_is_refused(tmp_path):
    for code, ids in (("201511", [1, 2]), ("201612", [2, 3])):
        np.save(tmp_path / f"{code}.ids.npy", np.array(ids))
        np.save(tmp_path / f"{code}.last.npy", np.zeros((2, 4), np.float16))
        np.save(tmp_path / f"{code}.mean.npy", np.zeros((2, 4), np.float16))
    with pytest.raises(ValueError, match="two checkpoints"):
        features_n3.stacked(tmp_path, "last")
    np.save(tmp_path / "201612.ids.npy", np.array([3, 4]))
    ids, vectors = features_n3.stacked(tmp_path, "mean")
    assert ids.tolist() == [1, 2, 3, 4] and vectors.shape == (4, 4)
