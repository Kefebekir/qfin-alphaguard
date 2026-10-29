"""N6's record (research/news_reader/n6_record.py): its inputs, scale and page."""

import importlib.util
import sys
import types
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("n6_record", HERE / "n6_record.py")
n6r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n6r)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def article(id, created, updated=None, symbols=("AAA",)):
    return {
        "id": id,
        "created_at": created,
        "updated_at": updated or created,
        "symbols": list(symbols),
        "headline": f"h{id}",
    }


def test_202412_s_reading_is_scaled_as_n6_s_eyes_scale_it():
    eyes = {
        "mean_202412": np.array([1.0, 2.0], np.float32),
        "spread_202412": np.array([2.0, 4.0], np.float32),
    }
    out = n6r.scaled(np.array([[3.0, 10.0]], np.float16), eyes)
    assert out.dtype == np.float16 and out.tolist() == [[1.0, 2.0]]


def test_a_member_day_s_last_headlines_go_in_with_their_age_and_flags():
    # Monday 19 October 2026: 15:30 in New York is 19:30 UTC
    articles = [
        article(1, "2026-10-19T13:30:00Z"),
        # known from its edit
        article(2, "2026-10-19T07:00:00Z", "2026-10-19T17:30:00Z"),
        article(3, "2026-10-19T15:30:00Z", symbols=("AAA", "BBB")),
    ]
    pairs = [(1, "AAA", "h1"), (3, "AAA", "h3"), (3, "BBB", "h3"), (2, "AAA", "h2")]
    vectors = {i: np.full(4, float(i), np.float16) for i in (1, 2, 3)}
    tickers, x, store = n6r.member_inputs(
        articles, pairs, date(2026, 10, 19), vectors, most=2
    )
    assert tickers == ["AAA", "BBB"]
    assert store.dtype == np.float16
    assert store[:, 0].tolist() == [2.0, 3.0]  # ids 2 and 3
    # AAA's last two by the time they became known: 3 (15:30Z), then 2 (17:30Z)
    assert store[x["heads"][0], 0].tolist() == [3.0, 2.0]
    assert x["heads"][1].tolist() == [1, -1]
    np.testing.assert_allclose(x["info"][0, :, 0], np.log1p([4.0, 2.0]), rtol=1e-6)
    assert x["info"][0, :, 1].tolist() == [0.0, 1.0]  # the second counts from its edit
    assert x["info"][0, :, 2].tolist() == [1.0, 0.0]  # the first tags two stocks
    # the count keeps all three of AAA's
    np.testing.assert_allclose(x["count"], np.log1p([3, 1]))


def test_n6_is_set_against_n5_on_the_sessions_both_scored():
    days = pl.DataFrame(
        {
            "day": [date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)],
            "score": [0.01, None, 0.03],
        }
    )
    n5 = pl.DataFrame(
        {
            "day": [date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)],
            "score": [0.02, 0.01, 0.01],
        }
    )
    bps, t, both = n6r.against_n5(days, n5)
    assert both == 2 and bps == pytest.approx(((0.01 - 0.02) + (0.03 - 0.01)) / 2 * 1e4)
    assert t is not None  # two days give a t
    assert n6r.against_n5(days, None) == (0.0, None, 0)


def test_the_page_names_n6_s_test_and_shows_every_session():
    days = pl.DataFrame(
        {
            "day": [date(2026, 10, 30), date(2026, 11, 2)],
            "score": [0.012, None],
            "season": [True, True],
            "long": [0.02, None],
            "short": [0.006, None],
            "everyone": [0.01, None],
            "members": [150, None],
            "n_long": [15, None],
            "n_short": [15, None],
            "why": ["", "no capture"],
        }
    )
    card = types.SimpleNamespace(
        record_from=date(2026, 10, 30),
        frozen_at=utc(2026, 10, 29, 10, 26),
        weights_sha256="e6" * 32,
        trained_from=date(2016, 1, 5),
        trained_to=date(2026, 10, 9),
    )
    result = n6r.evidence(days)
    assert result.threshold == pytest.approx(160)
    n5 = pl.DataFrame({"day": [date(2026, 10, 30)], "score": [0.002]})
    page = n6r.page(card, days, result, utc(2026, 11, 3, 9), n5)
    assert "N6: our own network" in page and "against 160: not promoted" in page
    assert "9.2%" in page and "0.625%" in page
    assert "N6 less N5" in page and "+100.00 bps" in page
    assert "<td>2026-11-02</td>" in page and "no capture" in page
