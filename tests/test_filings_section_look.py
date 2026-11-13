"""The last look, on the risk factors and MD&A (research/filings/section_look.py)."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

HERE = Path(__file__).parents[1] / "research" / "filings"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location(
    "filings_section_look", HERE / "section_look.py"
)
sl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sl)
_world = importlib.util.spec_from_file_location(
    "first_look_tests", Path(__file__).parent / "test_filings_first_look.py"
)
first_look_tests = importlib.util.module_from_spec(_world)
_world.loader.exec_module(first_look_tests)


@pytest.fixture(autouse=True)
def from_2021(monkeypatch):
    monkeypatch.setattr(sl.fl, "FIRST_MONTH", date(2021, 1, 1))


def world(signal=0.02):
    """The first look's test world, its reports given two sections."""
    prices, change, members = first_look_tests.world(signal)
    sections = change.with_columns(
        pl.col("cosine").alias("risk_cosine"),
        (1 - pl.col("cosine")).alias("mdna_cosine"),  # the other way round
        pl.lit(300).alias("risk_words"),
        pl.lit(300).alias("earlier_risk_words"),
        pl.lit(4700).alias("mdna_words"),
        pl.lit(4700).alias("earlier_mdna_words"),
    )
    return prices, sections, members


def test_the_first_look_s_filter_counts_the_sections_words():
    assert sl.fl.MIN_WORDS == sl.MIN_WORDS == 1_000
    prices, sections, members = world()
    few = sections.with_columns(pl.lit(900).alias("earlier_words"))
    assert sl.fl.monthly(prices, sl.scored(few), members).is_empty()


def test_a_report_without_a_score_or_a_section_is_left_out():
    rows = pl.DataFrame(
        {
            "cosine": [0.9, None, float("nan"), 0.8],
            "risk_cosine": [0.9, 0.9, 0.9, float("nan")],
            "risk_words": [300, 300, 300, 0],
            "earlier_risk_words": [300, 300, 300, 0],
        }
    )
    assert sl.scored(rows)["cosine"].to_list() == [0.9, 0.8]
    assert sl.alone(rows, "risk")["cosine"].to_list() == [0.9]
    thin = rows.with_columns(pl.lit(99).alias("earlier_risk_words"))
    assert sl.alone(thin, "risk").is_empty()


def test_the_gate_decides_whether_a_test_on_2022_2026_is_written():
    assert "worth writing down" in sl.verdict(2.0)
    assert "B is closed" in sl.verdict(1.99)
    assert sl.GATE == 2.0


def test_the_look_reads_nothing_after_2021_and_finds_a_planted_edge():
    prices, sections, members = world()
    lines = sl.look(prices, sections, members)
    after = next(line for line in lines if line.startswith("After costs"))
    assert float(after.split()[2]) > 100  # bps a month: the planted 2% a month
    assert (
        next(line for line in lines if line.startswith("MD&A alone")).split()[2][0]
        == "-"
    )
    later = pl.col("date") > sl.fl.LAST_SEEN
    wrecked = prices.with_columns(
        pl.when(later)
        .then(pl.col("close") * 1000)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    filed = pl.col("filed") > sl.fl.LAST_SEEN
    changed = sections.with_columns(
        pl.when(filed).then(0.0).otherwise(pl.col("cosine")).alias("cosine")
    )
    assert sl.look(wrecked, changed, members) == lines
    # a later report with no score, or none of its own, changes nothing
    unscored = sections.head(40).with_columns(
        (pl.col("accession") + "-z").alias("accession"),
        pl.lit(None, pl.Float64).alias("cosine"),
    )
    assert sl.look(prices, pl.concat([sections, unscored]), members) == lines
    assert "10-K reports only                 no months" in lines  # the world has none
