"""How much each report changed (research/filings/change.py)."""

import gzip
import importlib.util
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl
import pytest

_spec = importlib.util.spec_from_file_location(
    "filings_change",
    Path(__file__).parents[1] / "research" / "filings" / "change.py",
)
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)


def test_words_are_lower_case_letters_of_two_or_more():
    assert fc.words("Net Sales rose 12% in Q3, a net-gain") == Counter(
        {"net": 2, "sales": 1, "rose": 1, "in": 1, "gain": 1}
    )


def test_cosine_and_jaccard_by_hand():
    a, b = Counter({"x": 3, "y": 4}), Counter({"x": 3, "z": 4})
    assert fc.cosine(a, b) == pytest.approx(9 / 25)
    assert fc.jaccard(a, b) == pytest.approx(1 / 3)
    assert fc.cosine(a, a) == pytest.approx(1) and fc.jaccard(a, a) == 1
    assert (
        fc.cosine(a, Counter({"z": 1})) == 0 and fc.jaccard(a, Counter({"z": 1})) == 0
    )
    assert fc.cosine(a, Counter()) != fc.cosine(a, Counter())  # nan: nothing to compare
    # the order does not matter, whichever text is the longer
    long = Counter({"x": 1, "y": 2, "z": 3, "w": 4})
    assert fc.cosine(a, long) == pytest.approx(fc.cosine(long, a))


def report(id, form, end, cik=1, filed=None):
    return {
        "cik": cik,
        "accession": id,
        "form": form,
        "filed": filed or date.fromisoformat(end),
        "report": end,
        "document": "d.htm",
    }


def test_the_earlier_report_is_the_same_form_about_a_year_before():
    rows = [
        report("k15", "10-K", "2015-12-31"),
        report("k16", "10-K", "2016-12-31"),
        report("q15", "10-Q", "2015-03-31"),
        report("q15b", "10-Q", "2015-06-30"),
        report("q16", "10-Q", "2016-03-31"),
    ]
    assert fc.earlier_report(rows[1], rows)["accession"] == "k15"
    assert (
        fc.earlier_report(rows[4], rows)["accession"] == "q15"
    )  # not the next quarter
    assert fc.earlier_report(rows[0], rows) is None
    gap = [report("k14", "10-K", "2014-12-31"), report("k16", "10-K", "2016-12-31")]
    assert fc.earlier_report(gap[1], gap) is None  # two years is not a year
    close = [
        report("a", "10-K", "2015-11-30"),
        report("b", "10-K", "2015-12-28"),
        report("c", "10-K", "2016-12-31"),
    ]
    assert (
        fc.earlier_report(close[2], close)["accession"] == "b"
    )  # the closest to a year
    nameless = report("n", "10-K", "2016-12-31") | {"report": None}
    assert fc.earlier_report(nameless, close) is None


def test_a_chain_is_a_business_and_its_older_registrants():
    members = pl.DataFrame(
        {
            "cik": [5, 5, 7, 7, None],
            "also": [[], [6], [], [], []],
        },
        schema_overrides={"cik": pl.Int64, "also": pl.List(pl.Int64)},
    )
    assert fc.chains(members) == [(5, 6), (7,)]
    # a registrant listed as another's older one is not a business of its own
    both = pl.DataFrame(
        {"cik": [5, 6], "also": [[6], []]},
        schema_overrides={"cik": pl.Int64, "also": pl.List(pl.Int64)},
    )
    assert fc.chains(both) == [(5, 6)]


def put(root, row, text):
    path = root / str(row["cik"]) / f"{row['accession']}.txt.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode()))


def test_a_chain_is_scored_across_its_registrants_and_gaps_are_skipped(tmp_path):
    old = report("old", "10-K", "2015-12-31", cik=6)
    new = report("new", "10-K", "2016-12-31", cik=5)
    same = report("same", "10-K", "2017-12-31", cik=5)
    lost = report("lost", "10-K", "2018-12-31", cik=5)  # its text was not read
    put(tmp_path, old, "alpha beta beta")
    put(tmp_path, new, "alpha beta gamma")
    put(tmp_path, same, "alpha beta gamma")
    got = fc.scores([old, new, same, lost], tmp_path)
    assert [r["accession"] for r in got] == ["new", "same"]
    assert got[0]["earlier"] == "old" and got[1]["earlier"] == "new"
    assert got[0]["cosine"] == pytest.approx(3 / (5**0.5 * 3**0.5))
    assert got[0]["jaccard"] == pytest.approx(2 / 3)
    assert got[1]["cosine"] == pytest.approx(1) and got[1]["jaccard"] == 1
    assert got[0]["words"] == 3 and got[0]["earlier_words"] == 3
