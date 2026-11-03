"""Every member's 10-K and 10-Q as text (research/filings/download.py)."""

import gzip
import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

_spec = importlib.util.spec_from_file_location(
    "filings_download",
    Path(__file__).parents[1] / "research" / "filings" / "download.py",
)
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)


def test_a_document_keeps_its_words_and_nothing_else():
    page = b"""<html><head><title>Not this</title></head><body>
    <script>var no = 1;</script><STYLE>p { color: red }</STYLE>
    <ix:header><ix:hidden>hidden facts</ix:hidden></ix:header>
    <!-- a comment -->
    <p>Risk&nbsp;factors &amp; more</p>
    <div>Net <b>sales</b>
       rose</div></body></html>"""
    assert fd.to_text(page) == "Risk factors & more Net sales rose"


def test_a_bad_byte_does_not_lose_the_document():
    assert fd.to_text(b"<p>caf\xe9 sales</p>").endswith("sales")


def test_a_company_s_windows_cover_its_tickers_and_its_older_registrants():
    members = pl.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "first": [date(2017, 1, 3), date(2020, 1, 2), date(2018, 1, 2)],
            "last": [date(2018, 1, 2), date(2021, 1, 4), date(2019, 1, 2)],
            "cik": [1, 1, None],
            "also": [[], [9], []],
        },
        schema_overrides={"cik": pl.Int64, "also": pl.List(pl.Int64)},
    )
    got = fd.windows(members)
    assert sorted(got) == [1, 9]
    assert got[1][0] == (date(2017, 1, 3) - fd.BEFORE, date(2018, 1, 2) + fd.AFTER)
    assert len(got[1]) == 2 and got[9] == [got[1][1]]


def block(*rows):
    fields = list(zip(*rows, strict=True))
    return dict(zip(fd.FIELDS, fields, strict=True))


def test_only_reports_in_the_company_s_windows_are_kept():
    spans = [(date(2016, 1, 1), date(2017, 1, 1)), (date(2019, 1, 1), date(2019, 6, 1))]
    got = fd.filings_in(
        block(
            ("a1", "10-K", "2016-02-01", "2015-12-31", "a.htm"),
            ("a2", "10-Q", "2019-05-01", "2019-03-31", "b.htm"),
            ("a3", "10-K/A", "2016-03-01", "2015-12-31", "c.htm"),  # an amendment
            ("a4", "8-K", "2016-03-01", "", "d.htm"),
            ("a5", "10-Q", "2018-05-01", "2018-03-31", "e.htm"),  # between the windows
            ("a6", "10-Q", "2019-06-02", "2019-03-31", "f.htm"),  # a day after
            ("a7", "10-K", "2016-04-01", "", ""),  # no document
        ),
        7,
        spans,
    )
    assert [r["accession"] for r in got] == ["a1", "a2"]
    assert got[0] == {
        "cik": 7,
        "accession": "a1",
        "form": "10-K",
        "filed": date(2016, 2, 1),
        "report": "2015-12-31",
        "document": "a.htm",
    }


def test_nothing_before_2015_is_kept_whatever_the_window():
    spans = [(date(2013, 1, 1), date(2016, 1, 1))]
    got = fd.filings_in(
        block(
            ("a1", "10-K", "2014-12-31", "2014-09-30", "a.htm"),
            ("a2", "10-K", "2015-01-01", "2014-09-30", "b.htm"),
        ),
        7,
        spans,
    )
    assert [r["accession"] for r in got] == ["a2"]


ROW = {"cik": 320193, "accession": "0000320193-25-000079", "document": "aapl.htm"}


def test_a_filing_s_address_and_its_place_on_disk():
    assert fd.url_of(ROW) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl.htm"
    )
    assert fd.path_of(ROW, Path("t")) == Path("t/320193/0000320193-25-000079.txt.gz")


def test_a_filing_is_read_once_and_a_failure_is_named(tmp_path, monkeypatch):
    calls = []

    def fake(url):
        calls.append(url)
        return b"<p>Net sales</p>"

    monkeypatch.setattr(fd, "get", fake)
    assert fd.keep(ROW, tmp_path) is None
    assert fd.keep(ROW, tmp_path) is None
    assert len(calls) == 1
    path = fd.path_of(ROW, tmp_path)
    assert gzip.decompress(path.read_bytes()) == b"Net sales"
    assert not path.with_suffix(".part").exists()

    def broken(url):
        raise RuntimeError("the network")

    monkeypatch.setattr(fd, "get", broken)
    other = ROW | {"accession": "0000320193-25-000080"}
    assert fd.keep(other, tmp_path) == "the network"
    assert not fd.path_of(other, tmp_path).exists()
