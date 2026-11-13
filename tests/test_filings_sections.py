"""The risk factors and MD&A of each report (research/filings/sections.py)."""

import gzip
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "research" / "filings"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("filings_sections", HERE / "sections.py")
sx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sx)

CONTENTS_10K = (
    "Table of Contents Item 1. Business 3 Item 1A. Risk Factors 12 Item 1B. Unresolved "
    "Staff Comments 20 Item 2. Properties 21 Item 7. Management's Discussion and Analysis "
    "30 Item 7A. Quantitative and Qualitative Disclosures 50 Item 8. Financial Statements 52 "
)


def ten_k(risk="Our suppliers may fail us.", mdna="Sales rose in every segment."):
    return (
        CONTENTS_10K
        + "Item 1. Business We make things; see Item 1A. Risk Factors below, and as "
        + "Item 1A. Risk Factors explains, we compete. "
        f"Item 1A. Risk Factors {risk} Item 1B. Unresolved Staff Comments None. "
        "Item 2. Properties We own plants. "
        f"Item 7. Management’s Discussion and Analysis {mdna} "
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk Rates. "
        "Item 8. Financial Statements Balance sheet."
    )


def test_a_section_is_its_heading_to_the_next_item_not_the_contents_entry():
    found = sx.sections(ten_k(), "10-K")
    assert found["risk"] == "Item 1A. Risk Factors Our suppliers may fail us. "
    assert found["mdna"].startswith("Item 7. Management’s Discussion")
    assert found["mdna"].endswith("Sales rose in every segment. ")


def test_a_reference_is_not_a_heading_and_does_not_cut_a_section_short():
    risk = 'As said in "Item 1A. Risk Factors" in our last report, costs rise.'
    mdna = "Costs rose; see Item 8. Financial Statements and Note 4. Margins fell."
    found = sx.sections(ten_k(risk=risk, mdna=mdna), "10-K")
    assert found["risk"].startswith("Item 1A. Risk Factors As said in")
    assert found["mdna"].endswith("Margins fell. ")
    assert sx.RISK.search("the “Item 1A. Risk Factors” of our 10-K") is None
    assert sx.RISK.search('Item 1A. Risk Factors" in our 10-K') is None
    # a word that ends like a pointing word does not hide a heading
    assert sx.RISK.search("prices demand Item 1A. Risk Factors We face") is not None
    assert sx.RISK.search("as shown in Item 1A. Risk Factors") is None


def test_ten_q_sections_end_at_their_own_items():
    text = (
        "Item 1. Financial Statements Balance sheet. "
        "Item 2. Management's Discussion and Analysis Revenue grew. "
        "Item 3. Quantitative and Qualitative Disclosures Rates. "
        "Item 4. Controls and Procedures Effective. PART II "
        "Item 1. Legal Proceedings None. "
        "Item 1A. Risk Factors There have been no material changes. "
        "Item 2. Issuer Purchases of Equity Securities None. Item 6. Exhibits List."
    )
    found = sx.sections(text, "10-Q")
    assert (
        found["mdna"] == "Item 2. Management's Discussion and Analysis Revenue grew. "
    )
    assert (
        found["risk"] == "Item 1A. Risk Factors There have been no material changes. "
    )


def test_headings_as_some_companies_write_them():
    capital_one = (
        "Item 2. Management’s Discussion and Analysis Loans grew. "
        "Item 1. Financial Statements Balance sheet. Item 3. Quantitative Rates."
    )
    assert sx.sections(capital_one, "10-Q")["mdna"].endswith("Loans grew. ")
    freeport = (
        "Items 7. and 7A. Management’s Discussion and Analysis Copper fell. "
        "Item 8. Financial Statements Sheet."
    )
    assert sx.sections(freeport, "10-K")["mdna"].endswith("Copper fell. ")
    exelon = (
        "Item 7. Combined Management's Discussion and Analysis Power sold. "
        "Item 7A. Quantitative Rates."
    )
    assert sx.sections(exelon, "10-K")["mdna"].endswith("Power sold. ")
    split = (
        "Ite m 7: Management’s Discussion and Analysis Gas sold. Item 8. Financial S."
    )
    assert sx.sections(split, "10-K")["mdna"].endswith("Gas sold. ")


def test_a_missing_section_is_empty():
    assert sx.sections("Item 1. Business We make things.", "10-K") == {
        "risk": "",
        "mdna": "",
    }


def report(tmp_path, cik, accession, form, filed, period, text):
    folder = tmp_path / str(cik)
    folder.mkdir(exist_ok=True)
    (folder / f"{accession}.txt.gz").write_bytes(gzip.compress(text.encode()))
    return {
        "cik": cik,
        "accession": accession,
        "form": form,
        "filed": filed,
        "report": period,
        "document": "x.htm",
    }


def test_a_report_is_scored_on_its_two_sections_pooled_and_on_each(tmp_path):
    old = report(
        tmp_path,
        1,
        "a",
        "10-K",
        date(2017, 2, 1),
        "2016-12-31",
        ten_k(risk="supply supply demand", mdna="sales rose"),
    )
    new = report(
        tmp_path,
        1,
        "b",
        "10-K",
        date(2018, 2, 1),
        "2017-12-31",
        ten_k(risk="supply demand demand", mdna="sales rose"),
    )
    first, second = sx.scores([new, old], root=tmp_path)
    assert first["accession"] == "a" and "cosine" not in first
    assert first["risk_words"] == 6  # "item", "risk", "factors" and its three words
    pooled = sx.change.words(
        sx.sections(ten_k("supply demand demand", "sales rose"), "10-K")["risk"]
    )
    pooled += sx.change.words(
        sx.sections(ten_k("supply demand demand", "sales rose"), "10-K")["mdna"]
    )
    assert second["words"] == sum(pooled.values())
    assert second["earlier"] == "a"
    assert second["mdna_cosine"] == pytest.approx(1.0)
    assert second["risk_cosine"] < 1.0
    assert second["risk_cosine"] < second["cosine"] < 1.0
    assert second["earlier_risk_words"] == 6
