"""Which company filed for each member (research/filings/members.py)."""

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

_spec = importlib.util.spec_from_file_location(
    "members_of_filings",
    Path(__file__).parents[1] / "research" / "filings" / "members.py",
)
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)


def test_a_filing_is_named_after_the_ticker_its_instance_begins_with():
    assert fm.named_after("aapl-20150926.xml") == "aapl"
    assert fm.named_after("aapl-20250927_htm.xml") == "aapl"
    assert fm.named_after("brk.b-20151231.xml") == "brkb"
    assert fm.named_after("msft-201506.xml") == "msft"
    assert fm.named_after("report.xml") is None
    assert fm.named_after(None) is None


def test_a_ticker_is_compared_without_its_old_mark_punctuation_or_case():
    assert fm.plain("FB_old") == "fb"
    assert fm.plain("ALTR_old1") == "altr"
    assert fm.plain("BRK.B") == fm.plain("BRK-B") == fm.plain("BRK/B") == "brkb"


def test_a_member_runs_from_its_first_to_its_last_day_in_the_index():
    prices = pl.DataFrame(
        {
            "date": [
                date(2016, 1, 4),
                date(2016, 1, 5),
                date(2016, 1, 6),
                date(2016, 1, 7),
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "sp500": [False, True, True, False],
        }
    )
    assert fm.members(prices).rows() == [("AAA", date(2016, 1, 5), date(2016, 1, 6))]


def filings(*rows):
    frame = pl.DataFrame(
        [
            {
                "adsh": str(k),
                "cik": str(cik),
                "name": name,
                "form": form,
                "filed": filed,
                "instance": instance,
            }
            for k, (cik, name, form, filed, instance) in enumerate(rows)
        ]
    )
    return fm.filings_named_after(frame)


def test_only_annual_and_quarterly_reports_named_after_a_ticker_are_kept():
    got = filings(
        (1, "A", "10-K", "20160301", "aaa-20151231.xml"),
        (1, "A", "8-K", "20160301", "aaa-20160301.xml"),
        (1, "A", "10-Q", "20160501", "report.xml"),
    )
    assert got["form"].to_list() == ["10-K"]
    assert got["named"].to_list() == ["aaa"]
    assert got["filed"].to_list() == [date(2016, 3, 1)]
    assert got["cik"].to_list() == [1]


def member(ticker, first=date(2017, 1, 3), last=date(2018, 6, 1)):
    return {"ticker": ticker, "first": first, "last": last}


def test_a_ticker_used_by_two_companies_goes_to_the_one_of_its_years():
    # XYZ meant company 1 in 2017 and company 2 from 2024: the ticker was used again.
    got = filings(
        (1, "Old Co", "10-K", "20170301", "xyz-20161231.xml"),
        (1, "Old Co", "10-K", "20180301", "xyz-20171231.xml"),
        (2, "New Co", "10-K", "20240301", "xyz-20231231.xml"),
    )
    assert fm.company_in_its_years(member("XYZ_old"), got) == (1, "Old Co")
    later = member("XYZ", date(2024, 1, 2), date(2026, 1, 2))
    assert fm.company_in_its_years(later, got) == (2, "New Co")
    assert fm.company_in_its_years(member("QQQQ"), got) is None


def test_the_window_runs_400_days_before_joining_and_120_after_leaving():
    first, last = date(2017, 1, 3), date(2018, 6, 1)
    before, after = fm.BEFORE, fm.AFTER

    def one(day):
        return filings((1, "A", "10-K", day.strftime("%Y%m%d"), "abc-20161231.xml"))

    assert fm.company_in_its_years(member("ABC"), one(first - before))
    assert not fm.company_in_its_years(
        member("ABC"), one(first - before - timedelta(days=1))
    )
    assert fm.company_in_its_years(member("ABC"), one(last + after))
    assert not fm.company_in_its_years(
        member("ABC"), one(last + after + timedelta(days=1))
    )


def test_the_company_with_the_most_filings_named_after_the_ticker_wins():
    got = filings(
        (1, "Few", "10-Q", "20170501", "abc-20170331.xml"),
        (2, "Many", "10-Q", "20170501", "abc-20170331.xml"),
        (2, "Many", "10-K", "20170301", "abc-20161231.xml"),
    )
    assert fm.company_in_its_years(member("ABC"), got) == (2, "Many")


def test_who_a_member_is_goes_by_the_rules_in_their_order(monkeypatch):
    monkeypatch.setattr(fm, "BY_HAND", {"HND": 3})
    monkeypatch.setattr(fm, "ALSO", {"BBB": (55,)})
    got = filings(
        (1, "Then", "10-K", "20170301", "aaa-20161231.xml"),
        (5, "Same", "10-K", "20170301", "bbb-20161231.xml"),
        (6, "File name", "10-K", "20250301", "now-20241231.xml"),
        (6, "File name", "10-K", "20260301", "now-20251231.xml"),
        (3, "Checked", "10-K", "20170301", "zzz-20161231.xml"),
        (4, "Wrong", "10-K", "20170301", "hnd-20161231.xml"),
    )
    latest = date(2026, 10, 12)
    rows = pl.DataFrame(
        [
            member("AAA"),  # left; today's list has passed the ticker on
            member("BBB"),  # left; both agree
            member("CCC"),  # left, no filing named after it: today's list
            member("DDD_old"),  # today's DDD is another company: not taken for an _old
            member("EEE"),  # nobody
            member("HND"),  # by hand, whatever the filings say
            member("NOW", date(2025, 1, 2), latest),  # still a member: today's list
        ]
    )
    today = {
        "aaa": (9, "Now"),
        "bbb": (5, "Same Inc"),
        "ccc": (7, "Listed"),
        "ddd": (8, "Other"),
        "now": (2, "Trades under it"),
    }
    table = fm.match(rows, got, today)
    assert table.select("ticker", "cik", "name", "how").rows() == [
        ("AAA", 1, "Then", "filings, against today's list"),
        ("BBB", 5, "Same", "filings"),
        ("CCC", 7, "Listed", "today's list"),
        ("DDD_old", None, None, "none"),
        ("EEE", None, None, "none"),
        ("HND", 3, "Checked", "by hand"),
        ("NOW", 2, "Trades under it", "today's list"),
    ]
    assert table["also"].to_list()[1] == [55] and table["also"].to_list()[0] == []
