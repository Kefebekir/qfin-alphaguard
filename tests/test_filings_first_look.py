"""The first look at lazy prices (research/filings/first_look.py), and its wall at 2021."""

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.sessions import sessions

_spec = importlib.util.spec_from_file_location(
    "filings_first_look",
    Path(__file__).parents[1] / "research" / "filings" / "first_look.py",
)
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)

DAYS = [s.day for s in sessions(date(2021, 1, 4), date(2022, 6, 30))]
TICKERS = [f"T{i:02}" for i in range(20)]


def world(signal=0.02, seed=0):
    """Twenty members: the lower T's report changed most each quarter, and it falls; the
    higher changed least, and it rises, by `signal` a month in all."""
    g = np.random.default_rng(seed)
    n = len(TICKERS)
    drift = np.linspace(-signal, signal, n) / 21
    rows, reports = [], []
    for i, t in enumerate(TICKERS):
        close = 100 * np.cumprod(1 + drift[i] + g.normal(0, 0.002, len(DAYS)))
        rows += [(d, t, float(c), True) for d, c in zip(DAYS, close, strict=True)]
        for quarter in pl.date_range(
            date(2020, 11, 15), date(2022, 5, 15), "3mo", eager=True
        ):
            reports.append(
                {
                    "cik": 100 + i,
                    "accession": f"{i}-{quarter}",
                    "form": "10-Q",
                    "filed": quarter,
                    "cosine": 0.5 + i / 50,
                    "jaccard": 0.5 + (n - i) / 50,
                    "words": 5000,
                    "earlier_words": 5000,
                }
            )
    prices = pl.DataFrame(
        rows, schema=["date", "ticker", "close", "sp500"], orient="row"
    )
    change = pl.DataFrame(reports)
    members = pl.DataFrame(
        {
            "ticker": TICKERS,
            "cik": [100 + i for i in range(n)],
            "also": [[] for _ in TICKERS],
        },
        schema_overrides={"also": pl.List(pl.Int64)},
    )
    return prices, change, members


@pytest.fixture(autouse=True)
def from_2021(monkeypatch):
    monkeypatch.setattr(fl, "FIRST_MONTH", date(2021, 1, 1))


def test_nothing_after_2021_reaches_the_look():
    prices, change, members = world()
    base = fl.monthly(prices, change, members)
    assert base["month"].max() == date(2021, 11, 30)  # its month ends at the wall
    # Whatever 2022 holds, the look is the same, to the last digit.
    later = pl.col("date") > fl.LAST_SEEN
    wrecked = prices.with_columns(
        pl.when(later)
        .then(pl.col("close") * 1000)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    filed = pl.col("filed") > fl.LAST_SEEN
    changed = change.with_columns(
        pl.when(filed).then(0.0).otherwise(pl.col("cosine")).alias("cosine")
    )
    assert fl.monthly(wrecked, changed, members).equals(base)


def test_the_wall_is_the_data_s_only_door():
    frame = pl.DataFrame({"date": [date(2021, 12, 31), date(2022, 1, 3)]})
    assert fl.seen(frame, "date")["date"].to_list() == [date(2021, 12, 31)]
    assert fl.LAST_SEEN == date(2021, 12, 31)


def test_a_planted_edge_is_found_and_its_sign_is_the_study_s():
    prices, change, members = world()
    base = fl.monthly(prices, change, members)
    assert base["after"].mean() > 0.01  # long the least changed, short the most changed
    flipped = change.with_columns(cosine=1 - pl.col("cosine"))
    assert fl.monthly(prices, flipped, members)["after"].mean() < -0.01
    assert (
        fl.monthly(prices, change, members, measure="jaccard")["after"].mean() < -0.01
    )


def test_each_month_runs_from_its_last_close_to_the_next():
    prices, change, members = world()
    closes = prices.select("date", "ticker", "close")
    base = fl.monthly(prices, change, members)
    t, until = base["month"][0], fl.month_ends(sorted(set(DAYS)))[1]
    returns = fl.held_return(closes, t, until)
    day = closes.filter(
        (pl.col("ticker") == "T19") & pl.col("date").is_in([t, until])
    ).sort("date")
    assert returns["T19"] == pytest.approx(day["close"][1] / day["close"][0] - 1)
    long = np.mean([returns[x] for x in ("T16", "T17", "T18", "T19")])
    short = np.mean([returns[x] for x in ("T00", "T01", "T02", "T03")])
    everyone = np.mean(list(returns.values()))
    row = base.row(0, named=True)
    assert row["long"] == pytest.approx(long) and row["short"] == pytest.approx(short)
    assert row["everyone"] == pytest.approx(everyone)
    # the first month's whole book is new: both sides trade all of it
    assert row["after"] == pytest.approx(long - short - 2 * fl.COST)
    assert row["names"] == 20


def test_a_stock_that_stops_trading_is_held_at_its_last_close():
    closes = pl.DataFrame(
        {
            "date": [
                date(2021, 3, 31),
                date(2021, 4, 15),
                date(2021, 3, 31),
                date(2021, 4, 30),
            ],
            "ticker": ["GONE", "GONE", "STAYS", "STAYS"],
            "close": [100.0, 150.0, 100.0, 110.0],
        }
    )
    got = fl.held_return(closes, date(2021, 3, 31), date(2021, 4, 30))
    assert got == pytest.approx({"GONE": 0.5, "STAYS": 0.1})
    # one that was not trading on t is not held at all
    assert "GONE" not in fl.held_return(closes, date(2021, 4, 30), date(2021, 5, 31))


def reports(**by_cik):
    return {
        int(c[1:]): [
            {
                "filed": f,
                "form": form,
                "cosine": s,
                "jaccard": s,
                "accession": f"{c}-{f}",
            }
            for f, form, s in rows
        ]
        for c, rows in by_cik.items()
    }


T = date(2021, 3, 31)


def test_a_report_counts_from_the_day_after_it_was_filed_for_three_months(monkeypatch):
    chains = {f"A{i}": (i,) for i in range(5)}
    rep = reports(
        c0=[(T, "10-Q", 0.1)],  # filed on t itself: not known in time
        c1=[(T - timedelta(days=1), "10-Q", 0.2)],
        c2=[(T - fl.RECENT, "10-Q", 0.3)],  # the first day of the three months
        c3=[(T - fl.RECENT - timedelta(days=1), "10-Q", 0.4)],  # too old
        c4=[
            (T - timedelta(days=50), "10-Q", 0.5),
            (T - timedelta(days=9), "10-Q", 0.6),
        ],
    )
    assert fl.formation(
        T, list(chains), chains, rep
    ).is_empty()  # three, under MIN_REPORTS
    monkeypatch.setattr(fl, "MIN_REPORTS", 1)
    got = fl.formation(T, list(chains), chains, rep).sort("ticker")
    assert got["ticker"].to_list() == ["A1", "A2", "A4"]
    assert got["score"].to_list() == [0.2, 0.3, 0.6]  # A4's latest
    assert got["rank"].to_list() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_ranks_are_within_a_form_and_a_company_counts_once(monkeypatch):
    monkeypatch.setattr(fl, "MIN_REPORTS", 1)
    chains = {
        "GOOG": (1,),
        "GOOGL": (1,),
        "K1": (2,),
        "K2": (3,),
        "OLD": (4, 5),
        "NONE": (9,),
    }
    day = T - timedelta(days=5)
    rep = reports(
        c1=[(day, "10-Q", 0.9)],
        c2=[(day, "10-K", 0.1)],
        c3=[(day, "10-K", 0.2)],
        c5=[(day, "10-Q", 0.3)],  # the older registrant's filing counts for OLD
    )
    got = fl.formation(T, list(chains), chains, rep, "cosine").sort("ticker")
    assert got["ticker"].to_list() == ["GOOG", "K1", "K2", "OLD"]  # GOOGL is GOOG
    assert dict(zip(got["ticker"], got["rank"], strict=True)) == pytest.approx(
        {"GOOG": 1.0, "OLD": 0.5, "K1": 0.5, "K2": 1.0}
    )


def test_a_thin_document_is_left_out_on_either_side():
    change = pl.DataFrame(
        {
            "cik": [1, 2, 3],
            "filed": [T, T, T],
            "words": [5000, 100, 5000],
            "earlier_words": [5000, 5000, 100],
        }
    )
    assert list(fl.reports_by_cik(change)) == [1]


def test_only_members_are_ranked_and_only_the_tickers_given():
    prices, change, members = world()
    out = prices.with_columns(
        pl.when(pl.col("ticker") == "T19")
        .then(False)
        .otherwise(pl.col("sp500"))
        .alias("sp500")
    )
    assert fl.monthly(out, change, members)["names"][0] == 19


def test_holding_longer_blends_the_months_and_trades_less():
    prices, change, members = world()
    one = fl.monthly(prices, change, members)
    three = fl.monthly(prices, change, members, months_held=3)
    assert three.height == one.height
    # quarterly reports keep the book still in this world: no cost once it is built
    assert three["after"][5] == pytest.approx(three["before"][5])
    assert three["before"].mean() > 0
