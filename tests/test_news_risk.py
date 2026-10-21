"""N4's volatilities and statistical sectors (research/news_reader/risk.py)."""

import importlib.util
from datetime import date

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.sessions import sessions

_spec = importlib.util.spec_from_file_location(
    "risk",
    __import__("pathlib").Path(__file__).parents[1] / "research/news_reader/risk.py",
)
risk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(risk)

DAYS = [s.day for s in sessions(date(2016, 1, 4), date(2017, 3, 31))]


def prices_of(paths: dict[str, np.ndarray], members=None) -> pl.DataFrame:
    """Closes from each ticker's daily returns, every ticker a member unless listed."""
    rows = []
    for ticker, r in paths.items():
        close = 100 * np.cumprod(1 + r)
        for day, c in zip(DAYS, close, strict=False):
            member = True if members is None else ticker in members.get(day.year, ())
            rows.append((day, ticker, float(c), member))
    return pl.DataFrame(rows, schema=["date", "ticker", "close", "sp500"], orient="row")


def test_a_volatility_ends_the_session_before_its_day():
    r = np.random.default_rng(0).normal(0, 0.01, len(DAYS))
    r[30] = 0.5  # a jump on day 30: not in day 30's volatility, in day 31's
    vol = risk.volatility(prices_of({"AAA": r})).sort("day")["vol"].to_numpy()
    returns = np.r_[np.nan, r[1:]]  # day i's return is r[i]; day 0 has none
    assert np.isnan(vol[20])  # 19 returns before day 20: not enough
    assert vol[21] == pytest.approx(np.std(returns[1:21], ddof=1))
    assert vol[30] == pytest.approx(np.std(returns[10:30], ddof=1))
    assert vol[31] == pytest.approx(np.std(returns[11:31], ddof=1))
    assert vol[31] > 5 * vol[30]


def two_blocks(seed=1, n=12):
    length = len(DAYS)
    g = np.random.default_rng(seed)
    a, b = g.normal(0, 0.01, length), g.normal(0, 0.01, length)
    paths = {}
    for i in range(n):
        factor = a if i < n // 2 else b
        paths[f"{'A' if i < n // 2 else 'B'}{i:02d}"] = factor + g.normal(
            0, 0.003, length
        )
    return paths


def test_sectors_never_mix_stocks_that_move_with_different_factors():
    groups = risk.sectors(prices_of(two_blocks()), 2017)
    assert len(groups) == 12 and set(groups.values()) <= set(range(1, 12))
    for label in set(groups.values()):
        letters = {t[0] for t, g in groups.items() if g == label}
        assert len(letters) == 1  # each group within one block


def test_sectors_use_nothing_after_the_year_s_first_session():
    paths = two_blocks()
    before = risk.sectors(prices_of(paths), 2017)
    first = DAYS.index(next(d for d in DAYS if d.year == 2017))
    changed = {t: r.copy() for t, r in paths.items()}
    for r in changed.values():
        r[first:] = np.random.default_rng(9).normal(0, 0.05, len(r) - first)
    assert risk.sectors(prices_of(changed), 2017) == before


def test_a_member_without_the_history_or_new_in_the_year_is_in_group_zero():
    paths = two_blocks(n=4)
    paths["NEW"] = np.random.default_rng(3).normal(0, 0.01, len(DAYS))
    members = {
        2016: {"A00", "A01", "B02", "B03"},
        2017: {"A00", "A01", "B02", "B03", "NEW"},
    }
    prices = prices_of(paths, members)
    # NEW traded all along but is only a member from 2017: on the first session it is one,
    # with its history, so it is grouped; drop its early closes and it has none.
    young = prices.filter(
        ~((pl.col("ticker") == "NEW") & (pl.col("date") < date(2016, 12, 1)))
    )
    # LATE has all its history but joins the index after the year's first session.
    late = prices.filter(pl.col("ticker") == "A00").with_columns(
        ticker=pl.lit("LATE"), sp500=pl.col("date") >= date(2017, 2, 1)
    )
    table = risk.table(pl.concat([young, late]), 2017)
    for ticker in ("NEW", "LATE"):
        assert table.filter(pl.col("ticker") == ticker)[
            "sector"
        ].unique().to_list() == [0]
    others = table.filter(~pl.col("ticker").is_in(["NEW", "LATE"]))["sector"]
    assert 0 not in others.unique().to_list()
