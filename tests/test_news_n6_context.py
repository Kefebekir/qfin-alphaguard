"""N6's context (research/news_reader/n6_context.py): known by the session before, exactly."""

import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.sessions import sessions

_spec = importlib.util.spec_from_file_location(
    "n6_context",
    Path(__file__).parents[1] / "research" / "news_reader" / "n6_context.py",
)
nc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nc)

DAYS = [s.day for s in sessions(date(2014, 6, 2), date(2016, 3, 31))]
FIRST = date(2015, 9, 1)
DAY = date(2015, 11, 18)  # a Wednesday, an FOMC day in the fake calendar
BEFORE = DAYS[DAYS.index(DAY) - 1]


def walk(seed: int, n: int = len(DAYS), scale: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, n)


def bars(name: str, r: np.ndarray, seed: int) -> pl.DataFrame:
    g = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + r)
    return pl.DataFrame(
        {
            "date": DAYS,
            "ticker": name,
            "open": close * (1 + g.normal(0, 0.003, len(r))),
            "high": close * (1 + np.abs(g.normal(0, 0.01, len(r)))),
            "low": close * (1 - np.abs(g.normal(0, 0.01, len(r)))),
            "close": close,
            "volume": g.integers(1_000, 100_000, len(r)),
            "adjustment": g.uniform(0.9, 1.1, len(r)),
        }
    )


@pytest.fixture(scope="module")
def world():
    """SPY, the nine funds, three members (EN follows XLE), one non-member, Cboe, events."""
    market = walk(1)
    etfs = {"SPY": bars("SPY", market, 2)}
    fund_r = {}
    for k, fund in enumerate(nc.FUNDS):
        fund_r[fund] = market + walk(10 + k, scale=0.006)
        etfs[fund] = bars(fund, fund_r[fund], 20 + k)
    members = {
        "EN": fund_r["XLE"] + walk(40, scale=0.002),
        "AA": 1.5 * market + walk(41),
        "BB": walk(42, scale=0.02),
        "OUT": walk(43),
    }
    prices = pl.concat(
        [
            bars(t, r, 50 + i).with_columns(
                split_factor=pl.lit(2.0) if t == "AA" else pl.lit(1.0),
                sp500=pl.lit(t != "OUT"),
            )
            for i, (t, r) in enumerate(members.items())
        ]
    )
    g = np.random.default_rng(60)
    vix = {
        name: pl.DataFrame(
            {"date": DAYS, "close": g.uniform(12, 30, len(DAYS))}
        ).filter(pl.col("date") != BEFORE if name == "VIX9D" else pl.lit(True))
        for name in nc.VIX
    }
    events = pl.DataFrame(
        {
            "event": ["fomc", "cpi", "fomc_emergency", "nfp"],
            "day": [DAY, DAYS[DAYS.index(DAY) + 1], DAYS[DAYS.index(DAY) + 2], BEFORE],
        }
    )
    table = nc.context(prices, etfs, vix, events, first=FIRST)
    return prices, etfs, vix, events, table


def row(table, ticker, day=DAY):
    return table.filter((pl.col("day") == day) & (pl.col("ticker") == ticker)).row(
        0, named=True
    )


def series(frame, ticker=None):
    if ticker is not None:
        frame = frame.filter(pl.col("ticker") == ticker)
    frame = frame.filter(pl.col("date") <= BEFORE).sort("date")
    return {
        c: frame[c].to_numpy() for c in frame.columns if c not in ("date", "ticker")
    }


def test_nothing_from_the_day_on_reaches_it_and_the_session_before_does(world):
    prices, etfs, vix, events, table = world
    kept, moved = nc.unchanged_before(prices, etfs, vix, events, table, DAY, FIRST)
    assert kept and moved


def test_the_member_s_numbers_are_those_of_the_close_before(world):
    prices, _, _, _, table = world
    got, s = row(table, "BB"), series(prices, "BB")
    c, r = s["close"], s["close"][1:] / s["close"][:-1] - 1
    assert got["r1"] == pytest.approx(c[-1] / c[-2] - 1)
    assert got["r5"] == pytest.approx(c[-1] / c[-6] - 1)
    assert got["r20"] == pytest.approx(c[-1] / c[-21] - 1)
    assert got["r60"] == pytest.approx(c[-1] / c[-61] - 1)
    assert got["mom"] == pytest.approx(c[-22] / c[-253] - 1)
    assert got["vol20"] == pytest.approx(np.std(r[-20:], ddof=1))
    assert got["vol60"] == pytest.approx(np.std(r[-60:], ddof=1))
    assert got["range"] == pytest.approx((s["high"][-1] - s["low"][-1]) / c[-1])
    assert got["gap"] == pytest.approx(s["open"][-1] / c[-2] - 1)
    assert got["high52"] == pytest.approx(c[-1] / s["high"][-252:].max() - 1)


def test_traded_value_is_the_price_as_traded_times_the_shares(world):
    prices, _, _, _, table = world
    s = series(prices, "AA")  # its split_factor is 2
    traded = s["close"] / s["adjustment"] * s["volume"] / 2.0
    got = row(table, "AA")
    assert got["dollars"] == pytest.approx(np.log(traded[-20:].mean()))
    assert got["turnover"] == pytest.approx(np.log(traded[-1] / traded[-20:].mean()))


def test_beta_and_correlation_come_from_60_returns_against_spy(world):
    prices, etfs, _, _, table = world
    a, m = series(prices, "AA")["close"], series(etfs["SPY"])["close"]
    ra, rm = (a[1:] / a[:-1] - 1)[-60:], (m[1:] / m[:-1] - 1)[-60:]
    got = row(table, "AA")
    assert got["beta"] == pytest.approx(np.polyfit(rm, ra, 1)[0])
    assert got["corr60"] == pytest.approx(np.corrcoef(ra, rm)[0, 1])
    r5m = m[-1] / m[-6] - 1
    assert got["resid5"] == pytest.approx(got["r5"] - got["beta"] * r5m)


def test_the_sector_is_the_fund_moving_most_alike(world):
    _, etfs, _, _, table = world
    got = row(table, "EN")
    assert nc.FUNDS[got["sector"]] == "XLE"
    x = series(etfs["XLE"])["close"]
    assert got["sector_r1"] == pytest.approx(x[-1] / x[-2] - 1)
    assert got["sector_r5"] == pytest.approx(x[-1] / x[-6] - 1)
    assert got["rel5"] == pytest.approx(got["r5"] - got["sector_r5"])


def test_the_market_and_the_cboe_closes_are_the_last_by_the_session_before(world):
    _, etfs, vix, _, table = world
    got, m = row(table, "BB"), series(etfs["SPY"])["close"]
    assert got["spy_r1"] == pytest.approx(m[-1] / m[-2] - 1)
    assert got["spy_r5"] == pytest.approx(m[-1] / m[-6] - 1)
    assert got["spy_r20"] == pytest.approx(m[-1] / m[-21] - 1)
    rm = m[1:] / m[:-1] - 1
    assert got["spy_vol20"] == pytest.approx(np.std(rm[-20:], ddof=1))
    v = {k: series(f)["close"] for k, f in vix.items()}
    assert got["vix"] == v["VIX"][-1]
    assert got["vix_change5"] == pytest.approx(np.log(v["VIX"][-1] / v["VIX"][-6]))
    # VIX9D has no close on the session before: its last one, the session before that
    assert got["vix9d_ratio"] == pytest.approx(v["VIX9D"][-1] / v["VIX"][-1])
    assert got["vix3m_ratio"] == pytest.approx(v["VIX"][-1] / v["VIX3M"][-1])
    assert got["vvix"] == v["VVIX"][-1]


def test_dispersion_is_across_the_members_only(world):
    prices, _, _, _, table = world
    last = []
    for t in ("EN", "AA", "BB"):  # not OUT
        c = series(prices, t)["close"]
        last.append(c[-1] / c[-2] - 1)
    got = row(table, "EN")
    assert got["dispersion"] == pytest.approx(np.std(last, ddof=1))
    assert row(table, "AA")["dispersion"] == got["dispersion"]


def test_the_calendar_is_the_day_s_own_and_only_members_have_rows(world):
    *_, table = world
    got = row(table, "BB")
    assert (got["weekday"], got["fomc"], got["cpi"], got["nfp"]) == (
        2,
        True,
        False,
        False,
    )
    after = row(table, "BB", DAYS[DAYS.index(DAY) + 1])
    assert (after["fomc"], after["cpi"]) == (False, True)
    emergency = row(table, "BB", DAYS[DAYS.index(DAY) + 2])
    assert not emergency["fomc"]  # an emergency meeting is not scheduled ahead
    assert set(table["ticker"]) == {"EN", "AA", "BB"}
    assert table["day"].min() == min(d for d in DAYS if d >= FIRST)
    assert table.height == 3 * len([d for d in DAYS if d >= FIRST])


def test_a_member_without_a_row_the_session_before_has_no_numbers(world):
    prices, etfs, vix, events, _ = world
    gap = prices.filter(~((pl.col("ticker") == "BB") & (pl.col("date") == BEFORE)))
    table = nc.context(gap, etfs, vix, events, first=FIRST)
    got = row(table, "BB")
    assert got["r1"] is None and got["vol20"] is None
    assert got["vix"] is not None  # the market's numbers are still there


def test_dispersion20_and_comove60_average_the_members_by_the_session_before(world):
    prices, etfs, _, _, table = world
    m = series(etfs["SPY"])["close"]
    rm = m[1:] / m[:-1] - 1
    spreads, corrs = [], []
    for t in ("EN", "AA", "BB"):
        c = series(prices, t)["close"]
        r = c[1:] / c[:-1] - 1
        spreads.append(r[-20:])
        corrs.append(np.corrcoef(r[-60:], rm[-60:])[0, 1])
    daily = np.std(np.array(spreads), axis=0, ddof=1)
    got = row(table, "BB")
    assert got["dispersion20"] == pytest.approx(daily.mean())
    assert got["comove60"] == pytest.approx(np.mean(corrs))


def test_the_52_week_high_looks_back_252_sessions_and_no_more(world):
    prices, etfs, vix, events, _ = world
    at = DAYS.index(BEFORE)
    inside, outside = DAYS[at - 251], DAYS[at - 252]
    peak = pl.col("ticker") == "BB"
    raised = prices.with_columns(
        high=pl.when(peak & (pl.col("date") == inside))
        .then(1_000.0)
        .when(peak & (pl.col("date") == outside))
        .then(2_000.0)
        .otherwise(pl.col("high"))
    )
    table = nc.context(raised, etfs, vix, events, first=FIRST)
    close = series(prices, "BB")["close"][-1]
    assert row(table, "BB")["high52"] == pytest.approx(close / 1_000.0 - 1)
