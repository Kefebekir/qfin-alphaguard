"""KUTBIT's data set (research/kutbit/dataset.py)."""

import importlib.util
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qfin_alphaguard.kutbit.features import FULL_SESSION, inputs
from qfin_alphaguard.sessions import sessions

HERE = Path(__file__).parents[1] / "research" / "kutbit"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("kutbit_dataset", HERE / "dataset.py")
kd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kd)

HALF_DAY = sessions(date(2024, 7, 3), date(2024, 7, 3))[0]  # closes at 13:00 New York


def bars(session, minutes, price=100.0, volume=1000):
    """Stored bars at the given minutes after the open, each a little above the one before."""
    return pl.DataFrame(
        [
            {
                "start": session.open + timedelta(minutes=m),
                "ticker": "AAA",
                "open": price + m * 0.01,
                "high": price + m * 0.01 + 0.05,
                "low": price + m * 0.01 - 0.05,
                "close": price + m * 0.01 + 0.02,
                "volume": volume,
                "source": "test",
            }
            for m in minutes
        ]
    )


def test_a_session_is_put_on_a_grid_of_one_bar_a_minute():
    stored = bars(HALF_DAY, [-1, 1, 2, 4, 210])  # one before the open, one at the close
    grid = kd.on_grid(stored, HALF_DAY)
    assert len(grid["close"]) == HALF_DAY.minutes == 210
    assert grid["seen"][:6].tolist() == [False, True, True, False, True, False]
    assert not grid["seen"][5:].any()
    # before the first trade, its open; after a trade, its close; no volume when unseen
    assert grid["close"][0] == grid["high"][0] == grid["open"][1] == 100.01
    assert (
        grid["open"][3] == grid["low"][3] == grid["close"][2] == pytest.approx(100.04)
    )
    assert (grid["close"][5:] == grid["close"][4]).all()
    assert grid["volume"][[0, 3, 5]].tolist() == [0, 0, 0] and grid["volume"][1] == 1000
    assert kd.on_grid(bars(HALF_DAY, [-5, 300]), HALF_DAY) is None


def test_a_bar_whose_open_or_close_lies_outside_its_range_is_widened():
    grid = {
        "open": np.array([100.0, 101.0]),
        "high": np.array([101.0, 100.5]),  # the second's high is below its open
        "low": np.array([99.0, 100.0]),
        "close": np.array([100.5, 100.2]),
    }
    assert kd.repaired(grid) == 1
    assert grid["high"].tolist() == [101.0, 101.0] and grid["low"].tolist() == [
        99.0,
        100.0,
    ]


def test_an_etf_split_is_found_from_a_clean_overnight_ratio():
    days = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    opens = [
        100.0,
        50.2,
        44.0,
        44.5,
    ]  # a 2-for-1 split on the 3rd; a 12% fall on the 4th
    closes = [100.0, 50.0, 44.2, 44.8]
    factor, found = kd.etf_splits(days, opens, closes)
    assert [factor[d] for d in days] == [2.0, 1.0, 1.0, 1.0]
    assert found == [(date(2024, 1, 3), 2)]


def history(closes, volume=1000.0):
    return [
        {"close": np.array([c]), "volume": np.full(FULL_SESSION, volume)}
        for c in closes
    ]


def test_what_was_known_before_a_session_is_put_on_its_basis():
    # 21 sessions before a 10-for-1 split: prices ten times higher, volumes ten times lower
    closes = [1000.0 * (1.01 if i % 2 else 1.0) for i in range(21)]
    prev_close, vol, profile = kd.lookback(history(closes, 100.0), [10.0] * 21, 1.0)
    assert prev_close == pytest.approx(closes[-1] / 10)
    returns = np.diff(np.log(np.array(closes) / 10))
    assert vol == pytest.approx(np.std(returns, ddof=1))
    assert (profile == 1000.0).all()  # in the session's own shares
    assert kd.lookback(history(closes)[:20], [1.0] * 20, 1.0) is None
    missing = history(closes)
    missing[3] = None
    assert kd.lookback(missing, [1.0] * 21, 1.0) is None


def test_a_half_day_before_counts_only_its_own_minutes_in_the_profile():
    sessions_ = history([100.0] * 21, 1000.0)
    sessions_[-1] = {"close": np.array([100.0]), "volume": np.full(210, 4000.0)}
    _, _, profile = kd.lookback(sessions_, [1.0] * 21, 1.0)
    # the median of 19 sessions at 1000 and one at 4000, then of 19 alone
    assert profile[:210].tolist() == [1000.0] * 210 and profile[300] == 1000.0


def test_headlines_are_counted_by_the_minute_they_were_first_published():
    session = sessions(date(2024, 2, 5), date(2024, 2, 5))[0]
    previous = sessions(date(2024, 2, 2), date(2024, 2, 2))[0]
    start = kd.minute_of(session.open)
    times = np.array(
        [
            kd.minute_of(previous.close) - 1,  # before the previous close
            kd.minute_of(previous.close),  # the previous close's minute
            start - 30,  # before the open
            start,  # the open's minute
            start + 30,
            start + 30,
            start + session.minutes,  # the close: after the session
        ]
    )
    per_minute, before_open = kd.counted(times, session, previous)
    assert len(per_minute) == 390 and per_minute.dtype == np.uint16
    assert per_minute[0] == 1 and per_minute[30] == 2 and per_minute.sum() == 3
    assert before_open == 2
    assert kd.counted(times, session, None)[1] == 0


def test_only_headlines_naming_one_or_two_symbols_count(tmp_path):
    lines = [
        {"created_at": "2024-02-05T15:00:20Z", "symbols": ["AAA"]},
        {"created_at": "2024-02-05T15:01:59Z", "symbols": ["BRK.B", "AAA"]},
        {"created_at": "2024-02-05T15:02:00Z", "symbols": ["AAA", "BBB", "CCC"]},
        {"created_at": "2024-02-05T15:03:00Z", "symbols": []},
        {"created_at": "2024-02-05T15:04:00Z", "symbols": ["ZZZ"]},
    ]
    (tmp_path / "2024-02.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    found = kd.headline_minutes({"AAA", "BRK-B"}, tmp_path)
    minute = int(datetime(2024, 2, 5, 15, 0, tzinfo=UTC).timestamp() // 60)
    assert found["AAA"].tolist() == [minute, minute + 1]
    assert found["BRK-B"].tolist() == [minute + 1]
    assert set(found) == {"AAA", "BRK-B"}


def test_an_earlier_company_s_headlines_go_by_its_symbol():
    assert kd.news_symbol("FB_old") == "FB" and kd.news_symbol("CHK_old1") == "CHK"
    assert kd.news_symbol("BRK-B") == "BRK-B"


def test_the_vix_known_at_a_session_is_the_close_of_the_day_before(tmp_path):
    path = tmp_path / "VIX.csv"
    path.write_text(
        "DATE,OPEN,HIGH,LOW,CLOSE\n01/31/2024,1,1,1,14\n02/02/2024,1,1,1,15\n"
    )
    days, values = kd.vix_closes(path)
    assert kd.last_before(days, values, date(2024, 2, 2)) == 14
    assert kd.last_before(days, values, date(2024, 2, 5)) == 15
    assert kd.last_before(days, values, date(2024, 1, 31)) is None


def write_minutes(raw, ticker, first, last, price, split_day=None):
    """Full sessions of bars from `first` to `last`; at `split_day` a 2-for-1 split halves the
    prices as traded and doubles the volume."""
    rows = []
    for s in sessions(first, last):
        ratio = 2.0 if split_day and s.day >= split_day else 1.0
        level = price * (1 + 0.002 * (s.day.toordinal() % 5)) / ratio
        for m in range(s.minutes):
            p = level * (1 + 0.001 * math.sin(m / 7))
            rows.append(
                {
                    "start": s.open + timedelta(minutes=m),
                    "ticker": ticker,
                    "open": p,
                    "high": p * 1.0005,
                    "low": p * 0.9995,
                    "close": p,
                    "volume": int(1000 * ratio),
                    "source": "test",
                }
            )
    frame = pl.DataFrame(rows)
    folder = raw / "intraday" / ticker
    folder.mkdir(parents=True)
    years = frame.with_columns(pl.col("start").dt.year().alias("year"))
    for (year,), part in years.partition_by(
        "year", as_dict=True, include_key=False
    ).items():
        part.write_parquet(folder / f"{year}.parquet")


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """A data folder: AAA joins the universe in 2024 and splits 2-for-1 on 1 February; QQQ's
    minutes reach back to November 2023; BBB, also in the universe, has no minutes."""
    raw, news = tmp_path / "raw", tmp_path / "news"
    split = date(2024, 2, 1)
    write_minutes(raw, "AAA", date(2024, 1, 2), date(2024, 2, 29), 200.0, split)
    write_minutes(raw, "QQQ", date(2023, 11, 1), date(2024, 2, 29), 400.0)
    days = [s.day for s in sessions(date(2023, 11, 1), date(2024, 2, 29))]
    pl.DataFrame(
        {
            "ticker": ["AAA"] * len(days),
            "date": days,
            "split_factor": [2.0 if d < split else 1.0 for d in days],
        }
    ).write_parquet(raw / "prices_eodhd.parquet")
    (raw / "cboe").mkdir()
    vix = "\n".join(
        f"{(date(2023, 10, 1) + timedelta(n)).strftime('%m/%d/%Y')},1,1,1,{15 + n % 3}"
        for n in range(160)
    )
    (raw / "cboe" / "VIX_History.csv").write_text("DATE,OPEN,HIGH,LOW,CLOSE\n" + vix)
    news.mkdir()
    lines = [
        {"created_at": "2024-02-05T15:00:20Z", "symbols": ["AAA"]},  # 10:00 New York
        {"created_at": "2024-02-05T12:00:00Z", "symbols": ["AAA"]},  # before the open
    ]
    (news / "2024-02.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    universe = {2024: ("AAA", "BBB")}  # BBB has no minutes stored
    monkeypatch.setattr(kd, "trading_universes", lambda prices, years: universe)
    monkeypatch.setattr(kd, "ETFS", ("QQQ",))
    return kd.build(raw, news, years=[2024])


def test_the_data_set_holds_each_instrument_s_sessions_of_its_years(stored):
    days, arrays, notes = stored
    assert set(days["ticker"]) == {"AAA", "QQQ"} and set(days["year"]) == {2024}
    assert days.filter(pl.col("ticker") == "QQQ")["day"].min() == date(2024, 1, 2)
    assert arrays[2024]["close"].shape == (days.height, FULL_SESSION)
    assert arrays[2024]["close"].dtype == np.float32
    assert (
        arrays[2024]["news"].dtype == np.uint16 and arrays[2024]["seen"].dtype == bool
    )
    assert days["etf"].to_list() == [t == "QQQ" for t in days["ticker"]]
    assert notes == ["BBB: in 2024's instruments, but no minutes stored then"]


def test_a_stock_new_to_the_universe_is_usable_from_its_22nd_session(stored):
    days, _, _ = stored
    aaa = days.filter(pl.col("ticker") == "AAA").sort("day")
    assert aaa["usable"].to_list() == [False] * 21 + [True] * (aaa.height - 21)
    assert days.filter(pl.col("ticker") == "QQQ")["usable"].all()


def test_a_split_neither_moves_the_previous_close_nor_the_volume_profile(stored):
    days, arrays, _ = stored
    row = days.filter(
        (pl.col("ticker") == "AAA") & (pl.col("day") == date(2024, 2, 1))
    ).row(0, named=True)
    before = days.filter(
        (pl.col("ticker") == "AAA") & (pl.col("day") == date(2024, 1, 31))
    ).row(0, named=True)
    raw_close = arrays[2024]["close"][before["row"], before["minutes"] - 1]
    assert row["prev_close"] == pytest.approx(raw_close / 2, rel=1e-6)
    assert row["vol"] < 0.01  # the split is no 50% fall
    assert row["split"] == 1.0 and before["split"] == 2.0
    # 1000 shares a minute before the split are 2000 of today's
    assert (arrays[2024]["profile"][row["row"]] == 2000).all()


def test_headlines_and_the_vix_reach_the_session(stored):
    days, arrays, _ = stored
    row = days.filter(
        (pl.col("ticker") == "AAA") & (pl.col("day") == date(2024, 2, 5))
    ).row(0, named=True)
    news = arrays[2024]["news"][row["row"]]
    assert news[30] == 1 and news.sum() == 1 and row["news_before"] == 1
    assert row["vix"] == 15 + (date(2024, 2, 4) - date(2023, 10, 1)).days % 3


def test_a_stored_session_becomes_the_day_features_reads(stored):
    days, arrays, notes = stored
    day = kd.day_of(days, arrays, "AAA", date(2024, 2, 5))
    market = kd.day_of(days, arrays, "QQQ", date(2024, 2, 5))
    windows, scalars = inputs(day, market)
    assert windows.shape[0] == scalars.shape[0] == 390 and np.isfinite(windows).all()
    text = kd.report(days, arrays, notes, checks=5)
    assert "equal those from the whole session on 5 of 5" in text
