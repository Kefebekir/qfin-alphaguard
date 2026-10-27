"""N5's record, day by day (research/news_reader/forward_record.py)."""

import importlib.util
import json
import sys
import types
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest
import yaml

from qfin_alphaguard.challengers import ChallengerError, load_card

HERE = __import__("pathlib").Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fr = _load("forward_record")
forward = _load("forward")
harness = _load("harness")

LEVERS = {"target": "scaled", "side": "tenth", "weights": "equal", "sectors": "none"}


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def article(id, created, updated=None, symbols=("AAPL",), headline="h", captured=None):
    row = {
        "id": id,
        "created_at": created,
        "updated_at": updated or created,
        "symbols": list(symbols),
        "headline": headline,
        "summary": "",
    }
    return row | ({"captured_at": captured} if captured else {})


# --- which articles count, for whom


def test_an_article_is_known_from_its_edit_only_if_made_over_five_minutes_later():
    created = "2026-10-26T13:00:00+00:00"
    assert fr.known(article(1, created)) == utc(2026, 10, 26, 13, 0)
    assert fr.known(article(1, created, "2026-10-26T13:05:00+00:00")) == utc(
        2026, 10, 26, 13, 0
    )
    assert fr.known(article(1, created, "2026-10-26T13:05:01+00:00")) == utc(
        2026, 10, 26, 13, 5, 1
    )


def test_members_are_the_day_s_by_the_symbol_they_answer_to():
    prices = pl.DataFrame(
        {
            "date": [date(2026, 10, 26)] * 4 + [date(2026, 10, 27)],
            "ticker": ["AAPL", "FOX", "FOX_old1", "XOM", "MSFT"],
            "sp500": [True, True, True, False, True],
        }
    )
    assert dict(fr.members_on(prices, date(2026, 10, 26))) == {
        "AAPL": ["AAPL"],
        "FOX": ["FOX", "FOX_old1"],
    }


MEMBERS = {"AAPL": ["AAPL"], "BRK-B": ["BRK-B"], "FOX": ["FOX", "FOX_old1"]}


def test_an_article_counts_in_its_window_for_each_member_it_names():
    # Monday 26 October 2026: from Friday 16:00 (20:00 UTC) to Monday 15:30 (19:30 UTC).
    articles = [
        article(1, "2026-10-23T20:00:00+00:00"),  # at the start: the session before's
        article(2, "2026-10-23T20:00:01+00:00", headline="a"),
        article(3, "2026-10-26T19:30:00+00:00", headline="b"),  # at the end: in
        article(4, "2026-10-26T19:30:01+00:00"),  # after the end
        # created before the window, edited inside it: counts from the edit
        article(5, "2026-10-01T12:00:00+00:00", "2026-10-26T12:00:00+00:00", ["BRK.B"]),
        # created inside, edited after the end: counts from the edit, so not today
        article(6, "2026-10-26T12:00:00+00:00", "2026-10-26T19:40:00+00:00"),
        article(7, "2026-10-26T12:00:00+00:00", symbols=()),  # no stock
        article(8, "2026-10-26T12:00:00+00:00", symbols=("AAPL", "BRK/B", "XOM")),
        article(9, "2026-10-26T12:00:00+00:00", symbols=("FOX",)),  # two members answer
        article(10, "2026-10-26T12:00:00+00:00", symbols=("XOM",)),  # not a member
        # two tags naming one member: two pairs, as N2 counts it
        article(11, "2026-10-26T11:00:00+00:00", symbols=("BRK.B", "BRK/B")),
        article(12, "2026-10-26T12:00:00+00:00", symbols=("AAPL", " brk.b")),
    ]
    got = fr.counted(articles, date(2026, 10, 26), MEMBERS)
    assert [(i, t) for i, t, _ in got] == [
        (2, "AAPL"),
        (11, "BRK-B"),
        (11, "BRK-B"),
        (5, "BRK-B"),
        (12, "AAPL"),
        (3, "AAPL"),
    ]
    assert got[0][2] == "a" and got[-1][2] == "b"


def test_member_days_average_the_embeddings_and_log_the_count():
    vectors = {
        1: np.array([1.0, 0.0], np.float16),
        2: np.array([3.0, 2.0], np.float16),
        3: np.array([5.0, 5.0], np.float16),
    }
    pairs = [(1, "MSFT", "x"), (2, "MSFT", "y"), (3, "AAPL", "z"), (3, "AAPL", "z")]
    tickers, counts, x = fr.member_days(pairs, vectors)
    assert tickers == ["AAPL", "MSFT"] and counts == [2, 2]
    np.testing.assert_allclose(x, [[5, 5, np.log(3)], [2, 1, np.log(3)]])


# --- returns and the day's score


def closes(rows):
    return pl.DataFrame(rows, schema=["date", "ticker", "close"], orient="row")


def test_returns_run_from_the_day_s_close_to_the_next_session_s():
    prices = closes(
        [
            (date(2026, 10, 23), "AAPL", 100.0),  # a Friday
            (date(2026, 10, 23), "GONE", 50.0),
            (date(2026, 10, 24), "AAPL", 999.0),  # not a session
            (date(2026, 10, 26), "AAPL", 102.0),
            (date(2026, 10, 26), "NEW", 10.0),
        ]
    )
    assert fr.next_returns(prices, date(2026, 10, 23)) == pytest.approx({"AAPL": 0.02})
    assert fr.next_returns(prices, date(2026, 10, 26)) is None  # Tuesday is not in yet
    assert fr.next_returns(prices, date(2026, 10, 22)) is None  # nor is Thursday


def test_a_day_s_score_is_the_tenths_long_and_short_less_two_bps():
    tickers = [f"T{i:02}" for i in range(25)]
    forecasts = np.arange(25.0)
    returns = {t: 0.001 * i for i, t in enumerate(tickers)}
    row = fr.score_day(tickers, forecasts, returns, LEVERS, harness)
    # 25 // 10 = 2 a side: long T24 and T23, short T00 and T01
    assert row == pytest.approx(
        {
            "members": 25,
            "long": 0.0235,
            "short": 0.0005,
            "everyone": 0.012,
            "score": 0.0235 - 0.0005 - 0.0002,
            "n_long": 2,
            "n_short": 2,
            "why": "",
        }
    )


def test_ties_at_an_edge_are_all_in_and_members_without_a_next_close_are_out():
    tickers = [f"T{i:02}" for i in range(22)]
    forecasts = np.array([0.0, 0.0, 0.0] + [5.0] * 16 + [9.0, 9.0, 9.0])
    returns = {t: 0.01 for t in tickers} | {"T00": -0.02}
    row = fr.score_day(
        tickers + ["LATE"], np.append(forecasts, 99.0), returns, LEVERS, harness
    )
    assert row["members"] == 22
    assert (row["n_long"], row["n_short"]) == (3, 3)
    assert row["short"] == pytest.approx((-0.02 + 0.01 + 0.01) / 3)
    assert row["everyone"] == pytest.approx((21 * 0.01 - 0.02) / 22)  # the mean


def test_a_day_with_under_20_members_or_edges_that_meet_has_no_score():
    tickers = [f"T{i:02}" for i in range(20)]
    returns = {t: 0.01 for t in tickers[1:]}
    row = fr.score_day(tickers, np.arange(20.0), returns, LEVERS, harness)
    assert (
        row["score"] is None and row["why"] == "19 members with news and a next close"
    )
    returns = {t: 0.01 for t in tickers}
    row = fr.score_day(tickers, np.ones(20), returns, LEVERS, harness)
    assert row["score"] is None and row["why"] == "no forecast separates the sides"
    assert fr.score_day(tickers, np.arange(20.0), returns, LEVERS, harness)["score"]


def test_positions_refuse_levers_that_need_volatility_or_sectors():
    for change in ({"weights": "inverse"}, {"sectors": "neutral"}):
        with pytest.raises(ValueError, match="no volatilities or sectors"):
            fr.positions(np.arange(30.0), LEVERS | change, harness)


def test_the_earnings_season_runs_from_the_15th_to_the_end_of_the_month_after():
    inside = [
        date(2027, 1, 15),
        date(2027, 2, 28),
        date(2026, 10, 15),
        date(2026, 11, 30),
    ]
    outside = [
        date(2027, 1, 14),
        date(2027, 3, 1),
        date(2026, 12, 15),
        date(2026, 6, 30),
    ]
    assert all(fr.earnings_season(d) for d in inside)
    assert not any(fr.earnings_season(d) for d in outside)


def test_a_capture_could_run_until_16_30_new_york_time():
    assert not fr.capture_ended(date(2026, 10, 26), utc(2026, 10, 26, 20, 30))  # 16:30
    assert fr.capture_ended(date(2026, 10, 26), utc(2026, 10, 26, 20, 31))
    assert not fr.capture_ended(date(2026, 11, 2), utc(2026, 11, 2, 21, 30))  # EST
    assert fr.capture_ended(date(2026, 11, 2), utc(2026, 11, 2, 21, 31))


def test_a_capture_must_be_taken_on_its_day_between_15_30_and_16_30(tmp_path):
    path = tmp_path / "c.jsonl"
    day = date(2026, 10, 26)

    def write(*moments):
        path.write_text(
            "".join(
                json.dumps(article(i, "2026-10-26T12:00:00Z", captured=m)) + "\n"
                for i, m in enumerate(moments)
            )
        )

    write("2026-10-26T19:31:00Z", "2026-10-26T19:31:00Z")
    articles, moment = fr.read_capture(path, day)
    assert len(articles) == 2 and moment == utc(2026, 10, 26, 19, 31)
    for moments in (
        ("2026-10-26T19:29:00Z",),  # 15:29
        ("2026-10-26T20:31:00Z",),  # 16:31
        ("2026-10-27T19:31:00Z",),  # the next day
        ("2026-10-26T19:31:00Z", "2026-10-26T19:32:00Z"),
    ):
        write(*moments)
        with pytest.raises(SystemExit):
            fr.read_capture(path, day)
    path.write_text("")
    assert fr.read_capture(path, day) == ([], None)


# --- the record, run on a small world

TICKERS = [f"T{i:02}" for i in range(25)]
DAYS = [date(2026, 10, d) for d in (26, 27, 28, 29, 30)]


class Embedder:
    """Headline 'h<k>' reads as (k, 0, 0, 0): the frozen arrays' forecast is k."""

    def __init__(self):
        self.asked = []

    def __call__(self, headlines):
        self.asked.append(list(headlines))
        return np.array([[float(h[1:]), 0, 0, 0] for h in headlines], np.float16)


def write_card(root, **change):
    card = {
        "name": "news-n4-forward",
        "description": "N4's variant 2, frozen.",
        "owner": "Efe",
        "code_commit": "0" * 40,
        "weights": "data/models/news-n4-forward.npz",
        "weights_sha256": forward.sha256(root / "data/models/news-n4-forward.npz"),
        "trained_from": date(2016, 1, 5),
        "trained_to": date(2026, 10, 9),
        "frozen_at": utc(2026, 10, 22, 10, 44),
        "record_from": DAYS[0],
        "scale": 1.0,
        "clip": 0.085,
        "alpha": 0.0125,
        "retired_on": None,
    } | change
    path = root / "challengers" / "news-n4-forward.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(card, sort_keys=False))
    return path


def write_capture(root, day, rows, moment):
    path = root / "data" / "news_capture" / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r | {"captured_at": moment}) + "\n" for r in rows)
    )


def write_prices(root, days):
    """Every ticker a member each day; T<i> returns 0.1% * i from each day to the next."""
    rows = []
    for k, day in enumerate(days):
        for i, t in enumerate(TICKERS):
            rows.append((day, t, 100 * (1 + 0.001 * i) ** k, True))
    path = root / "data" / "news_reader" / "forward" / "prices.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows, schema=["date", "ticker", "close", "sp500"], orient="row"
    ).write_parquet(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "data" / "models" / "news-n4-forward.npz"
    weights.parent.mkdir(parents=True)
    np.savez(
        weights,
        pca_mean=np.zeros(4),
        pca_components=np.array([[1.0, 0, 0, 0]]),
        scale_mean=np.zeros(2),
        scale_scale=np.ones(2),
        ridge_coef=np.array([1.0, 0.0]),
        ridge_intercept=np.array([0.0]),
    )
    write_card(tmp_path)
    (tmp_path / "variants").mkdir()
    (tmp_path / "variants" / "N4_chosen.json").write_text(json.dumps(LEVERS))
    fake = types.SimpleNamespace(
        VARIANTS=tmp_path / "variants",
        N4_LEVERS=harness.N4_LEVERS,
        n4_weights=harness.n4_weights,
    )
    every = [
        article(i, "2026-10-26T13:00:00Z", symbols=[t], headline=f"h{i}")
        for i, t in enumerate(TICKERS)
    ]
    write_capture(tmp_path, DAYS[0], every, "2026-10-26T19:31:00Z")
    # 27 October: no capture
    write_capture(
        tmp_path,
        DAYS[2],
        [
            article(100 + i, "2026-10-28T13:00:00Z", symbols=[t], headline=f"h{i}")
            for i, t in enumerate(TICKERS[:5])
        ],
        "2026-10-28T19:31:00Z",
    )
    write_capture(
        tmp_path,
        DAYS[3],
        [
            article(200 + i, "2026-10-29T13:00:00Z", symbols=[t], headline=f"h{24 - i}")
            for i, t in enumerate(TICKERS)
        ],
        "2026-10-29T19:31:00Z",
    )
    write_prices(tmp_path, DAYS[:4])  # 30 October's close is not in yet
    return tmp_path, fake


def test_the_record_scores_each_day_due_once_and_waits_for_the_next_close(world):
    root, fake = world
    embedder = Embedder()
    days = fr.run(fake, forward, now=utc(2026, 10, 30, 21, 0), embedder=embedder)
    assert days["day"].to_list() == DAYS[:3]  # the 29th waits for the 30th's close
    first, missing, few = days.iter_rows(named=True)
    assert first["score"] == pytest.approx(0.0235 - 0.0005 - 0.0002)
    assert (first["articles"], first["members"], first["why"]) == (25, 25, "")
    assert first["captured_at"] == utc(2026, 10, 26, 19, 31)
    assert first["season"] and first["scored_at"] == utc(2026, 10, 30, 21, 0)
    assert (missing["why"], missing["score"], missing["members"]) == (
        "no capture",
        None,
        None,
    )
    assert few["why"] == "5 members with news and a next close"
    assert [sorted(a) for a in embedder.asked] == [
        sorted([f"h{i}" for i in range(25)] + [f"h{i}" for i in range(5)])
    ]

    # The 30th's close arrives; the scored days stay as they were, though their captures go.
    write_prices(root, DAYS)
    (root / "data" / "news_capture" / f"{DAYS[0]}.jsonl").unlink()
    embedder = Embedder()
    days = fr.run(fake, forward, now=utc(2026, 11, 2, 22, 0), embedder=embedder)
    assert days["day"].to_list() == [*DAYS, date(2026, 11, 2)]
    assert days.row(0, named=True) == first
    later = days.row(3, named=True)  # forecasts reversed: the long side lost
    assert later["score"] == pytest.approx(0.0005 - 0.0235 - 0.0002)
    assert days["why"].to_list()[4:] == ["no capture", "no capture"]
    assert [sorted(a) for a in embedder.asked] == [sorted(f"h{i}" for i in range(25))]
    assert pl.read_parquet(root / fr.OUT / "days.parquet").equals(days)


def test_the_record_lists_nothing_before_a_session_s_capture_could_have_run(world):
    root, fake = world
    for now in (utc(2026, 10, 22, 12, 0), utc(2026, 10, 26, 20, 0)):  # before; 16:00
        days = fr.run(fake, forward, now=now, embedder=Embedder())
        assert days.is_empty() and not (root / fr.OUT / "days.parquet").exists()


def test_the_record_refuses_weights_or_a_test_other_than_the_card_s(world):
    root, fake = world
    now = utc(2026, 10, 30, 21, 0)
    write_card(root, clip=0.09)
    with pytest.raises(SystemExit, match="pre-registration N5"):
        fr.run(fake, forward, now=now, embedder=Embedder())
    write_card(root, weights="data/models/other.npz")
    with pytest.raises(SystemExit, match="names data/models/other.npz"):
        fr.run(fake, forward, now=now, embedder=Embedder())
    write_card(root, weights_sha256="0" * 64)
    with pytest.raises(ChallengerError, match="not the file that was frozen"):
        fr.run(fake, forward, now=now, embedder=Embedder())
    assert not (root / fr.OUT / "days.parquet").exists()


# --- the evidence and the page


def test_the_evidence_bets_on_the_scored_days_in_order():
    days = pl.DataFrame(
        {
            "day": [date(2026, 10, 28), date(2026, 10, 26), date(2026, 10, 27)],
            "score": [0.2, 0.01, None],
        }
    )
    result = fr.evidence(days)
    assert result.outcomes == pytest.approx((0.01 / 0.085, 1.0))
    assert result.clipped == (1,)
    assert result.threshold == pytest.approx(80)


def test_the_page_shows_the_test_and_every_day_and_no_headline(world):
    root, fake = world
    now = utc(2026, 11, 2, 22, 0)
    write_prices(root, DAYS)
    days = fr.run(fake, forward, now=now, embedder=Embedder())
    card = load_card(root / fr.CARD)
    page = fr.record_page(card, days, fr.evidence(days), now)
    assert "N5: N4&#x27;s variant 2" in page or "N5: N4's variant 2" in page
    assert "against 80: not promoted" in page
    assert "6 sessions, 2 scored, 1 without a trade, 3 without a capture" in page
    for day in days["day"]:
        assert f"<td>{day}</td>" in page
    assert "5 members with news and a next close" in page
    assert ">+228.00</td>" in page and ">−232.00</td>" in page  # the two days scored
    assert ">h1<" not in page and '"h1' not in page


def test_the_page_says_when_the_record_is_promoted():
    sessions = [date(2026, 10, 26 + k) for k in range(5)] + [
        date(2026, 11, 2 + k) for k in range(5)
    ]
    n = len(sessions)
    days = pl.DataFrame(
        {
            "day": sessions,
            "score": [0.08] * n,
            "season": [True] * n,
            "long": [0.05] * n,
            "short": [-0.03] * n,
            "everyone": [0.01] * n,
            "members": [30] * n,
            "n_long": [3] * n,
            "n_short": [3] * n,
            "why": [""] * n,
        }
    )
    card = types.SimpleNamespace(
        record_from=sessions[0],
        frozen_at=utc(2026, 10, 22, 10, 44),
        weights_sha256="e" * 64,
        trained_from=date(2016, 1, 5),
        trained_to=date(2026, 10, 9),
    )
    result = fr.BettingTest(scale=1.0, clip=0.085, alpha=0.5).run([0.08] * n, [0.0] * n)
    assert result.promoted
    page = fr.record_page(card, days, result, utc(2026, 11, 6, 22, 0))
    assert f"promoted on {sessions[result.promoted_on]}" in page
