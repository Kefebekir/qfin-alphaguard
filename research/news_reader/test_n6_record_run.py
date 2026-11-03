"""N6's record run on a small world (n6_record.run). They need PyTorch:
data/llmenv/bin/python -m pytest research/news_reader/test_n6_record_run.py
"""

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
import n6_freeze
import n6_model as nm
import n6_record as n6r

TICKERS = [f"T{i:02}" for i in range(25)]
DAYS = [date(2026, 10, d) for d in (26, 27, 28, 29, 30)]
SETTINGS = {
    "dim": 8,
    "attention": 4,
    "context": 8,
    "hidden": 8,
    "dropout": 0.0,
    "loss": "mse",
    "aux": 0.1,
    "lr": 0.001,
    "decay": 0.0,
    "epochs": 1,
    "days_per_batch": 8,
    "start": "random",
    "inputs": "news",
}


def utc(*args):
    return datetime(*args, tzinfo=UTC)


class Embedder:
    """Each headline a fixed random vector of PIT-4B's width."""

    def __init__(self):
        self.asked = []

    def __call__(self, headlines):
        self.asked.append(list(headlines))
        return np.array(
            [
                np.random.default_rng(int(h[1:])).normal(size=nm.WIDTH)
                for h in headlines
            ],
            dtype=np.float16,
        )


def article(id, created, ticker, headline):
    return {
        "id": id,
        "created_at": created,
        "updated_at": created,
        "symbols": [ticker],
        "headline": headline,
        "summary": "",
    }


def write_capture(root, day, rows, moment):
    path = root / "data" / "news_capture" / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r | {"captured_at": moment}) + "\n" for r in rows)
    )


def write_prices(root, days):
    rows = [
        (day, t, 100 * (1 + 0.001 * i) ** k, True)
        for k, day in enumerate(days)
        for i, t in enumerate(TICKERS)
    ]
    path = root / "data" / "news_reader" / "forward" / "prices.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows, schema=["date", "ticker", "close", "sp500"], orient="row"
    ).write_parquet(path)


def write_card(root, **change):
    card = {
        "name": "news-n6-forward",
        "description": "N6's network, frozen.",
        "owner": "Efe",
        "code_commit": "0" * 40,
        "weights": "data/models/news-n6-forward.npz",
        "weights_sha256": n6_freeze.sha256(root / "data/models/news-n6-forward.npz"),
        "trained_from": date(2016, 1, 5),
        "trained_to": date(2026, 10, 9),
        "frozen_at": utc(2026, 10, 22, 10, 0),
        "record_from": DAYS[0],
        "scale": 1.0,
        "clip": 0.092,
        "alpha": 0.00625,
        "retired_on": None,
    } | change
    path = root / "challengers" / "news-n6-forward.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(card, sort_keys=False))


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    np.savez(
        models / "n6-eyes.npz",
        mean_202412=np.zeros(nm.WIDTH, np.float32),
        spread_202412=np.ones(nm.WIDTH, np.float32),
    )
    torch.manual_seed(0)
    networks = [nm.N6(nm.Settings(**SETTINGS), 18, 12, 3) for _ in range(2)]
    np.savez(
        models / "news-n6-forward.npz",
        **n6_freeze.state(networks),
        settings=np.array(json.dumps(SETTINGS, sort_keys=True)),
        seeds=np.array([0, 1]),
        trained_from=np.array("2016-01-05"),
        trained_to=np.array("2026-10-09"),
        eyes_sha256=np.array(n6_freeze.sha256(models / "n6-eyes.npz")),
    )
    write_card(tmp_path)
    write_capture(
        tmp_path,
        DAYS[0],
        [article(i, "2026-10-26T13:00:00Z", t, f"h{i}") for i, t in enumerate(TICKERS)],
        "2026-10-26T19:31:00Z",
    )
    # 27 October: no capture
    write_capture(
        tmp_path,
        DAYS[2],
        [
            article(100 + i, "2026-10-28T13:00:00Z", t, f"h{100 + i}")
            for i, t in enumerate(TICKERS)
        ],
        "2026-10-28T19:31:00Z",
    )
    write_prices(tmp_path, DAYS[:3])  # the 29th's close is not in yet
    return tmp_path


def test_the_record_scores_each_day_due_once_and_waits_for_the_next_close(world):
    embedder = Embedder()
    days = n6r.run(now=utc(2026, 10, 30, 21, 0), embedder=embedder)
    assert days["day"].to_list() == DAYS[:2]  # the 28th waits for the 29th's close
    first, missing = days.iter_rows(named=True)
    assert first["members"] == 25 and first["n_long"] == 2 and first["n_short"] == 2
    assert np.isfinite(first["score"]) and first["why"] == ""
    assert (missing["why"], missing["score"]) == ("no capture", None)
    assert [sorted(a) for a in embedder.asked] == [sorted(f"h{i}" for i in range(25))]

    write_prices(world, DAYS)
    (world / "data" / "news_capture" / f"{DAYS[0]}.jsonl").unlink()
    embedder = Embedder()
    again = n6r.run(now=utc(2026, 11, 2, 22, 0), embedder=embedder)
    assert again.row(0, named=True) == first  # scored once, kept as it was
    assert again["day"].to_list()[:3] == DAYS[:3]
    assert np.isfinite(again.row(2, named=True)["score"])
    assert [sorted(a) for a in embedder.asked] == [
        sorted(f"h{100 + i}" for i in range(25))
    ]
    assert pl.read_parquet(world / n6r.OUT / "days.parquet").equals(again)


def test_the_record_refuses_weights_eyes_or_a_test_other_than_the_card_s(world):
    now = utc(2026, 10, 30, 21, 0)
    write_card(world, clip=0.085)
    with pytest.raises(SystemExit, match="pre-registration N6"):
        n6r.run(now=now, embedder=Embedder())
    write_card(world, weights_sha256="0" * 64)
    with pytest.raises(Exception, match="not the file that was frozen"):
        n6r.run(now=now, embedder=Embedder())
    write_card(world)
    np.savez(
        world / "data" / "models" / "n6-eyes.npz",
        mean_202412=np.ones(nm.WIDTH, np.float32),
        spread_202412=np.ones(nm.WIDTH, np.float32),
    )
    with pytest.raises(SystemExit, match="not the eyes N6 was frozen with"):
        n6r.run(now=now, embedder=Embedder())
    assert not (world / n6r.OUT / "days.parquet").exists()
