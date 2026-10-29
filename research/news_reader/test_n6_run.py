"""Efe's N6 runs (n6_run.py). They need PyTorch:
data/llmenv/bin/python -m pytest research/news_reader/test_n6_run.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
import n6_harness as nh
import n6_model as nm
import n6_run as nr
from test_n6_model import world


def test_settings_come_as_name_value_pairs_of_their_own_types():
    name, s = nr.parse(["v1", "dim=16", "lr=0.003", "loss=sharpe", "start=random"])
    assert name == "v1"
    assert (s.dim, s.lr, s.loss, s.start) == (16, 0.003, "sharpe", "random")
    assert nr.parse(["v2"])[1] == nm.Settings()
    assert nr.parse(["v3", "inputs=news"])[1].inputs == "news"
    with pytest.raises(SystemExit, match="unknown setting"):
        nr.parse(["v1", "depth=3"])
    with pytest.raises(SystemExit, match="usage"):
        nr.parse(["dim=16"])
    with pytest.raises(ValueError, match="loss is one of"):
        nr.parse(["v1", "loss=mae"])


def test_n4_s_development_years_are_read_from_its_stored_run(tmp_path, monkeypatch):
    stored = {
        "count_only": {},
        "variants": [
            {"n": 1, "result": {}},
            {
                "n": 2,
                "result": repr(
                    {
                        "mean_bps": 8.7,
                        "t": 2.07,
                        "days": 998,
                        "years": repr(
                            {"2018": {"days": 246, "mean_bps": 3.97, "t": 0.58}}
                        ),
                    }
                ),
            },
        ],
    }
    path = tmp_path / "N4.json"
    path.write_text(json.dumps(stored))
    monkeypatch.setattr(nr, "N4_DEV", path)
    assert nr.n4_years() == {2018: (3.97, 0.58, 246), "all": (8.7, 2.07, 998)}
    # as dev_report stored variant 2 itself: its number an int, its result a dict
    stored["variants"][1]["result"] = {
        "mean_bps": 8.7,
        "t": 2.07,
        "days": 998,
        "years": {"2018": {"days": 246, "mean_bps": 3.97, "t": 0.58}},
    }
    path.write_text(json.dumps(stored))
    assert nr.n4_years() == {2018: (3.97, 0.58, 246), "all": (8.7, 2.07, 998)}
    monkeypatch.setattr(nr, "N4_DEV", tmp_path / "none.json")
    assert nr.n4_years() == {}


def test_a_run_scores_the_variant_beside_its_placebos(monkeypatch):
    a, store = world(days=300, signal=1.0, first=date(2017, 6, 1))
    monkeypatch.setattr(nh, "DEVELOPMENT", (2018,))
    monkeypatch.setattr(
        nr, "n4_years", lambda: {2018: (3.0, 0.5, 90), "all": (3.0, 0.5, 90)}
    )
    s = nm.Settings(
        dim=8, attention=4, context=8, hidden=16, dropout=0.0, epochs=6, lr=3e-3
    )
    result = nr.run("t", s, a, store, seeds=(0, 1))
    _, validated = nh.fold(a, 2018)
    assert result["years"]["2018"]["model"]["days"] == len(np.unique(a.day[validated]))
    model, dealt, silent = (result["all"][k]["bps"] for k in nr.LABELS)
    assert model > max(dealt, silent) and result["gate"]
    assert result["seeds"] == [0, 1] and result["settings"]["dim"] == 8
    page = nr.page(result)
    assert "passes the placebo gate" in page and "N4&#x27;s variant 2" not in page
    assert "N4's variant 2</td>" in page and page.count("<tr>") == 5
    json.dumps(result)  # it is stored as JSON
    again = nr.run("t", s, a, store, seeds=(0,), years_run=(2018,), keep_days=True)
    assert len(again["days"]) == again["all"]["model"]["days"]
    assert all(isinstance(d["day"], str) for d in again["days"])
    json.dumps(again)


def fake(name):
    one = {"bps": 4.0, "t": 1.0, "days": 998}
    return {
        "name": name,
        "settings": {"dim": 16},
        "seeds": [0],
        "run_utc": "2026-10-23 10:00",
        "years": {"2018": {"model": one, "dealt": one, "silent": one}},
        "all": {
            "model": one,
            "dealt": {**one, "bps": 1.0},
            "silent": {**one, "bps": 0.5},
        },
        "gate": True,
        "n4": {},
    }


def test_a_variant_is_in_the_ledger_before_it_runs_and_stays_if_it_fails(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text("# Ledger\n")
    monkeypatch.setattr(nr.n6_ledger, "LEDGER", ledger)
    monkeypatch.setattr(nr.n6_ledger, "CHOSEN", tmp_path / "chosen.json")
    monkeypatch.setattr(nr, "OUT", tmp_path / "dev")
    seen = []

    def failing(name, s):
        seen.append(nr.n6_ledger.rows(ledger.read_text()))
        raise RuntimeError("out of memory")

    monkeypatch.setattr(nr, "run", failing)
    with pytest.raises(RuntimeError):
        nr.main(["v1", "dim=16"])
    assert seen[0][0]["result"] == "running"
    assert nr.n6_ledger.rows(ledger.read_text())[0]["result"] == "did not finish"

    monkeypatch.setattr(nr, "run", lambda name, s: fake(name))
    nr.main(["v1", "dim=16"])  # the same variant again, now finished
    row = nr.n6_ledger.rows(ledger.read_text())[0]
    assert row["n"] == 1 and row["result"].startswith("+4.00 bps")
    assert (tmp_path / "dev" / "v1.json").exists() and (
        tmp_path / "dev" / "v1.html"
    ).exists()
    nr.main(["choose"])
    chosen = json.loads((tmp_path / "chosen.json").read_text())
    assert (chosen["variant"], chosen["name"]) == (1, "v1")
    with pytest.raises(SystemExit, match="development is over"):
        nr.main(["v2", "dim=8"])


def test_a_news_alone_variant_s_page_names_its_second_placebo():
    result = fake("v3")
    result["settings"]["inputs"] = "news"
    assert "nothing to read" in nr.page(result)
    assert "the context alone" not in nr.page(result)
    assert "the context alone" in nr.page(fake("v1"))


def test_forecasts_that_are_not_finite_stop_the_run_and_it_does_not_finish(
    tmp_path, monkeypatch
):
    a, store = world(days=300, first=date(2017, 6, 1))
    monkeypatch.setattr(nh, "DEVELOPMENT", (2018,))
    monkeypatch.setattr(nr, "n4_years", dict)
    monkeypatch.setattr(
        nm,
        "ensemble",
        lambda arrays, trained, validated, *rest: np.full(validated.sum(), np.nan),
    )
    with pytest.raises(
        FloatingPointError, match="2018, model: forecasts that are not finite"
    ):
        nr.run("t", nm.Settings(dim=8), a, store, seeds=(0,))
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text("# Ledger\n")
    monkeypatch.setattr(nr.n6_ledger, "LEDGER", ledger)
    monkeypatch.setattr(nr.n6_ledger, "CHOSEN", tmp_path / "chosen.json")
    monkeypatch.setattr(nr, "OUT", tmp_path / "dev")
    monkeypatch.setattr(nh, "load", lambda: a)
    monkeypatch.setattr(nr, "gpu_store", lambda: store)
    with pytest.raises(FloatingPointError):
        nr.main(["v7", "dim=8"])
    assert nr.n6_ledger.rows(ledger.read_text())[0]["result"] == "did not finish"
    assert not (tmp_path / "dev" / "v7.json").exists()
