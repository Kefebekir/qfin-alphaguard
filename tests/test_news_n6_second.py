"""N6's second look (research/news_reader/n6_second.py): its report, lock and checks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).parents[1] / "research" / "news_reader"
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("n6_second", HERE / "n6_second.py")
ns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ns)


def test_the_clip_is_the_largest_absolute_score_rounded_up_to_a_tenth_of_a_percent():
    assert ns.clip_of([0.01, -0.08481, 0.02]) == 0.085
    assert ns.clip_of([0.085, -0.01]) == 0.085  # already on the grid
    assert ns.clip_of([0.00001]) == 0.001


def days_of(scores, start="2022-01-03"):
    g = np.random.default_rng(0)
    out = []
    when = pl.date_range(pl.date(2022, 1, 3), pl.date(2023, 12, 29), "1d", eager=True)
    for k, s in enumerate(scores):
        everyone = float(g.normal(0, 0.01))
        long = everyone + s / 2 + 0.0001
        short = everyone - s / 2 + 0.0001
        out.append(
            {
                "day": str(when[k * 3]),
                "score": s,
                "long": long,
                "short": short,
                "everyone": everyone,
            }
        )
    return out


def test_the_report_has_what_the_pre_registration_asks():
    scores = list(np.random.default_rng(1).normal(0.0005, 0.01, 200))
    days = days_of(scores)
    n4 = pl.DataFrame(
        {
            "day": [pl.Series([d["day"] for d in days[:150]]).str.to_date()][0],
            "score": 0.0002,
        }
    )
    r = ns.report(days, n4)
    assert r["after"]["bps"] == pytest.approx(np.mean(scores) * 1e4)
    assert r["after"]["days"] == 200
    long = np.array([d["long"] for d in days])
    short = np.array([d["short"] for d in days])
    everyone = np.array([d["everyone"] for d in days])
    assert r["before"]["bps"] == pytest.approx((long - short).mean() * 1e4)
    assert r["long"]["bps"] == pytest.approx((long - everyone).mean() * 1e4)
    assert r["short"]["bps"] == pytest.approx((everyone - short).mean() * 1e4)
    assert r["less_n4"]["days"] == 150
    assert r["less_n4"]["bps"] == pytest.approx((np.mean(scores[:150]) - 0.0002) * 1e4)
    assert list(r["years"]) == ["2022", "2023"]
    assert sum(v["days"] for v in r["years"].values()) == 200
    assert r["clip"] == ns.clip_of(scores) and r["passes"] == (np.mean(scores) > 0)


def lock_with(scores) -> dict:
    days = days_of(scores)
    n4 = pl.DataFrame(
        {"day": pl.Series([d["day"] for d in days]).str.to_date(), "score": 0.0}
    )
    one = {"bps": 1.0, "t": 0.5, "days": 10}
    return {
        "chosen": {"variant": 3, "name": "v3"},
        "commit": "0123456789abcdef",
        "run_utc": "2026-10-29 10:00",
        "result": {"all": {"model": one, "dealt": one, "silent": one}},
        "days": days,
        "report": ns.report(days, n4),
    }


def test_the_ledger_section_and_the_page_say_whether_a_card_may_be_made():
    good = lock_with([0.001] * 20 + [0.002] * 20)
    text = ns.section(good)
    assert "a card may be made" in text and "commit 0123456" in text
    assert "Variant 3, `v3`" in text and "0.002" in text  # the clip
    assert "a card may be made" in ns.page(good)
    bad = lock_with([-0.001] * 40)
    assert "N6 fails here" in ns.section(bad) and "no card" in ns.page(bad)


def test_the_choice_must_be_committed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = lambda *a: subprocess.run(a, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    chosen = Path("chosen.json")
    chosen.write_text("{}")
    with pytest.raises(SystemExit, match="commit"):
        ns.committed(chosen)  # not tracked
    run("git", "add", "chosen.json")
    run("git", "commit", "-q", "-m", "choice")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout
    assert ns.committed(chosen) == head.strip()
    chosen.write_text('{"changed": 1}')
    with pytest.raises(SystemExit, match="commit"):
        ns.committed(chosen)


def test_a_locked_second_look_runs_nothing_again(tmp_path, monkeypatch, capsys):
    lock = lock_with([0.001] * 30)
    monkeypatch.setattr(ns, "N6", tmp_path)
    monkeypatch.setattr(ns, "LOCK", tmp_path / "second_look.json")
    (tmp_path / "second_look.json").write_text(json.dumps(lock))
    ns.main()
    assert "nothing runs again" in capsys.readouterr().out
    assert "a card may be made" in (tmp_path / "second_look.html").read_text()
