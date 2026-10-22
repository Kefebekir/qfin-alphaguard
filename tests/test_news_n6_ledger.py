"""N6's variants in the ledger and the choice (research/news_reader/n6_ledger.py)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "n6_ledger",
    Path(__file__).parents[1] / "research" / "news_reader" / "n6_ledger.py",
)
nl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nl)

LEDGER = "# Ledger\n\n## Something before\n\ntext\n"


def result(name, model, dealt, silent, settings=None):
    def s(bps):
        return {"bps": bps, "t": bps / 4, "days": 998}

    return {
        "name": name,
        "settings": settings or {"dim": 32, "loss": "mse"},
        "seeds": [0, 1, 2, 3, 4],
        "run_utc": "2026-10-23 10:00",
        "all": {"model": s(model), "dealt": s(dealt), "silent": s(silent)},
        "gate": model > max(dealt, silent),
    }


def test_a_variant_is_written_before_it_runs_and_its_result_after(tmp_path):
    ledger, n = nl.start(LEDGER, "v1", {"dim": 32}, "2026-10-23 09:40", tmp_path / "c")
    assert n == 1 and ledger.startswith(LEDGER.rstrip("\n"))
    assert nl.rows(ledger) == [
        {"n": 1, "name": "v1", "settings": "dim=32", "result": nl.RUNNING}
    ]
    done = nl.finish(ledger, 1, result("v1", 9.5, 1.2, -3.0, {"dim": 32}))
    line = done.split("\n")[-2] if done.endswith("\n") else done.split("\n")[-1]
    assert line == (
        "| 1 | `v1` | 2026-10-23 10:00 | dim=32 | +9.50 bps (t +2.38) | "
        "+1.20 bps (t +0.30) | −3.00 bps (t −0.75) | passed |"
    )
    ledger, n = nl.start(done, "v2", {"dim": 16}, "2026-10-23 11:00", tmp_path / "c")
    assert n == 2 and [r["name"] for r in nl.rows(ledger)] == ["v1", "v2"]
    assert ledger.count(nl.HEADING) == 1


def test_names_and_settings_run_once_and_ten_at_most(tmp_path):
    chosen = tmp_path / "chosen.json"
    ledger = LEDGER
    for k in range(nl.LIMIT):
        ledger, _ = nl.start(ledger, f"v{k}", {"dim": k + 1}, "t", chosen)
    with pytest.raises(SystemExit, match="a name runs once"):
        nl.start(ledger, "v3", {"dim": 99}, "t", chosen)
    with pytest.raises(SystemExit, match="variant 4, v3, already"):
        nl.start(ledger, "other", {"dim": 4}, "t", chosen)
    with pytest.raises(SystemExit, match="its 10 variants"):
        nl.start(ledger, "v10", {"dim": 50}, "t", chosen)
    chosen.write_text("{}")
    with pytest.raises(SystemExit, match="development is over"):
        nl.start(LEDGER, "v1", {"dim": 1}, "t", chosen)


def test_a_run_that_did_not_finish_counts_and_may_run_again_as_itself(tmp_path):
    ledger, n = nl.start(LEDGER, "v1", {"dim": 32}, "t", tmp_path / "c")
    ledger = nl.unfinished(ledger, n)
    assert nl.rows(ledger)[0]["result"] == nl.UNFINISHED
    with pytest.raises(SystemExit, match="a name runs once"):
        nl.start(ledger, "v1", {"dim": 16}, "t", tmp_path / "c")
    again, m = nl.start(ledger, "v1", {"dim": 32}, "t2", tmp_path / "c")
    assert m == 1 and nl.rows(again)[0]["result"] == nl.RUNNING
    _, k = nl.start(ledger, "v2", {"dim": 16}, "t", tmp_path / "c")
    assert k == 2  # the unfinished one counts


def test_the_choice_is_the_best_of_those_above_both_placebos(tmp_path):
    ledger, results = LEDGER, {}
    runs = [
        ("a", 12.0, 13.0, 0.0),  # the highest, but below its dealt placebo
        ("b", 8.0, 2.0, 1.0),
        ("c", 9.0, 3.0, 8.5),
        ("d", 5.0, 0.0, 0.0),
    ]
    for k, (name, *numbers) in enumerate(runs):
        ledger, n = nl.start(ledger, name, {"dim": k + 1}, "t", tmp_path / "c")
        results[name] = result(name, *numbers, {"dim": k + 1})
        ledger = nl.finish(ledger, n, results[name])
    ledger, _ = nl.start(
        ledger, "e", {"dim": 9}, "t", tmp_path / "c"
    )  # running: not chosen
    chosen = nl.choose(ledger, results)
    assert (chosen["variant"], chosen["name"]) == (3, "c")
    assert chosen["settings"] == {"dim": 3} and chosen["development"]["bps"] == 9.0
    assert "above both their placebos" in chosen["rule"]
    with pytest.raises(SystemExit, match="N6 fails at development"):
        nl.choose(
            nl.finish(
                nl.start(LEDGER, "x", {}, "t", tmp_path / "c")[0],
                1,
                result("x", 1, 2, 0, {}),
            ),
            {"x": result("x", 1, 2, 0, {})},
        )
