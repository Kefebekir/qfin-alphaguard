"""Efe's development runs (research/news_reader/dev_report.py): counting, the ledger, the page."""

import importlib.util
import json
import types
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "dev_report",
    Path(__file__).parents[1] / "research" / "news_reader" / "dev_report.py",
)
dev_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dev_report)

LEDGER = """# Ledger

| # | Test | Run (UTC) | Settings (`efe_model.make`) | Mean a day | t | 2018 / 2019 / 2020 / 2021, bps a day | Note |
|---|---|---|---|---|---|---|---|
| 1 | N1 | 2026-10-17 | components 16, penalty 10.0 | −0.18 bps | −0.22 | +1.74 / −1.13 / +0.09 / −1.42 | x |
| 2 | N1 | 2026-10-17 | components 4, penalty 1000.0 | −0.25 bps | −0.31 | +3.45 / +1.57 / −3.60 / −2.38 | y |

## N1's holdout
"""


RULE = "ties at the edge included (amendment 5)"


def result(mean, t, years, scoring=RULE):
    return {
        "scoring": scoring,
        "mean_bps": mean,
        "t": t,
        "days": 998,
        "years": {
            str(y): {"mean_bps": v, "days": 250, "t": 0.0}
            for y, v in zip((2018, 2019, 2020, 2021), years, strict=True)
        },
    }


BAR = result(-7.06, -1.76, [-12.43, -2.65, -10.73, -2.53])


def test_numbers_carry_a_true_minus_sign_as_the_ledger_does():
    assert dev_report.signed(-1.234) == "−1.23"
    assert dev_report.signed(0.5) == "+0.50"


def test_a_row_names_the_settings_years_and_the_gap_to_counting_alone():
    row = dev_report.ledger_row(
        1,
        "N3",
        "2026-10-19 10:30",
        {"components": 16, "penalty": 10.0, "features": "base-last"},
        result(-3.0, -0.8, [1.0, -2.0, -6.0, -5.0]),
        BAR,
    )
    assert row == (
        "| 1 | N3 | 2026-10-19 10:30 | components 16, penalty 10.0, features base-last | "
        "−3.00 bps | −0.80 | +1.00 / −2.00 / −6.00 / −5.00 | "
        "+4.06 bps a day above counting alone (−7.06) |"
    )


def test_the_row_goes_after_the_last_variant_and_before_the_next_section():
    got = dev_report.add_row(LEDGER, "| 1 | N2 | new |")
    lines = got.split("\n")
    assert (
        lines[5]
        == "| 2 | N1 | 2026-10-17 | components 4, penalty 1000.0 | −0.25 bps | −0.31 | +3.45 / +1.57 / −3.60 / −2.38 | y |"
    )
    assert lines[6] == "| 1 | N2 | new |"
    assert lines[7] == "" and lines[8] == "## N1's holdout"


def test_variants_are_numbered_and_limited_within_their_test():
    store = {"count_only": BAR, "variants": []}
    settings = {"components": 16, "penalty": 10.0}
    assert dev_report.check_new(store, LEDGER, "N2", settings, RULE) == (1, False)
    full = LEDGER
    for i in range(10):
        full = dev_report.add_row(full, f"| {i + 1} | N2 | r | s | m | t | y | n |")
    assert dev_report.variants_in_ledger(full, "N2") == 10
    assert dev_report.variants_in_ledger(full, "N1") == 2
    with pytest.raises(SystemExit, match="its 10 variants"):
        dev_report.check_new(
            store, full, "N2", {"components": 2, "penalty": 10.0}, RULE
        )


def test_settings_already_tried_are_not_run_or_counted_again():
    settings = {"components": 16, "penalty": 10.0}
    store = {
        "count_only": BAR,
        "variants": [{"n": 1, "settings": settings, "result": BAR}],
    }
    with pytest.raises(SystemExit, match="variant 1 already"):
        dev_report.check_new(store, LEDGER, "N2", dict(settings), RULE)


def test_a_variant_scored_under_an_earlier_rule_is_scored_again_as_itself():
    settings = {"components": 16, "penalty": 10.0}
    old = result(-6.78, -1.61, [-4.25, -8.64, -7.56, -6.64], scoring=None)
    store = {
        "count_only": BAR,
        "variants": [{"n": 1, "settings": settings, "result": old}],
    }
    assert dev_report.check_new(store, LEDGER, "N2", dict(settings), RULE) == (1, True)
    ledger = dev_report.add_row(LEDGER, "| 1 | N2 | old row |")
    new = result(-3.0, -0.8, [1.0, -2.0, -6.0, -5.0])
    row = dev_report.ledger_row(1, "N2", "2026-10-19 11:00", settings, new, BAR, old)
    assert row.endswith(
        "; scored again, ties at the edge included (amendment 5): it was "
        "\u22126.78 bps (t \u22121.61) |"
    )
    got = dev_report.replace_row(ledger, "N2", 1, row)
    assert "old row" not in got and row in got
    assert dev_report.variants_in_ledger(got, "N2") == 1  # still one variant
    with pytest.raises(ValueError, match="variant 2"):
        dev_report.replace_row(ledger, "N2", 2, row)


def test_the_page_shows_each_variant_counting_alone_and_each_year():
    store = {
        "count_only": BAR,
        "variants": [
            {
                "n": 1,
                "settings": {"components": 16, "penalty": 10.0},
                "result": result(-3.0, -0.8, [1.0, -2.0, -6.0, -5.0]),
            },
            {
                "n": 2,
                "settings": {"components": 4, "penalty": 10.0},
                "result": result(2.5, 0.6, [4.0, 1.0, 3.0, 2.0]),
            },
        ],
    }
    page = dev_report.page("N2", store)
    assert "Variants tried: <b>2 of 10</b>" in page
    assert "counting alone −7.06" in page  # the dashed reference line's label
    assert "Highest so far: variant 2" in page
    assert page.count('<g class="bar">') == 2
    for year in ("2018", "2019", "2020", "2021"):
        assert f"<th class='num'>{year}</th>" in page
    # The strongest cell is at 70% of its pole; a gain is blue, a loss red.
    assert "var(--neg) 70%" in page and "var(--pos)" in page


def test_a_fresh_result_and_a_stored_one_go_on_one_page():
    # harness.run's years are numbers; the store's, read back from JSON, are strings.
    fresh = result(2.0, 0.5, [1.0, 2.0, 3.0, 4.0])
    fresh["years"] = {int(y): v for y, v in fresh["years"].items()}
    store = {
        "count_only": BAR,
        "variants": [
            {
                "n": 1,
                "settings": {"components": 64, "penalty": 10.0},
                "result": dev_report.stored(fresh),
            },
        ],
    }
    assert list(store["variants"][0]["result"]["years"]) == [
        "2018",
        "2019",
        "2020",
        "2021",
    ]
    assert "+2.00" in dev_report.page("N2", store)


def test_settings_already_tried_say_so_as_their_own_kind_of_stop():
    settings = {"components": 16, "penalty": 10.0}
    store = {
        "count_only": BAR,
        "variants": [{"n": 1, "settings": settings, "result": BAR}],
    }
    with pytest.raises(dev_report.AlreadyTried, match="written again"):
        dev_report.check_new(store, LEDGER, "N2", dict(settings), RULE)


def test_a_label_is_escaped_in_the_page():
    store = {
        "count_only": BAR,
        "variants": [
            {
                "n": 1,
                "settings": {"components": 4, "penalty": 10.0, "features": "<b>"},
                "result": result(1.0, 0.1, [1.0, 1.0, 1.0, 1.0]),
            },
        ],
    }
    page = dev_report.page("N3", store)
    assert "features <b>" not in page and "features &lt;b&gt;" in page


class FakeHarness:
    """harness.run as it answers: the years as numbers, not yet through JSON."""

    SCORING = RULE
    N3_FEATURES = ("base-last", "base-mean")

    def __init__(self):
        self.runs = []

    def run(self, test, make, count_only=False, features=None):
        self.runs.append("count" if count_only else make)
        mean = -1.22 if count_only else float(make)
        return {
            "scoring": RULE,
            "mean_bps": mean,
            "t": mean / 4,
            "days": 998,
            "years": {
                y: {"mean_bps": mean, "days": 250} for y in (2018, 2019, 2020, 2021)
            },
        }


def test_runs_on_their_own_and_after_a_stored_comparison_both_write_the_page(
    tmp_path, monkeypatch
):
    import sys
    import types

    fake = FakeHarness()
    model = types.SimpleNamespace(make=lambda components, penalty: components / 10)
    monkeypatch.setitem(sys.modules, "harness", fake)
    monkeypatch.setitem(sys.modules, "efe_model", model)
    monkeypatch.setattr(dev_report, "DEV", tmp_path)
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(LEDGER)
    monkeypatch.setattr(dev_report, "LEDGER", ledger)

    dev_report.main(["N2", "16", "10.0"])  # the comparison and variant 1, both fresh
    dev_report.main(["N2", "64", "10.0"])  # the comparison read back from JSON
    page = (tmp_path / "N2.html").read_text()
    assert "Variants tried: <b>2 of 10</b>" in page and "+6.40" in page
    assert fake.runs == ["count", 1.6, 6.4]  # the comparison once
    assert dev_report.variants_in_ledger(ledger.read_text(), "N2") == 2

    (tmp_path / "N2.html").unlink()
    with pytest.raises(dev_report.AlreadyTried):
        dev_report.main(["N2", "64", "10.0"])
    assert (tmp_path / "N2.html").exists()  # written again, nothing run
    assert fake.runs == ["count", 1.6, 6.4]


def test_a_long_label_gets_room_on_the_left():
    def svg_for(features):
        settings = {"components": 64, "penalty": 10.0, "features": features}
        label = f"1 · {dev_report.settings_text(settings)}"
        return label, dev_report._bars([(label, result(6.3, 1.4, [7, 12, 5, 1]))], BAR)

    label, svg = svg_for("base-last")
    x = float(svg.split('class="label"')[0].rsplit('<text x="', 1)[1].split('"')[0])
    assert x >= 6.6 * len(
        label
    )  # the label, right-aligned at x, starts inside the chart


def holdout_result(passes=False, t=1.8, years_as=str):
    def years(values):
        return {
            years_as(y): {"mean_bps": v, "days": 240, "t": 0.0}
            for y, v in zip(range(2022, 2027), values, strict=True)
        }

    return {
        "test": "N2",
        "split": "holdout",
        "scoring": RULE,
        "passes": passes,
        "t": t,
        "mean_bps": 3.1,
        "days": 1197,
        "years": years([1.0, 2.0, 3.0, 4.0, 5.5]),
        "count_only": {
            "mean_bps": -1.5,
            "t": -0.6,
            "days": 1197,
            "years": years([-1.0, -2.0, -1.0, -2.0, -1.5]),
        },
        "commit": "abcdef1234567",
        "variant": {"components": 64, "penalty": 10.0, "variant": 2, "rule": "x"},
    }


def test_the_holdout_page_gives_the_verdict_against_the_threshold():
    page = dev_report.holdout_page("N2", holdout_result() | {"threshold": 2.5})
    assert "&#10007; Failed:</b> t +1.80 against the pre-registered 2.50" in page
    assert "variant 2 · components 64, penalty 10.0" in page and "abcdef1" in page
    for year in range(2022, 2027):
        assert f"<th class='num'>{year}</th>" in page
    passed = dev_report.holdout_page(
        "N2", holdout_result(True, 2.7) | {"threshold": 2.5}
    )
    assert "&#10003; Passed:</b> t +2.70" in passed


def test_the_ledger_records_a_holdout_as_n1_s():
    text = dev_report.holdout_section(
        "N2", holdout_result() | {"threshold": 2.5}, "21 October 2026"
    )
    assert text == (
        "\n## N2's holdout (run once, 21 October 2026, commit abcdef1)\n\n"
        "`data/news_reader/holdout_N2.json`; a second run is refused.\n\n"
        "| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |\n"
        "|---|---|---|---|---|\n"
        "| Variant 2: components 64, penalty 10.0 | 1,197 | +3.10 bps | +1.80 | "
        "+1.00 / +2.00 / +3.00 / +4.00 / +5.50 |\n"
        "| The count alone (the comparison) | 1,197 | −1.50 bps | −0.60 | "
        "−1.00 / −2.00 / −1.00 / −2.00 / −1.50 |\n\n"
        "**N2 failed on its holdout: t +1.80 < 2.50.**\n"
    )
    passed = dev_report.holdout_section(
        "N2", holdout_result(True, 2.7) | {"threshold": 2.5}, "21 October 2026"
    )
    assert passed.endswith("**N2 passed on its holdout: t +2.70 ≥ 2.50.**\n")


class FakeHoldout:
    """harness's holdout as it answers: once, with a lock, the years as numbers."""

    def __init__(self, root):
        self.THRESHOLDS = {"N2": 2.5, "N3": 2.5}
        self.VARIANTS = root / "variants"
        self.LOCKS = root
        self.runs = []

    def run_holdout(self, test, make):
        self.runs.append(make)
        result = holdout_result(years_as=int) | {
            "variant": json.loads((self.VARIANTS / f"{test}_chosen.json").read_text())
        }
        (self.LOCKS / f"holdout_{test}.json").write_text(json.dumps(result))
        return result


def test_the_holdout_runs_the_committed_choice_once_and_is_never_lost(
    tmp_path, monkeypatch
):
    fake = FakeHoldout(tmp_path)
    model = types.SimpleNamespace(
        make=lambda components, penalty: (components, penalty)
    )
    monkeypatch.setattr(dev_report, "DEV", tmp_path / "dev")
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(LEDGER)
    monkeypatch.setattr(dev_report, "LEDGER", ledger)

    with pytest.raises(SystemExit, match="write and commit"):
        dev_report.holdout("N2", fake, model)
    fake.VARIANTS.mkdir()
    (fake.VARIANTS / "N2_chosen.json").write_text(
        '{"components": 64, "penalty": 10.0, "variant": 2, "rule": "x"}'
    )
    dev_report.holdout("N2", fake, model)
    assert fake.runs == [(64, 10.0)]  # the committed settings, run once
    page = tmp_path / "dev" / "N2_holdout.html"
    assert "Failed" in page.read_text()
    assert ledger.read_text().count("## N2's holdout") == 1

    # Again: nothing runs; the page is written from the lock, the record not repeated.
    page.unlink()
    dev_report.holdout("N2", fake, model)
    assert fake.runs == [(64, 10.0)] and page.exists()
    assert ledger.read_text().count("## N2's holdout") == 1

    # A record lost before it was committed comes back from the lock.
    ledger.write_text(LEDGER)
    dev_report.holdout("N2", fake, model)
    assert ledger.read_text().count("## N2's holdout") == 1 and fake.runs == [
        (64, 10.0)
    ]
