"""Efe's development runs (research/news_reader/dev_report.py): counting, the ledger, the page."""

import importlib.util
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
