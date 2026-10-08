import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from qfin_alphaguard.guard.config import (
    GuardConfig,
    GuardConfigError,
    load_guard_config,
)

REPO_GUARD_YAML = Path(__file__).resolve().parents[1] / "guard.yaml"

# A valid set of limits, kept separate from the repository's guard.yaml so a
# limit can change there without touching these tests.
VALID = {
    "allow_short": False,
    "max_weight": 0.25,
    "max_gross_exposure": 1.0,
    "max_order_usd": 2000,
    "price_collar_bps": 50,
    "max_orders_per_min": 10,
    "max_trades_per_symbol_day": 3,
    "no_same_day_reversal": True,
    "daily_loss_limit_pct": 2.0,
    "stale_data_sec": 15,
    "live_trading": False,
}


def make_config(**changes):
    return GuardConfig(**{**VALID, **changes})


def write_text(tmp_path, text):
    path = tmp_path / "guard.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_settings(tmp_path, settings):
    return write_text(tmp_path, yaml.safe_dump(settings, sort_keys=False))


def test_loads_a_valid_file(tmp_path):
    assert load_guard_config(write_settings(tmp_path, VALID)) == make_config()


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_guard_config(tmp_path / "guard.yaml")


@pytest.mark.parametrize("text", ["", "- max_weight: 0.25\n", "just text\n"])
def test_file_must_hold_a_mapping(tmp_path, text):
    with pytest.raises(GuardConfigError, match="mapping"):
        load_guard_config(write_text(tmp_path, text))


def test_broken_yaml_is_a_config_error(tmp_path):
    with pytest.raises(GuardConfigError):
        load_guard_config(write_text(tmp_path, "max_weight: [0.25\n"))


def test_setting_written_twice_is_rejected(tmp_path):
    # A plain YAML loader keeps the last value, which here would raise the limit.
    text = yaml.safe_dump(VALID) + "max_order_usd: 20000\n"
    with pytest.raises(GuardConfigError, match="max_order_usd"):
        load_guard_config(write_text(tmp_path, text))


def test_misspelt_setting_is_rejected(tmp_path):
    settings = dict(VALID)
    settings["max_oder_usd"] = settings.pop("max_order_usd")
    with pytest.raises(GuardConfigError, match="max_oder_usd"):
        load_guard_config(write_settings(tmp_path, settings))


def test_missing_setting_is_rejected(tmp_path):
    settings = dict(VALID)
    del settings["daily_loss_limit_pct"]
    with pytest.raises(GuardConfigError, match="daily_loss_limit_pct"):
        load_guard_config(write_settings(tmp_path, settings))


def test_python_tags_in_the_file_are_refused(tmp_path):
    # An unsafe loader would call len() here and read 2, a valid order limit.
    settings = dict(VALID)
    del settings["max_order_usd"]
    text = (
        yaml.safe_dump(settings)
        + "max_order_usd: !!python/object/apply:builtins.len [[1, 2]]\n"
    )
    with pytest.raises(GuardConfigError):
        load_guard_config(write_text(tmp_path, text))


def test_quoted_false_in_the_file_is_rejected(tmp_path):
    # Quoted, "false" is a string, and every non-empty string is truthy.
    path = write_settings(tmp_path, {**VALID, "live_trading": "false"})
    with pytest.raises(GuardConfigError, match="live_trading"):
        load_guard_config(path)


def test_config_cannot_be_changed_after_creation():
    config = make_config()
    with pytest.raises(FrozenInstanceError):
        config.live_trading = True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_weight", 1.0),  # the upper bound is allowed
        ("max_gross_exposure", 1),  # whole numbers are fine where a float is expected
        ("stale_data_sec", 0.5),  # so are fractions of a second
        ("price_collar_bps", 9_999),
        # The validator checks that limits make sense, not the project's scope;
        # the repository test at the bottom pins the scope.
        ("allow_short", True),
        ("live_trading", True),
    ],
)
def test_values_inside_the_rules_are_accepted(name, value):
    assert getattr(make_config(**{name: value}), name) == value


@pytest.mark.parametrize(
    ("name", "value"),
    [
        # Switches must be real booleans.
        ("live_trading", "false"),
        ("live_trading", 0),
        ("allow_short", None),
        ("no_same_day_reversal", "yes"),
        # bool is a subclass of int, so a plain number check lets True in as 1.
        ("max_order_usd", True),
        ("max_trades_per_symbol_day", True),
        # Numbers must be numbers, not text that looks like one.
        ("max_order_usd", "2000"),
        # YAML reads .inf and .nan as floats; .inf would switch a limit off.
        ("max_order_usd", math.inf),
        ("stale_data_sec", math.inf),
        ("daily_loss_limit_pct", math.nan),
        ("max_order_usd", 0),
        ("max_order_usd", -500),
        ("max_weight", 0),
        ("max_weight", 1.01),
        ("max_gross_exposure", 0),
        ("price_collar_bps", 0),
        ("price_collar_bps", 10_000),  # a sell limit would be at or below zero
        ("max_orders_per_min", 0),
        ("max_orders_per_min", 2.5),
        ("max_trades_per_symbol_day", 0),
        ("daily_loss_limit_pct", 0),
        ("daily_loss_limit_pct", 100),
        ("stale_data_sec", 0),
    ],
)
def test_invalid_value_is_rejected_by_name(name, value):
    with pytest.raises(GuardConfigError, match=name):
        make_config(**{name: value})


def test_one_stock_cannot_exceed_the_whole_exposure():
    with pytest.raises(GuardConfigError, match="max_weight"):
        make_config(max_weight=0.5, max_gross_exposure=0.4)


def test_repository_guard_yaml_is_valid():
    assert isinstance(load_guard_config(REPO_GUARD_YAML), GuardConfig)


def test_repository_guard_yaml_is_paper_only_long_only_and_unlevered():
    # Going live in Phase 3b means changing this test by hand, in the same
    # commit that changes guard.yaml.
    config = load_guard_config(REPO_GUARD_YAML)
    assert config.live_trading is False
    assert config.allow_short is False
    assert config.max_gross_exposure <= 1
