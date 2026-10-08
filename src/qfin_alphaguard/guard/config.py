"""Risk limits from guard.yaml.

guard.yaml is the only place Guard's limits come from, and it changes only by
hand. It is read once at start-up. If a setting is missing, unknown, written
twice or out of range, loading fails and the engine does not trade.
"""

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


class GuardConfigError(ValueError):
    """guard.yaml cannot be used: bad YAML, wrong settings or an invalid value."""


def _require_switch(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise GuardConfigError(f"{name} must be true or false, got {value!r}")


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardConfigError(f"{name} must be a number, got {value!r}")

    if not math.isfinite(value):
        raise GuardConfigError(f"{name} must be a finite number, got {value!r}")


def _require_count(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GuardConfigError(
            f"{name} must be a whole number of at least 1, got {value!r}"
        )


@dataclass(frozen=True)
class GuardConfig:
    """The limits in guard.yaml. Field names are the keys in the file.

    No field has a default: every limit must be written in the file, so a
    forgotten or misspelt setting fails loudly instead of falling back quietly.
    """

    allow_short: bool
    max_weight: float
    max_gross_exposure: float
    max_order_usd: float
    price_collar_bps: float
    max_orders_per_min: int
    max_trades_per_symbol_day: int
    no_same_day_reversal: bool
    daily_loss_limit_pct: float
    stale_data_sec: float
    live_trading: bool

    def __post_init__(self) -> None:
        for name in ("allow_short", "no_same_day_reversal", "live_trading"):
            _require_switch(name, getattr(self, name))
        for name in ("max_orders_per_min", "max_trades_per_symbol_day"):
            _require_count(name, getattr(self, name))
        for name in (
            "max_weight",
            "max_gross_exposure",
            "max_order_usd",
            "price_collar_bps",
            "daily_loss_limit_pct",
            "stale_data_sec",
        ):
            _require_number(name, getattr(self, name))

        if not (0 < self.max_weight <= 1):
            raise GuardConfigError(
                f"max_weight must be above 0 and at most 1, got {self.max_weight}"
            )
        for name in ("max_gross_exposure", "max_order_usd", "stale_data_sec"):
            if getattr(self, name) <= 0:
                raise GuardConfigError(
                    f"{name} must be positive, got {getattr(self, name)}"
                )
        if not (0 < self.price_collar_bps < 10_000):
            raise GuardConfigError(
                "price_collar_bps must be above 0 and under 10000, "
                f"got {self.price_collar_bps}"
            )
        if not (0 < self.daily_loss_limit_pct < 100):
            raise GuardConfigError(
                "daily_loss_limit_pct must be above 0 and below 100, "
                f"got {self.daily_loss_limit_pct}"
            )
        if self.max_weight > self.max_gross_exposure:
            raise GuardConfigError(
                f"max_weight ({self.max_weight}) cannot exceed "
                f"max_gross_exposure ({self.max_gross_exposure})"
            )


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that refuses a key written twice in the same mapping.

    The plain SafeLoader keeps the last value, so a second `max_order_usd` line
    further down the file would silently replace the first.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"{key!r} is set more than once", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_guard_config(path: Path | str) -> GuardConfig:
    """Read guard.yaml and return its limits.

    Raises GuardConfigError if the file is not valid YAML, sets a key twice,
    has an unknown or missing setting, or holds a value the rules do not allow.
    A file that does not exist raises FileNotFoundError.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            # A SafeLoader subclass: tags such as !!python/object in the file
            # are refused instead of being turned into Python objects.
            raw = yaml.load(file, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise GuardConfigError(f"cannot load {path}: {error}") from error

    if not isinstance(raw, dict):
        raise GuardConfigError(
            f"{path} must be a mapping of setting: value, got {type(raw).__name__}"
        )

    expected = {field.name for field in fields(GuardConfig)}
    unknown = sorted(str(key) for key in raw.keys() - expected)
    missing = sorted(expected - raw.keys())
    problems = []
    if unknown:
        problems.append(f"unknown: {', '.join(unknown)}")
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    if problems:
        raise GuardConfigError(f"{path} has wrong settings ({'; '.join(problems)})")

    return GuardConfig(**raw)
