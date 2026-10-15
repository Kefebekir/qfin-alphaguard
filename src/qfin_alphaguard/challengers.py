"""Challenger cards: a model frozen, and its test written down, before its record starts.

Decision 0018 lets machine learning trade only after it beats the rule forward; decision 0020
fixes how. Each challenger has a card, `challengers/<name>.yaml`, committed before its
record's first day:

- what it is and who owns it, and the commit of the code that runs it;
- its weights: a file under the git-ignored `data/`, and the SHA-256 of its bytes. Weights
  whose hash differs are refused, so a model cannot be changed during its record;
- the days it was trained on, when it was frozen, and the first day of its record, after
  both: only days it could not have seen count;
- its test (evidence.BettingTest): the scale that puts it at the rule's risk and the clip,
  both from its training days (evidence.training_settings), and its alpha, its share of
  TOTAL_ALPHA. The shares of every card in the directory, retired ones too, may add to at
  most TOTAL_ALPHA: a share is spent once a record starts, so a card is never deleted, only
  given a retirement day.

Every setting is required and an unknown one is an error, as in guard.yaml.
"""

import hashlib
import math
import re
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from qfin_alphaguard.evidence import BettingTest

NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")

# The chance of promoting any challenger that does not beat the rule, over every card ever.
TOTAL_ALPHA = 0.025


class ChallengerError(ValueError):
    """A challenger card or its weights cannot be used."""


@dataclass(frozen=True)
class ChallengerCard:
    """One frozen challenger. Field names are the keys of its card."""

    name: str
    description: str
    owner: str
    code_commit: str  # the commit of the code that runs it
    weights: str  # a path under data/, from the repository root
    weights_sha256: str
    trained_from: date
    trained_to: date
    frozen_at: datetime  # UTC
    record_from: date  # the first day that counts
    # from evidence.training_settings on its training days:
    scale: float  # the rule's volatility over its own
    clip: float  # the largest daily difference counted in full
    alpha: float  # its share of TOTAL_ALPHA
    retired_on: date | None  # null while its record runs

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not NAME.fullmatch(self.name):
            raise ChallengerError(
                f"name must be lower-case words joined by '-', got {self.name!r}"
            )
        for key in ("description", "owner"):
            if (
                not isinstance(getattr(self, key), str)
                or not getattr(self, key).strip()
            ):
                raise ChallengerError(f"{key} must be text")
        if not isinstance(self.code_commit, str) or not COMMIT.fullmatch(
            self.code_commit
        ):
            raise ChallengerError(
                f"code_commit must be a full 40-character commit, got {self.code_commit!r}"
            )
        if (
            not isinstance(self.weights, str)
            or not self.weights.startswith("data/")
            or ".." in Path(self.weights).parts
        ):
            raise ChallengerError(
                f"weights must be a path under data/, got {self.weights!r}"
            )
        if not isinstance(self.weights_sha256, str) or not SHA256.fullmatch(
            self.weights_sha256
        ):
            raise ChallengerError(
                "weights_sha256 must be 64 lower-case hexadecimal characters"
            )
        for key in ("trained_from", "trained_to", "record_from"):
            value = getattr(self, key)
            if isinstance(value, datetime) or not isinstance(value, date):
                raise ChallengerError(f"{key} must be a date, got {value!r}")
        if self.retired_on is not None and (
            isinstance(self.retired_on, datetime)
            or not isinstance(self.retired_on, date)
        ):
            raise ChallengerError(
                f"retired_on must be a date or null, got {self.retired_on!r}"
            )
        if (
            not isinstance(self.frozen_at, datetime)
            or self.frozen_at.utcoffset() is None
            or self.frozen_at.utcoffset().total_seconds() != 0
        ):
            raise ChallengerError(
                f"frozen_at must be a UTC date and time, got {self.frozen_at!r}"
            )
        if not self.trained_from <= self.trained_to:
            raise ChallengerError("trained_from must not be after trained_to")
        if not self.trained_to < self.record_from:
            raise ChallengerError("record_from must be after the last training day")
        if not self.frozen_at.date() < self.record_from:
            raise ChallengerError("record_from must be after the day it was frozen")
        if self.retired_on is not None and self.retired_on < self.record_from:
            raise ChallengerError("retired_on must not be before record_from")
        for key in ("scale", "clip", "alpha"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ChallengerError(f"{key} must be a finite number, got {value!r}")
        if not self.alpha <= TOTAL_ALPHA:
            raise ChallengerError(
                f"alpha must be at most the total, {TOTAL_ALPHA}, got {self.alpha!r}"
            )
        try:
            self.test()
        except ValueError as error:
            raise ChallengerError(str(error)) from error

    def test(self) -> BettingTest:
        """Its betting test: promoted when its evidence reaches 1 / alpha."""
        return BettingTest(scale=self.scale, clip=self.clip, alpha=self.alpha)


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that refuses a key written twice."""

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


def load_card(path: Path | str) -> ChallengerCard:
    """Read and check one card."""
    path = Path(path)
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ChallengerError(f"cannot read {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ChallengerError(f"{path} must be a mapping of setting: value")
    expected = {f.name for f in fields(ChallengerCard)}
    unknown, missing = (
        sorted(map(str, raw.keys() - expected)),
        sorted(expected - raw.keys()),
    )
    if unknown or missing:
        raise ChallengerError(
            f"{path}: unknown settings {unknown}, missing settings {missing}"
        )
    if isinstance(raw["frozen_at"], datetime) and raw["frozen_at"].tzinfo is None:
        raise ChallengerError(f"{path}: frozen_at must give its time zone (Z for UTC)")
    if isinstance(raw["frozen_at"], datetime):
        raw["frozen_at"] = raw["frozen_at"].astimezone(UTC)
    card = ChallengerCard(**raw)
    if card.name != path.stem:
        raise ChallengerError(
            f"{path}: the file must be named after the challenger, {card.name}.yaml"
        )
    return card


def load_cards(directory: Path | str = "challengers") -> list[ChallengerCard]:
    """Every card in `directory`, retired ones included, by name, within TOTAL_ALPHA."""
    cards = sorted(
        (load_card(p) for p in Path(directory).glob("*.yaml")), key=lambda c: c.name
    )
    spent = math.fsum(c.alpha for c in cards)
    # 1e-9: decimal shares that add to the total could add to a hair above it in binary
    if spent > TOTAL_ALPHA * (1 + 1e-9):
        raise ChallengerError(
            f"the cards' alphas add to {spent:.6g}, more than the total {TOTAL_ALPHA}; "
            "a retired card's share stays spent"
        )
    return cards


def weights_path(card: ChallengerCard, root: Path | str = ".") -> Path:
    """The challenger's weights file, after checking its bytes against the card."""
    path = Path(root) / card.weights
    if not path.is_file():
        raise ChallengerError(f"{card.name}: weights {card.weights} not found")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != card.weights_sha256:
        raise ChallengerError(
            f"{card.name}: {card.weights} is not the file that was frozen (SHA-256 differs)"
        )
    return path
