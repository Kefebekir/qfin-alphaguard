import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from qfin_alphaguard.challengers import (
    TOTAL_ALPHA,
    ChallengerCard,
    ChallengerError,
    load_card,
    load_cards,
    weights_path,
)

WEIGHTS = b"frozen weights"
CARD = {
    "name": "day-sizer",
    "description": "Sizes the rule's day from what is known at 10:00.",
    "owner": "Efe",
    "code_commit": "a" * 40,
    "weights": "data/models/day-sizer.npz",
    "weights_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
    "trained_from": date(2016, 1, 26),
    "trained_to": date(2026, 10, 9),
    "frozen_at": datetime(2026, 10, 16, 12, 0, tzinfo=UTC),
    "record_from": date(2026, 10, 19),
    "scale": 1.1,
    "clip": 0.05,
    "alpha": 0.025,
    "retired_on": None,
}


def write(tmp_path, card=None, name=None):
    card = CARD if card is None else card
    path = tmp_path / f"{name or card['name']}.yaml"
    path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
    return path


def test_a_written_card_loads_as_the_same_card(tmp_path):
    assert load_card(write(tmp_path)) == ChallengerCard(**CARD)


def test_the_card_gives_its_betting_test():
    test = ChallengerCard(**CARD).test()
    assert (test.scale, test.clip, test.alpha, test.threshold) == (1.1, 0.05, 0.025, 40)


def test_the_file_is_named_after_the_challenger(tmp_path):
    with pytest.raises(ChallengerError, match="named after"):
        load_card(write(tmp_path, name="other"))


@pytest.mark.parametrize("change", ["unknown", "missing"])
def test_every_setting_is_required_and_none_unknown(tmp_path, change):
    card = dict(CARD)
    if change == "unknown":
        card["learning_rate"] = 0.1
    else:
        del card["alpha"]
    with pytest.raises(ChallengerError, match=change):
        load_card(write(tmp_path, card))


def test_a_setting_written_twice_is_refused(tmp_path):
    path = write(tmp_path)
    path.write_text(path.read_text() + "alpha: 0.05\n", encoding="utf-8")
    with pytest.raises(ChallengerError, match="more than once"):
        load_card(path)


def test_frozen_at_must_say_its_time_zone(tmp_path):
    path = write(tmp_path)
    text = path.read_text().replace("2026-10-16 12:00:00+00:00", "2026-10-16 12:00:00")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ChallengerError, match="time zone"):
        load_card(path)


def test_a_time_in_another_zone_is_kept_in_utc(tmp_path):
    path = write(tmp_path)
    text = path.read_text().replace(
        "2026-10-16 12:00:00+00:00", "2026-10-16 14:00:00+02:00"
    )
    path.write_text(text, encoding="utf-8")
    assert load_card(path).frozen_at == datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("name", "Day Sizer", "name"),
        ("code_commit", "abc123", "code_commit"),
        ("weights", "models/day-sizer.npz", "data/"),
        ("weights", "data/../guard.yaml", "data/"),  # out of data/ again
        ("weights_sha256", "ABC", "weights_sha256"),
        ("trained_to", date(2015, 1, 1), "trained_from"),
        ("record_from", date(2026, 10, 9), "last training day"),
        ("record_from", date(2026, 10, 16), "frozen"),
        ("retired_on", date(2026, 10, 18), "retired_on"),
        ("frozen_at", CARD["frozen_at"].replace(tzinfo=None), "UTC"),  # no zone
        ("trained_from", "2016-01-26", "date"),
        ("scale", 0.0, "scale"),
        ("scale", True, "scale"),
        ("clip", -0.05, "clip"),
        ("alpha", 0.0, "alpha"),
        ("alpha", 0.03, "total"),  # more than the whole platform's alpha
        ("description", " ", "description"),
    ],
)
def test_a_card_that_breaks_a_rule_is_refused(key, value, match):
    with pytest.raises(ChallengerError, match=match):
        ChallengerCard(**{**CARD, key: value})


def test_cards_may_share_the_total_alpha_exactly(tmp_path):
    for n in range(5):
        write(tmp_path, {**CARD, "name": f"card-{n}", "alpha": 0.005})
    assert [c.name for c in load_cards(tmp_path)] == [f"card-{n}" for n in range(5)]
    assert TOTAL_ALPHA == 0.025


def test_a_retired_card_s_alpha_stays_spent(tmp_path):
    write(tmp_path, {**CARD, "alpha": 0.02})
    write(
        tmp_path,
        {**CARD, "name": "old-one", "alpha": 0.01, "retired_on": date(2026, 12, 31)},
    )
    with pytest.raises(ChallengerError, match="add to 0.03"):
        load_cards(tmp_path)


def test_the_frozen_weights_are_accepted(tmp_path):
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / CARD["weights"]).write_bytes(WEIGHTS)
    assert weights_path(ChallengerCard(**CARD), tmp_path) == tmp_path / CARD["weights"]


def test_weights_changed_after_freezing_are_refused(tmp_path):
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / CARD["weights"]).write_bytes(WEIGHTS + b"!")
    with pytest.raises(ChallengerError, match="not the file that was frozen"):
        weights_path(ChallengerCard(**CARD), tmp_path)


def test_missing_weights_are_refused(tmp_path):
    with pytest.raises(ChallengerError, match="not found"):
        weights_path(ChallengerCard(**CARD), tmp_path)


def test_the_repository_s_cards_are_valid_and_within_the_total():
    cards = load_cards(Path(__file__).parents[1] / "challengers")
    assert sum(c.alpha for c in cards) <= TOTAL_ALPHA
