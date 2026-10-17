import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

import pytest

from qfin_alphaguard.events import Bar, Order, OrderIntent, Side
from qfin_alphaguard.live.journal import (
    Journal,
    JournalError,
    decode,
    encode,
    filled_positions,
    rebuild,
)
from qfin_alphaguard.live.orders import (
    Accepted,
    Cancelled,
    CancelRefused,
    CancelRequested,
    Expired,
    FillReport,
    InvalidTransition,
    Rejected,
    Sent,
    Status,
    replay,
)

OPEN = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)
WRITTEN = datetime(2026, 10, 19, 13, 31, 0, 250000, tzinfo=UTC)


def bar(minute, close=600.5):
    return Bar(
        "QQQ", OPEN + timedelta(minutes=minute), 60, 600.0, 601.0, 599.0, close, 1532
    )


def make_order(client_order_id="qqq-2026-10-19-1", side=Side.BUY, quantity=100.0):
    t = OPEN + timedelta(minutes=30)
    intent = OrderIntent("QQQ", side, quantity, 600.12, t, "decision")
    return Order(client_order_id, intent, t)


def at(seconds):
    return OPEN + timedelta(minutes=30, seconds=seconds)


ID = "qqq-2026-10-19-1"
EVERY_KIND = [
    bar(0),
    make_order(),
    Sent(ID, at(0)),
    Accepted(ID, at(0.1)),
    FillReport(ID, at(1), 0.123456789, 600.0049999),
    CancelRequested(ID, at(2)),
    CancelRefused(ID, at(2.5), "order is not cancelable"),
    Cancelled(ID, at(3)),
    Expired(ID, at(4)),
    Rejected(ID, at(5), "insufficient buying power"),
    make_order("qqq-2026-10-19-2", Side.SELL, 0.5),
]


def journal(path, now=WRITTEN):
    return Journal(path, now=lambda: now)


def test_every_kind_of_record_reads_back_as_written(tmp_path):
    path = tmp_path / "2026-10-19.jsonl"
    with journal(path) as j:
        for record in EVERY_KIND:
            j.write(record)
        assert j.records == EVERY_KIND
    with journal(path) as j:
        assert j.records == EVERY_KIND
        assert j.dropped is None


def test_a_record_is_one_line_a_person_can_read(tmp_path):
    path = tmp_path / "day.jsonl"
    with journal(path) as j:
        j.write(Sent(ID, at(0)))
        j.write(FillReport(ID, at(1), 40.0, 600.01))
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "seq": 1,
            "at": "2026-10-19T13:31:00.250000+00:00",
            "type": "Sent",
            "client_order_id": ID,
            "time": "2026-10-19T14:00:00+00:00",
        },
        {
            "seq": 2,
            "at": "2026-10-19T13:31:00.250000+00:00",
            "type": "FillReport",
            "client_order_id": ID,
            "time": "2026-10-19T14:00:01+00:00",
            "filled_quantity": 40.0,
            "average_price": 600.01,
        },
    ]


def test_each_record_is_forced_to_disk_before_write_returns(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr("qfin_alphaguard.live.journal.os.fsync", synced.append)
    path = tmp_path / "day.jsonl"
    with journal(path) as j:
        j.write(bar(0))
        assert len(synced) == 1
        # Out of Python's buffer too: another reader already sees the line.
        assert json.loads(path.read_text())["type"] == "Bar"
        j.write(bar(1))
        assert len(synced) == 2


def test_a_record_that_cannot_be_written_is_not_written(tmp_path):
    path = tmp_path / "day.jsonl"
    with journal(path) as j:
        with pytest.raises(TypeError, match="dict"):
            j.write({"type": "Bar"})
        j.write(bar(0))
    with journal(path) as j:
        assert j.records == [bar(0)]


def test_a_reopened_journal_carries_on_its_sequence(tmp_path):
    path = tmp_path / "day.jsonl"
    with journal(path) as j:
        j.write(bar(0))
    with journal(path) as j:
        j.write(bar(1))
    assert [json.loads(line)["seq"] for line in path.read_text().splitlines()] == [1, 2]
    with journal(path) as j:
        assert j.records == [bar(0), bar(1)]


def test_a_line_cut_short_by_a_crash_is_dropped(tmp_path):
    path = tmp_path / "day.jsonl"
    with journal(path) as j:
        j.write(bar(0))
    whole = path.read_text()
    cut = '{"seq":2,"at":"2026-10-19T13:31:00+00:00","type":"Ba'
    path.write_text(whole + cut)
    with journal(path) as j:
        assert j.records == [bar(0)]
        assert j.dropped == cut
        assert path.read_text() == whole  # the file is whole again
        j.write(bar(1))  # on a line of its own, numbered 2
    with journal(path) as j:
        assert j.records == [bar(0), bar(1)] and j.dropped is None


@pytest.mark.parametrize(
    "lines",
    [
        ['{"seq":1,"type":"Ba', '{"seq":2,"type":"Sent"}'],  # broken before the last
        [
            '{"seq":2,"type":"Sent","client_order_id":"a","time":"2026-10-19T14:00:00+00:00"}'
        ],
        [
            '{"seq":1,"type":"Sent","client_order_id":"a","time":"2026-10-19T14:00:00+00:00"}',
            '{"seq":1,"type":"Sent","client_order_id":"a","time":"2026-10-19T14:00:00+00:00"}',
        ],
        ['{"type":"Sent","client_order_id":"a","time":"2026-10-19T14:00:00+00:00"}'],
        ["[1, 2]"],
        [""],  # an empty line
    ],
)
def test_a_damaged_journal_stops_the_engine(tmp_path, lines):
    path = tmp_path / "day.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(JournalError):
        Journal(path)


@pytest.mark.parametrize(
    "data",
    [
        {"type": "Signal"},
        {"client_order_id": "a"},
        {"type": "Sent", "client_order_id": "a"},  # no time
        {
            "type": "Sent",
            "client_order_id": "a",
            "time": "2026-10-19T14:00:00+00:00",
            "x": 1,
        },
        {
            "type": "Sent",
            "client_order_id": "a",
            "time": "2026-10-19T14:00:00",
        },  # no zone
        {"type": "Sent", "client_order_id": "a", "time": 5},
        {
            "type": "Order",
            "client_order_id": "a",
            "intent": {},
            "sent_at": "2026-10-19T14:00:00+00:00",
        },
        {
            **encode(make_order()),
            "intent": {**encode(make_order())["intent"], "side": "short"},
        },
        {**encode(bar(0)), "high": 1.0},  # a high below the close
    ],
)
def test_a_record_that_cannot_be_read_stops_the_engine(data):
    with pytest.raises(JournalError):
        decode(data)


def test_a_field_the_record_does_not_have_is_named():
    data = {**encode(Sent(ID, at(0))), "x": 1}
    with pytest.raises(JournalError, match=r"Sent has no field \['x'\]"):
        decode(data)


def test_an_order_is_written_with_its_intent():
    data = encode(make_order(side=Side.SELL))
    assert data["intent"]["side"] == "sell"
    assert data["intent"]["time"] == "2026-10-19T14:00:00+00:00"
    assert decode(data) == make_order(side=Side.SELL)


def test_only_one_engine_may_hold_a_journal(tmp_path):
    path = tmp_path / "day.jsonl"
    first = journal(path)
    with pytest.raises(JournalError, match="another engine"):
        Journal(path)
    first.close()
    with journal(path) as second:
        second.write(bar(0))


# Rebuilding the engine's state.


def test_a_rebuilt_order_is_the_order_its_events_make():
    events = [
        Sent(ID, at(0)),
        Accepted(ID, at(0.1)),
        FillReport(ID, at(1), 40.0, 600.01),
        CancelRequested(ID, at(2)),
        FillReport(ID, at(2.1), 70.0, 600.02),
        Cancelled(ID, at(3)),
    ]
    other = make_order("qqq-2026-10-19-2", Side.SELL, 70.0)
    rebuilt = rebuild(
        [
            bar(0),
            make_order(),
            bar(1),
            *events,
            other,
            Sent(other.client_order_id, at(9)),
        ]
    )
    assert rebuilt.bars == [bar(0), bar(1)]
    assert list(rebuilt.orders) == [ID, other.client_order_id]
    assert rebuilt.orders[ID] == replay(make_order(), events)
    assert rebuilt.orders[ID].status is Status.CANCELLED
    assert rebuilt.orders[other.client_order_id].status is Status.SUBMITTED


def sent_and_filled(order, total, price):
    cid = order.client_order_id
    return [order, Sent(cid, at(0)), FillReport(cid, at(1), total, price)]


def test_a_journal_s_fills_add_up_to_positions():
    spy = Order("spy", OrderIntent("SPY", Side.SELL, 5.0, 500.0, at(0)), at(0))
    rebuilt = rebuild(
        [
            *sent_and_filled(make_order("b", Side.BUY, 100.0), 100.0, 600.0),
            *sent_and_filled(make_order("s", Side.SELL, 30.0), 12.5, 601.0),
            *sent_and_filled(spy, 5.0, 500.0),
            make_order("unsent", Side.BUY, 50.0),
        ]
    )
    assert filled_positions(rebuilt.orders.values()) == {"QQQ": 87.5, "SPY": -5.0}
    flat = rebuild(
        [
            *sent_and_filled(make_order("b", Side.BUY, 100.0), 100.0, 600.0),
            *sent_and_filled(make_order("c", Side.SELL, 100.0), 100.0, 601.0),
        ]
    )
    assert filled_positions(flat.orders.values()) == {}


def test_an_event_for_an_order_the_engine_did_not_make_stops_it():
    with pytest.raises(JournalError, match="did not make"):
        rebuild([Accepted("manual-order", at(0))])


def test_an_order_made_twice_stops_the_engine():
    with pytest.raises(JournalError, match="twice"):
        rebuild([make_order(), make_order()])


def test_an_event_the_rules_refused_is_refused_again():
    # The engine stopped on it when it arrived; it does not start over it either.
    with pytest.raises(InvalidTransition):
        rebuild([make_order(), Sent(ID, at(0)), Sent(ID, at(1))])


# A real crash: a process writing the journal is killed mid-stream, again and again.

WRITER = """
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from qfin_alphaguard.events import Bar
from qfin_alphaguard.live.journal import Journal

journal = Journal(Path(sys.argv[1]))
start = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)
i = len(journal.records)
print("ready", flush=True)
while True:
    minute = start + timedelta(minutes=i)
    journal.write(Bar("QQQ", minute, 60, 600.0, 601.0, 599.0, 600.5, i))
    i += 1
"""


def test_a_journal_survives_its_writer_being_killed(tmp_path):
    path = tmp_path / "day.jsonl"
    rng = random.Random(21)
    for _ in range(5):
        writer = subprocess.Popen(
            [sys.executable, "-c", WRITER, str(path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert writer.stdout.readline() == "ready\n"
        time.sleep(rng.uniform(0.02, 0.2))
        writer.kill()  # SIGKILL: no clean-up, as a crash
        writer.wait()
        writer.stdout.close()
        with Journal(path) as j:  # the dead writer's lock is gone
            volumes = [b.volume for b in j.records]
        assert volumes == list(range(len(volumes)))  # nothing lost, nothing twice
    assert len(volumes) > 20
