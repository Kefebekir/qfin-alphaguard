import math
from datetime import UTC, datetime, timedelta

import polars as pl

from qfin_alphaguard.decision_log import (
    fills_table,
    notes_table,
    read_decision_log,
    write_decision_log,
)
from qfin_alphaguard.events import DecisionNote, Fill, Order, OrderIntent, Side

T = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)


def a_day():
    """A note, the order it explains and its fill, then the day's close."""
    note = DecisionNote(
        "QQQ", T, "long", "above the band: buy", {"price": 700.0, "upper": 699.0}
    )
    intent = OrderIntent("QQQ", Side.BUY, 85.5, 703.5, T, "decision")
    order = Order("bt-0000001", intent, T)
    fill = Fill("bt-0000001", "QQQ", Side.BUY, 85.5, 700.07, 0.01, T)
    close = DecisionNote(
        "QQQ",
        T + timedelta(hours=6),
        "flat",
        "the day's end: close the position",
        {"price": 705.0, "shares": 0.0},
    )
    return [note, intent, order, fill, close]


def test_notes_become_rows_and_what_they_saw_columns():
    table = notes_table(a_day())
    assert table["reason"].to_list() == [
        "above the band: buy",
        "the day's end: close the position",
    ]
    assert table.columns == [
        "time",
        "ticker",
        "choice",
        "reason",
        "price",
        "shares",
        "upper",
    ]
    assert table["price"].to_list() == [700.0, 705.0]
    # a value a note did not have is null, not zero
    assert table["upper"].to_list() == [699.0, None]
    assert table["shares"].to_list() == [None, 0.0]
    assert table.schema["time"] == pl.Datetime("us", "UTC")


def test_a_nan_seen_stays_nan():
    note = DecisionNote("QQQ", T, "flat", "no band", {"upper": math.nan})
    assert math.isnan(notes_table([note])["upper"][0])


def test_fills_carry_their_orders_reason():
    table = fills_table(a_day())
    assert table.height == 1
    row = table.row(0, named=True)
    assert (row["side"], row["quantity"], row["price"]) == ("buy", 85.5, 700.07)
    assert row["reason"] == "decision"
    assert row["time"] == T


def test_an_empty_log_gives_empty_tables_with_their_columns():
    assert notes_table([]).columns == ["time", "ticker", "choice", "reason"]
    assert fills_table([]).height == 0
    assert "commission" in fills_table([]).columns


def test_the_log_survives_a_trip_through_parquet(tmp_path):
    write_decision_log(a_day(), tmp_path / "log")
    notes, fills = read_decision_log(tmp_path / "log")
    assert notes.equals(notes_table(a_day()))
    assert fills.equals(fills_table(a_day()))
