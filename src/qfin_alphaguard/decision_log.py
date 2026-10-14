"""The decision log as tables: what was decided and why, and what filled.

A backtest's log (backtest/engine.py) holds every DecisionNote, order and
fill in the order they happened. These functions turn it into two tables, to
read a day back decision by decision (the day book) and to store it:

- notes: one row per DecisionNote; the numbers it saw become columns, so a
  value a note did not have is null;
- fills: one row per Fill, with the reason its order was sent for.

Times stay UTC. The live engine is to write the same tables (Phase 3), so a
paper day and a backtest day can be read and compared the same way.
"""

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from qfin_alphaguard.events import DecisionNote, Fill, Order

NOTE_COLUMNS = {
    "time": pl.Datetime("us", "UTC"),
    "ticker": pl.String,
    "choice": pl.String,
    "reason": pl.String,
}
FILL_SCHEMA = {
    "time": pl.Datetime("us", "UTC"),
    "client_order_id": pl.String,
    "ticker": pl.String,
    "side": pl.String,
    "quantity": pl.Float64,
    "price": pl.Float64,
    "commission": pl.Float64,
    "reason": pl.String,  # why its order was sent, e.g. "decision"
}


def notes_table(log: Iterable[object]) -> pl.DataFrame:
    """Every DecisionNote in `log`, in order, one column per value seen."""
    notes = [record for record in log if isinstance(record, DecisionNote)]
    names = sorted({name for note in notes for name in note.seen})
    schema = NOTE_COLUMNS | {name: pl.Float64 for name in names}
    rows = [
        {
            "time": note.time,
            "ticker": note.ticker,
            "choice": note.choice,
            "reason": note.reason,
        }
        | {name: note.seen.get(name) for name in names}
        for note in notes
    ]
    return pl.DataFrame(rows, schema=schema)


def fills_table(log: Iterable[object]) -> pl.DataFrame:
    """Every Fill in `log`, in order, with its order's reason."""
    reasons: dict[str, str] = {}
    rows = []
    for record in log:
        if isinstance(record, Order):
            reasons[record.client_order_id] = record.intent.reason
        elif isinstance(record, Fill):
            rows.append(
                {
                    "time": record.time,
                    "client_order_id": record.client_order_id,
                    "ticker": record.ticker,
                    "side": record.side.value,
                    "quantity": float(record.quantity),
                    "price": float(record.price),
                    "commission": float(record.commission),
                    "reason": reasons.get(record.client_order_id, ""),
                }
            )
    return pl.DataFrame(rows, schema=FILL_SCHEMA)


def write_decision_log(log: Iterable[object], directory: Path) -> None:
    """Store `log`'s notes and fills as notes.parquet and fills.parquet."""
    log = list(log)
    directory.mkdir(parents=True, exist_ok=True)
    notes_table(log).write_parquet(directory / "notes.parquet")
    fills_table(log).write_parquet(directory / "fills.parquet")


def read_decision_log(directory: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The notes and fills write_decision_log stored in `directory`."""
    return (
        pl.read_parquet(directory / "notes.parquet"),
        pl.read_parquet(directory / "fills.parquet"),
    )
