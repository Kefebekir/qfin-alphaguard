"""Run SQL against stored price data using DuckDB."""

from pathlib import Path

import duckdb
import polars as pl


def run_query(sql: str, path: Path) -> pl.DataFrame:
    """Run SQL against a Parquet file exposed as a table called 'prices'."""
    with duckdb.connect() as con:
        con.execute(
            f"CREATE VIEW prices AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        return con.execute(sql).pl()


def row_counts(path: Path) -> pl.DataFrame:
    """Number of rows and date range per ticker."""
    return run_query(
        """
        SELECT ticker, count(*) AS n_rows,min(date) AS first_date,max(date) AS last_date FROM prices GROUP BY ticker ORDER BY ticker""",
        path,
    )


def latest_closes(path: Path) -> pl.DataFrame:
    """Most recent close price for each ticker."""
    return run_query(
        """SELECT ticker, max(date) AS date,arg_max(close,date) AS close FROM prices GROUP BY ticker ORDER BY ticker """,
        path,
    )
