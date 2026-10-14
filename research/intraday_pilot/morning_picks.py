"""The stocks morning_ml.py's model would buy, for gap_quotes.py to price.

Each day, the k stocks with the highest score among all the members it
scored, and among its 100 most traded; one row per day and stock, without
repeats.

Usage: morning_picks.py SCORES OUT [SIGNAL]
"""

import sys
from pathlib import Path

import polars as pl

SCORES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SIGNAL = sys.argv[3] if len(sys.argv) > 3 else "gbt"
K = 10

scores = pl.read_parquet(SCORES)


def top(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        pl.col(SIGNAL).rank("ordinal", descending=True).over("day") <= K
    ).select("day", "ticker")


picks = pl.concat([top(scores), top(scores.filter("top100"))]).unique()
picks.sort("day", "ticker").write_parquet(OUT)
print(picks.height, "stock-days on", picks["day"].n_unique(), "days")
