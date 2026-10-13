"""Pilot for decision 0017: the top 10 of the year's 50 against the 50, by year."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

PRED = Path(sys.argv[1])
UNIVERSES = {int(k): v for k, v in json.loads(Path(sys.argv[2]).read_text()).items()}
H = int(sys.argv[3])
SIGNAL = sys.argv[4]

pred = pl.read_parquet(PRED)
label = f"fwd_{H}"
rows = []
for day in pred["date"].unique().sort().to_list()[::H]:
    cross = pred.filter(
        (pl.col("date") == day) & pl.col("ticker").is_in(UNIVERSES[day.year])
    ).drop_nulls(label)
    if cross.height < 20:
        continue
    returns = np.expm1(cross[label].to_numpy())
    top = np.argsort(-cross[SIGNAL].to_numpy(), kind="stable")[:10]
    rows.append((day.year, returns[top].mean() - returns.mean()))
active = pl.DataFrame(rows, schema=["year", "active"], orient="row")
by_year = active.group_by("year").agg(pl.col("active").sum()).sort("year")
print({year: f"{value:+.1%}" for year, value in by_year.rows()})
print(f"periods won: {(active['active'] > 0).mean():.0%}")
