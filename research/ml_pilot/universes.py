"""Pilot for decision 0017: each year's trading universe (decision 0005), as JSON."""

import json
import sys
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.universe import trading_universes

daily = pl.read_parquet(sys.argv[1])
universes = trading_universes(daily, range(2016, 2027))
Path(sys.argv[2]).write_text(
    json.dumps({str(y): list(t) for y, t in universes.items()})
)
