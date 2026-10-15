"""QQQ and SPY options open interest, 2016-2022, from Databento OPRA statistics.
Stops if the spend would pass the cap. Keeps only open interest (stat_type 9), one
row per record: the quantity, the record's times and its OCC symbol."""

import os
import sys
from pathlib import Path

import databento as db
import polars as pl

CAP = 15.0
client = db.Historical(os.environ["DATABENTO_API_KEY"])
OUT = Path("data/raw/opra")
spent = 0.0
for sym in ("QQQ.OPT", "SPY.OPT"):
    for y in range(2016, 2023):
        out = OUT / f"{sym[:3]}_oi_{y}.parquet"
        if out.exists():
            print("have", out)
            continue
        kw = {
            "dataset": "OPRA.PILLAR",
            "symbols": [sym],
            "stype_in": "parent",
            "schema": "statistics",
            "start": f"{y}-01-01",
            "end": f"{y + 1}-01-01",
        }
        cost = client.metadata.get_cost(**kw)
        if spent + cost > CAP:
            sys.exit(
                f"stopping: {sym} {y} would take the spend to {spent + cost:.2f} USD"
            )
        df = client.timeseries.get_range(**kw).to_df()
        spent += cost
        oi = df[df["stat_type"] == 9].reset_index()
        frame = pl.from_pandas(
            oi[
                [
                    "ts_event",
                    "ts_recv",
                    "instrument_id",
                    "quantity",
                    "publisher_id",
                    "symbol",
                ]
            ]
        )
        frame.write_parquet(out)
        print(
            f"{sym} {y}: {cost:.2f} USD (total {spent:.2f}), {len(df)} records, {frame.height} open interest rows",
            flush=True,
        )
print(f"spent {spent:.2f} USD")
