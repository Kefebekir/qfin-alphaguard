"""Nikkei 225 futures (CME, yen, NIY) 1-minute bars, 2016 to 9 October 2026, from Databento.

The most traded NIY contract each day (continuous symbol NIY.v.0, rolled by volume), every
minute it traded, with each bar's instrument id so that the rolls can be seen. About 12 USD of
Databento credit (quoted by metadata.get_cost just before; the run stops above CAP_USD).
Reads DATABENTO_API_KEY from the environment and never prints it.

Usage: download_niy.py OUT_DIR
"""

import sys
from pathlib import Path

import databento as db

OUT = Path(sys.argv[1])
QUERY = {
    "dataset": "GLBX.MDP3",
    "symbols": ["NIY.v.0"],
    "stype_in": "continuous",
    "schema": "ohlcv-1m",
    "start": "2016-01-01",
    "end": "2026-10-10",
}
CAP_USD = 13.0

client = db.Historical()
cost = client.metadata.get_cost(**QUERY)
print(f"quoted cost: {cost:.2f} USD")
if cost > CAP_USD:
    sys.exit(f"above the cap of {CAP_USD} USD: nothing downloaded")
OUT.mkdir(parents=True, exist_ok=True)
target = OUT / "NIY_ohlcv1m.dbn.zst"
data = client.timeseries.get_range(**QUERY, path=target)
frame = data.to_df()
print(
    f"{len(frame):,} bars, {frame.index.min()} to {frame.index.max()}, "
    f"{frame['instrument_id'].nunique()} contracts; saved {target}"
)
(OUT / "NIY_spent.txt").write_text(f"{cost:.4f}\n")
