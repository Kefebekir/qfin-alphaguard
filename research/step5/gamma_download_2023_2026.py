"""QQQ options open interest, 2023-01-03 to 2026-10-09, one session at a time, only the
window in which open interest is published (09:31 ET before OPRA's Pillar format on 28 March
2023, 06:30 ET after it). Resumable: one file a session. Six requests at once; each reserves
its cost in the ledger before it is sent, so the spend never passes CAP, across runs too."""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import databento as db
import polars as pl

from qfin_alphaguard.sessions import sessions

CAP = 95.0
client = db.Historical(os.environ["DATABENTO_API_KEY"])
OUT = Path("data/raw/opra/QQQ_daily")
LEDGER = OUT / "_spent.txt"
lock = threading.Lock()
state = {
    "spent": float(LEDGER.read_text()) if LEDGER.exists() else 0.0,
    "stop": None,
    "done": 0,
}


def fetch(day: date) -> None:
    out = OUT / f"{day}.parquet"
    if out.exists() or state["stop"]:
        return
    window = (
        ("13:00", "15:30") if day < date(2023, 3, 28) else ("10:00", "12:00")
    )  # summer and winter time
    kw = {
        "dataset": "OPRA.PILLAR",
        "symbols": ["QQQ.OPT"],
        "stype_in": "parent",
        "schema": "statistics",
        "start": f"{day}T{window[0]}",
        "end": f"{day}T{window[1]}",
    }
    for attempt in range(1, 6):
        try:
            cost = client.metadata.get_cost(**kw)
            break
        except (db.BentoError, OSError):
            if attempt == 5:
                raise
            time.sleep(15 * attempt)
    with lock:
        if state["stop"]:
            return
        if state["spent"] + cost > CAP:
            state["stop"] = (
                f"stopping before {day}: the spend would reach {state['spent'] + cost:.2f} USD"
            )
            return
        state["spent"] += cost  # reserved before the request goes
        LEDGER.write_text(f"{state['spent']:.4f}")
    for attempt in range(
        1, 6
    ):  # the server sometimes times out (504): wait and try again
        try:
            df = client.timeseries.get_range(**kw).to_df()
            break
        except (db.BentoError, OSError) as error:  # any Databento or network error
            if (
                attempt == 5
            ):  # give this session up, keep the rest going; it stays missing
                with lock:
                    state.setdefault("missing", []).append(f"{day}: {error!r}")
                return
            time.sleep(15 * attempt)
    oi = df[df["stat_type"] == 9].reset_index() if len(df) else df
    if len(oi) == 0:
        with lock:
            state["stop"] = f"no open interest in the window on {day}: stopping to look"
        return
    cols = [
        "ts_event",
        "ts_recv",
        "instrument_id",
        "quantity",
        "publisher_id",
        "symbol",
    ]
    pl.from_pandas(oi[cols]).write_parquet(out)
    with lock:
        state["done"] += 1
        if state["done"] % 50 == 0:
            print(
                f"{state['done']} sessions, spent {state['spent']:.2f} USD", flush=True
            )


days = [s.day for s in sessions(date(2023, 1, 3), date(2026, 10, 9))]
with ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(fetch, days))
print(f"done: {state['done']} sessions this run, spent {state['spent']:.2f} USD in all")
for line in state.get("missing", []):
    print("missing", line)
if state["stop"]:
    print(state["stop"])
    sys.exit(1)
