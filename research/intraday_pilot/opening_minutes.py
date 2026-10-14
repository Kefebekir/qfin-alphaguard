"""Download the first 16 minutes of every S&P 500 member's session, 2016-2026.

For the check of the opening-gap reversal (gap_check.py): is its profit in a
price one can trade at? Each request asks Alpaca for the 1-minute bars of
all of a day's members (point in time, from the daily bars' sp500 column)
from the open to 15 minutes after it (09:30 to 09:45, 16 bars), from the
consolidated tape (SIP), as traded. One Parquet file per year, under OUT. A
year already written is skipped, so the download can be resumed.

Usage: uv run --env-file .env python opening_minutes.py PRICES OUT
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.alpaca import PAGE_SIZE, AlpacaClient, api_keys
from qfin_alphaguard.sessions import sessions

PRICES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MINUTES = 16
WORKERS = 4
PAUSE = 1.5  # seconds between one worker's requests: about 160 a minute in all

members = (
    pl.read_parquet(PRICES)
    .filter(pl.col("sp500"))
    .group_by("date")
    .agg(pl.col("ticker"))
)
by_day = {d: sorted(t) for d, t in members.iter_rows()}
client = AlpacaClient(*api_keys())


def one_day(s) -> list[tuple]:
    tickers = by_day.get(s.day)
    if not tickers:
        return []
    # Alpaca's codes for each daily ticker, as for the stored minutes: CHK_old
    # is CHK, BRK-B is BRK.B, and AABA's minutes before June 2017 are YHOO's.
    codes = {code: t for t in tickers for code in client.minute_codes(t, None)}
    params = {
        "symbols": ",".join(codes),
        "timeframe": "1Min",
        "start": s.open.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (s.open + timedelta(minutes=MINUTES - 1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feed": "sip",
        "adjustment": "raw",
        "sort": "asc",
        "limit": str(PAGE_SIZE),
    }
    found = []
    while True:
        page = client._get(params)
        for code, bars in (page.get("bars") or {}).items():
            for b in bars:
                found.append(
                    (
                        s.day,
                        codes.get(code, code),
                        b["t"],
                        b["o"],
                        b["h"],
                        b["l"],
                        b["c"],
                        b["v"],
                    )
                )
        time.sleep(PAUSE)
        if not page.get("next_page_token"):
            return found
        params = {**params, "page_token": page["next_page_token"]}


OUT.mkdir(parents=True, exist_ok=True)
for year in range(2016, 2027):
    path = OUT / f"{year}.parquet"
    if path.exists():
        continue
    with ThreadPoolExecutor(WORKERS) as pool:
        days = sessions(date(year, 1, 1), date(year, 12, 31))
        rows = [r for found in pool.map(one_day, days) for r in found]
    frame = (
        pl.DataFrame(
            rows,
            schema={
                "day": pl.Date,
                "ticker": pl.String,
                "t": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
            orient="row",
        )
        .with_columns(
            pl.col("t")
            .str.to_datetime(time_zone="UTC")
            .dt.cast_time_unit("us")
            .alias("start")
        )
        .drop("t")
    )
    frame.write_parquet(path)
    print(year, frame.height, "bars", frame["ticker"].n_unique(), "stocks", flush=True)
