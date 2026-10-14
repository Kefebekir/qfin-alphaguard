"""The opening-gap reversal at the prices a small account would really pay.

gap_check.py buys at the first trade after a moment. A market order pays
more: it buys at the ask and sells at the bid. Here, for every session since
2016, the stocks the reversal would trade are picked from the daily bars as
in gap_check.py (the k largest falls and rises from yesterday's official
close to today's official open, among all members and among the 100 most
traded), and the consolidated tape's quotes (NBBO) of those stocks are
fetched from Alpaca at 09:31, 09:35 and 15:59. The first valid quote after
each moment is kept. Written under the folder OUT as one file a year, one
row per day and stock; a year already written is skipped, so the download
can be resumed.

With --picks, the stocks come from a Parquet file of day and ticker instead,
such as the top of morning_ml.py's scores.

Usage: uv run --env-file .env python gap_quotes.py PRICES OUT
       [--picks FILE] [--days N] [--workers N]
--days keeps only the first N sessions, written to OUT/trial.parquet.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.universe import dollar_volume
from qfin_alphaguard.sessions import sessions

parser = argparse.ArgumentParser()
parser.add_argument("prices", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--picks", type=Path)
parser.add_argument("--days", type=int)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
PRICES, OUT, LIMIT = args.prices, args.out, args.days
K = 10
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"
# Minutes after the open, or before the close if negative.
MOMENTS = {"q0931": 1, "q0935": 5, "q1559": -1}
# Seconds to look for a quote; the longer only for stocks without one in the
# shorter.
WINDOWS = (2, 30)
WORKERS, PAUSE = args.workers, 1.5
key_id, secret = api_keys()
headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
CODES = AlpacaClient(key_id, secret)  # only its ticker map is used

daily = (
    pl.read_parquet(PRICES)
    .sort("ticker", "date")
    .with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("ticker") - 1).alias("gap"),
        dollar_volume().shift(1).rolling_mean(20).over("ticker").alias("dollar_volume"),
    )
    .filter(pl.col("sp500") & (pl.col("volume") > 0) & pl.col("gap").is_not_null())
    .with_columns(
        (pl.col("dollar_volume").rank(descending=True).over("date") <= 100).alias(
            "top100"
        )
    )
)


def extremes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        (pl.col("gap").rank("ordinal").over("date") <= K)
        | (pl.col("gap").rank("ordinal", descending=True).over("date") <= K)
    )


if args.picks:
    picks = pl.read_parquet(args.picks).select(pl.col("day").alias("date"), "ticker")
else:
    picks = (
        pl.concat([extremes(daily), extremes(daily.filter("top100"))])
        .select("date", "ticker")
        .unique()
    )
by_day = {d: sorted(t) for d, t in picks.group_by("date").agg("ticker").iter_rows()}


def first_quotes(symbols: list[str], start, seconds: int) -> dict[str, dict]:
    """The first quote with a positive bid below a positive ask, per symbol."""
    params = {
        "symbols": ",".join(symbols),
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (start + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feed": "sip",
        "limit": "10000",
        "sort": "asc",
    }
    found: dict[str, dict] = {}
    while True:
        request = urllib.request.Request(
            f"{QUOTES_URL}?{urllib.parse.urlencode(params)}", headers=headers
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    page = json.load(response)
                break
            except OSError:
                time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"quotes for {start} failed")
        for symbol, quotes in (page.get("quotes") or {}).items():
            if symbol in found:
                continue
            for q in quotes:
                if 0 < q["bp"] < q["ap"]:
                    found[symbol] = q
                    break
        time.sleep(PAUSE)
        token = page.get("next_page_token")
        if not token or len(found) == len(symbols):
            return found
        params = {**params, "page_token": token}


def one_day(s) -> list[dict]:
    tickers = by_day.get(s.day)
    if not tickers:
        return []
    # Alpaca's codes, as for the stored minutes (AABA was YHOO until 2017).
    codes = {code: t for t in tickers for code in CODES.minute_codes(t, None)}
    rows = {t: {"day": s.day, "ticker": t} for t in tickers}
    for name, minutes in MOMENTS.items():
        moment = (
            s.open + timedelta(minutes=minutes)
            if minutes >= 0
            else s.close + timedelta(minutes=minutes)
        )
        missing = list(codes)
        for seconds in WINDOWS:
            got = first_quotes(missing, moment, seconds)
            for code, q in got.items():
                rows[codes[code]][f"{name}_bid"] = q["bp"]
                rows[codes[code]][f"{name}_ask"] = q["ap"]
            missing = [c for c in missing if c not in got]
            if not missing:
                break
    return list(rows.values())


SCHEMA = {"day": pl.Date, "ticker": pl.String} | {
    f"{m}_{side}": pl.Float64 for m in MOMENTS for side in ("bid", "ask")
}


def fetch(days: list, path: Path) -> None:
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = [r for found in pool.map(one_day, days) for r in found]
    frame = pl.DataFrame(rows, schema=SCHEMA)
    frame.write_parquet(path)
    missing = {c: frame[c].null_count() for c in frame.columns if c.endswith("_bid")}
    print(path.name, frame.height, "stock-days; without a quote:", missing, flush=True)


OUT.mkdir(parents=True, exist_ok=True)
sessions_with_picks = [
    s for s in sessions(date(2016, 1, 1), date(2026, 12, 31)) if s.day in by_day
]
if LIMIT:
    fetch(sessions_with_picks[:LIMIT], OUT / "trial.parquet")
else:
    for year in range(2016, 2027):
        path = OUT / f"{year}.parquet"
        if not path.exists():
            fetch([s for s in sessions_with_picks if s.day.year == year], path)
