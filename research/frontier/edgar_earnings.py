"""Earnings release times of S&P 500 members since 2015, from SEC EDGAR (free, public).

A company files an 8-K with Item 2.02 (Results of Operations and Financial Condition) when it
releases its earnings. EDGAR's submissions API lists each company's filings with their items
and acceptance time; this collects every 8-K with Item 2.02 since 2015 for the members in
data/raw/prices.parquet whose ticker EDGAR's ticker list knows (companies that left the market
under their ticker are missed: the coverage is printed).

The API's acceptanceDateTime cannot be trusted for the time of day: for some filings it is the
true UTC time, for others (Apple's, for one) it is shifted twice, ten hours late in winter and
eight in summer. So each filing's acceptance time is read from its own header
(ACCEPTANCE-DATETIME, New York time), and the run reports how often the API's time agreed.

SEC asks for at most 10 requests a second and a User-Agent with a contact, SEC_USER_AGENT in
.env. Responses are cached in OUT_DIR/submissions/.

Usage: edgar_earnings.py PRICES_PARQUET OUT_DIR   (with SEC_USER_AGENT set)
"""

import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

PRICES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
AGENT = os.environ["SEC_USER_AGENT"]
NEW_YORK = ZoneInfo("America/New_York")
FIRST = "2015-01-01"
_last_request = [0.0]
_lock = threading.Lock()


def get(url: str) -> bytes:
    """One request, at most 8 a second across threads, retried on errors."""
    for attempt in range(5):
        with _lock:  # space requests 0.125 s apart, whichever thread sends them
            wait = 0.125 - (time.monotonic() - _last_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as error:  # network errors of every kind
            if attempt == 4:
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError


def cached(name: str, url: str) -> dict:
    path = OUT / "submissions" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get(url))
    return json.loads(path.read_text())


def header_time(cik: int, accession: str) -> datetime | None:
    """The filing's acceptance time in UTC, from its header (New York time there).

    None if the header gives no time after three tries (a missing page, or SEC's page for
    too many requests).
    """
    for attempt in range(3):
        header = get(
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{accession}-index-headers.html"
        ).decode("latin-1")
        found = re.search(r"ACCEPTANCE-DATETIME>(\d{14})", header)
        if found:
            stamp = found.group(1)
            local = datetime(
                *(
                    int(stamp[i : i + n])
                    for i, n in ((0, 4), (4, 2), (6, 2), (8, 2), (10, 2), (12, 2))
                ),
                tzinfo=NEW_YORK,
            )
            return local.astimezone(UTC)
        time.sleep(10 * (attempt + 1))
    return None


def rows(block: dict, cik: int, ticker: str) -> list[dict]:
    found = []
    for i, form in enumerate(block["form"]):
        if form != "8-K" or "2.02" not in block["items"][i].split(","):
            continue
        if block["filingDate"][i] < FIRST:
            continue
        found.append(
            {
                "ticker": ticker,
                "cik": cik,
                "accession": block["accessionNumber"][i],
                "filing_date": block["filingDate"][i],
                "listed": block["acceptanceDateTime"][i],
            }
        )
    return found


tickers = (
    pl.read_parquet(PRICES, columns=["ticker", "sp500"])
    .filter(pl.col("sp500"))["ticker"]
    .unique()
    .sort()
    .to_list()
)
known = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
cik_of = {v["ticker"].upper(): int(v["cik_str"]) for v in known.values()}
mapped = {
    t: cik_of[t.replace(".", "-").upper()]
    for t in tickers
    if t.replace(".", "-").upper() in cik_of
}
print(f"{len(tickers)} members since 2015; {len(mapped)} found in EDGAR's ticker list")

events = []
for number, (ticker, cik) in enumerate(sorted(mapped.items()), 1):
    data = cached(
        f"CIK{cik:010d}.json", f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    )
    events += rows(data["filings"]["recent"], cik, ticker)
    for page in data["filings"].get("files", []):
        if page["filingTo"] < FIRST:
            continue
        older = cached(page["name"], f"https://data.sec.gov/submissions/{page['name']}")
        events += rows(older, cik, ticker)
    if number % 100 == 0:
        print(f"   {number} companies, {len(events)} earnings filings", flush=True)

frame = pl.DataFrame(events).unique(["ticker", "accession"]).sort("ticker", "listed")
print(
    f"{frame.height} earnings 8-Ks from {frame['filing_date'].min()} to {frame['filing_date'].max()}"
)

# Every filing's time from its header, cached so that a stopped run resumes.
times_file = OUT / "header_times.json"
times = json.loads(times_file.read_text()) if times_file.exists() else {}
todo = [r for r in frame.to_dicts() if r["accession"] not in times]
print(f"headers to read: {len(todo)} (cached {len(times)})", flush=True)
try:
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = pool.map(
            lambda r: (r["accession"], header_time(r["cik"], r["accession"])), todo
        )
        for number, (accession, accepted) in enumerate(found, 1):
            times[accession] = accepted.isoformat() if accepted else None
            if number % 2000 == 0:
                times_file.write_text(json.dumps(times))
                print(f"   {number} headers", flush=True)
finally:
    times_file.write_text(json.dumps(times))
missing = [a for a, t in times.items() if t is None]
print(f"{len(missing)} filings without a time in their header are left out")
frame = frame.filter(~pl.col("accession").is_in(missing))

frame = frame.with_columns(
    accepted=pl.col("accession")
    .replace_strict(times, return_dtype=pl.String)
    .str.to_datetime(time_zone="UTC"),
    listed_utc=pl.col("listed").str.to_datetime(time_zone="UTC"),
)
agree = (frame["accepted"] == frame["listed_utc"]).mean()
print(f"the API's time agreed with the header on {agree:.1%} of filings")
frame.drop("listed_utc").write_parquet(OUT / "earnings.parquet")
print(f"written {OUT / 'earnings.parquet'}")
