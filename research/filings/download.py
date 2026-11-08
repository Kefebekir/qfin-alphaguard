"""Every member's 10-K and 10-Q as plain text, from SEC's EDGAR (free; no return is read).

For each company in members.parquet, and the older registrants it lists, EDGAR's list of filings
gives its 10-Ks and 10-Qs filed from 400 days before the member joined to 120 days after it left,
and from 2015 on: a report is compared later with the same report a year before. Amendments are
left out, as they are partial. A filing made for two registrants at once is kept once.

Each filing's main document is read once and kept as text only: its tags, scripts, styles and
hidden XBRL header removed. It stops and resumes: what is on disk is not asked for again.

Output, under data/filings/:
- filings.parquet: cik, accession, form, filed, report, document, one row a filing;
- text/<cik>/<accession>.txt.gz: its text;
- failed.json: those that could not be read, with why.

Usage: uv run python research/filings/download.py   (with SEC_USER_AGENT set; it reads .env's)
"""

import gzip
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl

OUT = Path("data/filings")
LISTS = OUT / "submissions"
TEXT = OUT / "text"
FORMS = ("10-K", "10-Q")
FIRST = date(2015, 1, 1)
BEFORE, AFTER = timedelta(days=400), timedelta(days=120)
SPACING = 0.125  # seconds between requests, across threads: at most 8 a second
THREADS = 4
FIELDS = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
_gate = threading.Lock()
_last = [0.0]


def get(url: str, tries: int = 5) -> bytes:
    """One request, spaced across threads, retried; the body, unpacked if it came packed."""
    headers = {"User-Agent": os.environ["SEC_USER_AGENT"], "Accept-Encoding": "gzip"}
    for attempt in range(tries):
        with _gate:
            wait = SPACING - (time.monotonic() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as error:
            if error.code == 404 or attempt == tries - 1:
                raise
            time.sleep(10 * 2**attempt)  # 403 and 429: SEC asks to slow down
        except (OSError, EOFError) as error:
            if attempt == tries - 1:
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError


def to_text(page: bytes) -> str:
    """A filing's document as plain text: no tags, scripts, styles or hidden XBRL header."""
    text = page.decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style|ix:header|head)\b.*?</\1\s*>", " ", text)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def windows(members: pl.DataFrame) -> dict[int, list[tuple[date, date]]]:
    """Each company's filing windows: its member tickers' years, and its older registrants'."""
    out: dict[int, list[tuple[date, date]]] = {}
    for row in members.filter(pl.col("cik").is_not_null()).iter_rows(named=True):
        span = (row["first"] - BEFORE, row["last"] + AFTER)
        for cik in [row["cik"], *row["also"]]:
            out.setdefault(cik, []).append(span)
    return out


def filings_in(block: dict, cik: int, spans: list[tuple[date, date]]) -> list[dict]:
    """A block of EDGAR's list, as rows: the 10-Ks and 10-Qs filed in the company's windows."""
    out = []
    for accession, form, filed, report, document in zip(
        *(block[field] for field in FIELDS), strict=True
    ):
        day = date.fromisoformat(filed)
        if (
            form in FORMS
            and day >= FIRST
            and document
            and any(a <= day <= b for a, b in spans)
        ):
            out.append(
                {
                    "cik": cik,
                    "accession": accession,
                    "form": form,
                    "filed": day,
                    "report": report or None,
                    "document": document,
                }
            )
    return out


def listing(cik: int) -> list[dict]:
    """EDGAR's list of a company's filings: its recent block and its older pages from 2015."""

    def fetched(name: str) -> dict:
        path = LISTS / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(get(f"https://data.sec.gov/submissions/{name}"))
        return json.loads(path.read_text())

    top = fetched(f"CIK{cik:010d}.json")
    blocks = [top["filings"]["recent"]]
    for page in top["filings"].get("files", []):
        if page["filingTo"] >= FIRST.isoformat():
            blocks.append(fetched(page["name"]))
    return blocks


def url_of(row: dict) -> str:
    plain = row["accession"].replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{plain}/{row['document']}"


def path_of(row: dict, root: Path = TEXT) -> Path:
    return root / str(row["cik"]) / f"{row['accession']}.txt.gz"


def keep(row: dict, root: Path = TEXT) -> str | None:
    """A filing's text on disk, unless already there; why not, if it could not be read."""
    path = path_of(row, root)
    if path.exists():
        return None
    try:
        text = to_text(get(url_of(row)))
    except (urllib.error.HTTPError, RuntimeError) as error:
        return str(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(gzip.compress(text.encode("utf-8")))
    partial.rename(path)
    return None


def main() -> None:
    members = pl.read_parquet(OUT / "members.parquet")
    spans = windows(members)
    rows = []
    for number, (cik, company) in enumerate(sorted(spans.items()), 1):
        for block in listing(cik):
            rows += filings_in(block, cik, company)
        if number % 100 == 0:
            print(
                f"{number} of {len(spans)} companies listed, {len(rows):,} filings",
                flush=True,
            )
    table = (
        pl.DataFrame(rows)
        .unique("accession", keep="first", maintain_order=True)
        .sort("cik", "filed", "accession")
    )
    table.write_parquet(OUT / "filings.parquet")
    print(table.group_by("form").len().sort("form"))
    todo = [r for r in table.to_dicts() if not path_of(r).exists()]
    print(f"{table.height:,} filings, {len(todo):,} to read", flush=True)
    failed = {}
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for number, (row, why) in enumerate(
            zip(todo, pool.map(keep, todo), strict=True), 1
        ):
            if why:
                failed[row["accession"]] = why
            if number % 1000 == 0:
                rate = number / (time.monotonic() - began)
                print(
                    f"{number:,} read, {len(failed)} failed, {rate:.1f} a second",
                    flush=True,
                )
    (OUT / "failed.json").write_text(json.dumps(failed, indent=1))
    print(
        f"done: {table.height - len(failed):,} kept, {len(failed)} failed", flush=True
    )


if __name__ == "__main__":
    main()
