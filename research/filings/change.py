"""How much each report changed from the same report a year before (no return is read).

A 10-K is set against the company's 10-K of a year before, a 10-Q against the 10-Q of the same
quarter a year before, as in "Lazy Prices" (Cohen, Malloy and Nguyen, 2020). A company's chain
holds its own filings and those of the older registrants it lists (members.py's `also`). The
earlier report is the one of the same form whose period ended 320 to 410 days before, the
closest to 365. A report with none has no score.

Two measures, on each text's lower-case words of two letters or more, numbers left out:
- cosine: the cosine of the two word counts, which the study's own results lean on;
- jaccard: the words they share, over all the words either uses.
1 is no change; the lower, the more changed.

Output: data/filings/change.parquet: cik, accession, form, filed, report, earlier, cosine,
jaccard, words, earlier_words, one row a report with a score.

Usage: uv run python research/filings/change.py
"""

import gzip
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl

OUT = Path("data/filings")
TEXT = OUT / "text"
NEAREST, FARTHEST = timedelta(days=320), timedelta(days=410)
YEAR = timedelta(days=365)
WORD = re.compile(r"[a-z]{2,}")
WORKERS = 6


def words(text: str) -> Counter:
    return Counter(WORD.findall(text.lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    small, large = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(n * large[w] for w, n in small.items() if w in large)
    return dot / math.sqrt(
        sum(n * n for n in a.values()) * sum(n * n for n in b.values())
    )


def jaccard(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    shared = len(a.keys() & b.keys())
    return shared / (len(a) + len(b) - shared)


def chains(members: pl.DataFrame) -> list[tuple[int, ...]]:
    """Each business's registrants, the main one first; a company once, in its longest chain."""
    found: dict[int, tuple[int, ...]] = {}
    for row in members.filter(pl.col("cik").is_not_null()).iter_rows(named=True):
        chain = (row["cik"], *row["also"])
        if len(chain) >= len(found.get(row["cik"], ())):
            found[row["cik"]] = chain
    inside = {cik for chain in found.values() for cik in chain[1:]}
    return [chain for main, chain in sorted(found.items()) if main not in inside]


def earlier_report(row: dict, rows: list[dict]) -> dict | None:
    """The same form's report whose period ended 320 to 410 days before, closest to a year."""
    if row["report"] is None:
        return None
    end = date.fromisoformat(row["report"])
    best, gap = None, None
    for other in rows:
        if other["form"] != row["form"] or other["report"] is None:
            continue
        before = end - date.fromisoformat(other["report"])
        if NEAREST <= before <= FARTHEST:
            off = abs(before - YEAR)
            if gap is None or off < gap:
                best, gap = other, off
    return best


def read(row: dict, root: Path = TEXT) -> str | None:
    path = root / str(row["cik"]) / f"{row['accession']}.txt.gz"
    return gzip.decompress(path.read_bytes()).decode("utf-8") if path.exists() else None


def scores(rows: list[dict], root: Path = TEXT) -> list[dict]:
    """One chain's reports with their scores against the report a year before."""
    counts: dict[str, Counter] = {}

    def of(row: dict) -> Counter | None:
        if row["accession"] not in counts:
            text = read(row, root)
            if text is None:
                return None
            counts[row["accession"]] = words(text)
        return counts[row["accession"]]

    out = []
    for row in sorted(rows, key=lambda r: (r["filed"], r["accession"])):
        other = earlier_report(row, rows)
        if other is None:
            continue
        a, b = of(row), of(other)
        if not a or not b:
            continue
        out.append(
            row
            | {
                "earlier": other["accession"],
                "cosine": cosine(a, b),
                "jaccard": jaccard(a, b),
                "words": sum(a.values()),
                "earlier_words": sum(b.values()),
            }
        )
    return out


def main() -> None:
    members = pl.read_parquet(OUT / "members.parquet")
    filings = pl.read_parquet(OUT / "filings.parquet")
    by_cik = filings.partition_by("cik", as_dict=True)
    jobs = [
        [r for cik in chain for r in by_cik.get((cik,), pl.DataFrame()).to_dicts()]
        for chain in chains(members)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for number, done in enumerate(pool.map(scores, jobs, chunksize=8), 1):
            rows += done
            if number % 100 == 0:
                print(
                    f"{number} of {len(jobs)} companies, {len(rows):,} reports",
                    flush=True,
                )
    table = pl.DataFrame(rows).drop("document").sort("cik", "filed", "accession")
    table.write_parquet(OUT / "change.parquet")
    print(f"{table.height:,} reports with a score, of {filings.height:,} filings")
    print(
        table.group_by(pl.col("filed").dt.year().alias("year"), "form")
        .agg(
            pl.len().alias("n"),
            pl.col("cosine").median().round(3).alias("median_cosine"),
            pl.col("words").median().alias("median_words"),
        )
        .sort("year", "form")
    )


if __name__ == "__main__":
    main()
