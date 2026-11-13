"""Each report's risk factors and its managers' discussion, and how much they changed (no
return is read).

The first look (first_look.py) read whole reports, and their word counts are mostly common
words: a 10-K's median cosine against last year's is 0.999. The study's strongest results are
in single sections. This finds two in each report's text:
- the risk factors: Item 1A in a 10-K, and in Part II of a 10-Q;
- the managers' discussion and analysis (MD&A): Item 7 in a 10-K, Item 2 of Part I in a 10-Q.

The text is one line, so a section runs from its heading to the next item's heading. The table
of contents and references in the text ("see Item 1A. Risk Factors") name the same items: a
heading is not quoted or pointed to, a start counts only if no other start comes before its
end, and the longest such stretch is the section.

A report's sections are then set against those of the same report a year before, as change.py
sets whole reports: cosine and Jaccard, on the two sections pooled and on each.

Output: data/filings/sections.parquet: change.py's columns, scored on the two sections pooled,
with risk_words, mdna_words and each section's cosine. It also writes
results_sections.txt beside this file: how often each section is found, and how long it is.

Usage: uv run python research/filings/sections.py
"""

import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import change

OUT = change.OUT
RESULTS = Path(__file__).parent / "results_sections.txt"

_ = r"\s*[.:\-–—]?\s*"  # what may stand between an item's number and its title
# A heading, not a reference to one such as “Item 1A. Risk Factors” or "see Item 8 ...":
# not after a quote or a word that points to it, nor before a closing quote.
HEADING = (
    r"(?<![“\"'‘])(?<!\bsee )(?<!\bin )(?<!\bunder )(?<!\bto )(?<!\bof )(?<!\band )"
    r"(?<!, )(?<!\bwith )(?<!\bwithin )"
)
ITEM = rf"{HEADING}ite\s?ms?\s*"  # some text splits "Ite m"
RISK = re.compile(rf"{ITEM}1a{_}risk\s+factors(?!\s*[”\"’'])", re.IGNORECASE)
# Exelon heads it "Combined Management's Discussion".
MDNA = r"(?:combined\s+)?management\W{0,3}s\s+discussion"
SECTIONS = {
    "10-K": {
        "risk": (
            RISK,
            re.compile(
                rf"{ITEM}(?:1b{_}unresolved|1c{_}cyber|2{_}propert)", re.IGNORECASE
            ),
        ),
        "mdna": (
            # Freeport-McMoRan heads it "Items 7. and 7A."
            re.compile(rf"{ITEM}7(?:\W{{0,3}}and\s*7a)?{_}{MDNA}", re.IGNORECASE),
            re.compile(
                rf"{ITEM}(?:7a{_}quantitative|8{_}(?:consolidated\s+)?financial)",
                re.IGNORECASE,
            ),
        ),
    },
    "10-Q": {
        "risk": (
            RISK,
            re.compile(
                rf"{ITEM}(?:2{_}(?:unregistered|issuer|purchases|changes|sales|share"
                rf"|repurchase)|3{_}defaults|4{_}(?:mine|submission)|5{_}other"
                rf"|6{_}exhibits)",
                re.IGNORECASE,
            ),
        ),
        "mdna": (
            re.compile(rf"{ITEM}2{_}{MDNA}", re.IGNORECASE),
            # Some, such as Capital One, put the financial statements after the MD&A.
            re.compile(
                rf"{ITEM}(?:3{_}quantitative|4{_}controls|1{_}financial)", re.IGNORECASE
            ),
        ),
    },
}


def section(text: str, start: re.Pattern, end: re.Pattern) -> str:
    """A section: from its heading to the next item's, "" if there is none.

    Each start runs to the first end after it. The table of contents names the items too,
    one right after another, and a reference may come before the heading; so a start counts
    only if it is the last before its end, and the longest stretch is the section.
    """
    starts = [m.start() for m in start.finditer(text)]
    ends = [m.start() for m in end.finditer(text)]
    best = (0, 0)
    for i, s in enumerate(starts):
        after = [e for e in ends if e > s]
        if not after:
            continue
        stop = after[0]
        if i + 1 < len(starts) and starts[i + 1] < stop:
            continue  # a later start comes first: this one is a reference
        if stop - s > best[1] - best[0]:
            best = (s, stop)
    return text[best[0] : best[1]]


def sections(text: str, form: str) -> dict[str, str]:
    return {name: section(text, *found) for name, found in SECTIONS[form].items()}


def scores(rows: list[dict], root: Path = change.TEXT) -> list[dict]:
    """One chain's reports, each with its sections' scores against the report a year before."""
    counts: dict[str, dict[str, Counter]] = {}

    def of(row: dict) -> dict[str, Counter] | None:
        if row["accession"] not in counts:
            text = change.read(row, root)
            if text is None:
                return None
            found = sections(text, row["form"])
            counts[row["accession"]] = {k: change.words(v) for k, v in found.items()}
        return counts[row["accession"]]

    out = []
    for row in sorted(rows, key=lambda r: (r["filed"], r["accession"])):
        other = change.earlier_report(row, rows)
        a = of(row)
        if a is None:
            continue
        found = {f"{k}_words": sum(v.values()) for k, v in a.items()}
        if other is None or (b := of(other)) is None:
            out.append(row | found)
            continue
        pooled_a, pooled_b = a["risk"] + a["mdna"], b["risk"] + b["mdna"]
        out.append(
            row
            | found
            | {
                "earlier": other["accession"],
                "cosine": change.cosine(pooled_a, pooled_b),
                "jaccard": change.jaccard(pooled_a, pooled_b),
                "words": sum(pooled_a.values()),
                "earlier_words": sum(pooled_b.values()),
                "earlier_risk_words": sum(b["risk"].values()),
                "earlier_mdna_words": sum(b["mdna"].values()),
                "risk_cosine": change.cosine(a["risk"], b["risk"]),
                "mdna_cosine": change.cosine(a["mdna"], b["mdna"]),
            }
        )
    return out


def coverage(table: pl.DataFrame) -> str:
    """How often each section is found, and its median length, by year and form."""
    summary = (
        table.group_by(pl.col("filed").dt.year().alias("year"), "form")
        .agg(
            pl.len().alias("reports"),
            (pl.col("risk_words") >= 100).mean().round(3).alias("risk_found"),
            (pl.col("mdna_words") >= 100).mean().round(3).alias("mdna_found"),
            pl.col("risk_words").median().alias("risk_median"),
            pl.col("mdna_words").median().alias("mdna_median"),
            pl.col("cosine").is_not_null().mean().round(3).alias("scored"),
            pl.col("cosine").median().round(4).alias("median_cosine"),
        )
        .sort("year", "form")
    )
    with pl.Config(
        tbl_rows=40, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True
    ):
        return str(summary)


def main() -> None:
    members = pl.read_parquet(OUT / "members.parquet")
    filings = pl.read_parquet(OUT / "filings.parquet")
    by_cik = filings.partition_by("cik", as_dict=True)
    jobs = [
        [r for cik in chain for r in by_cik.get((cik,), pl.DataFrame()).to_dicts()]
        for chain in change.chains(members)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=change.WORKERS) as pool:
        for number, done in enumerate(pool.map(scores, jobs, chunksize=8), 1):
            rows += done
            if number % 100 == 0:
                print(f"{number} of {len(jobs)} companies", flush=True)
    table = pl.DataFrame(rows, infer_schema_length=None).drop("document")
    table = table.sort("cik", "filed", "accession")
    table.write_parquet(OUT / "sections.parquet")
    text = (
        f"sections.py: {table.height:,} reports read, "
        f"{table['cosine'].is_not_null().sum():,} with a report a year before.\n"
        "A section counts as found with 100 words or more.\n\n" + coverage(table) + "\n"
    )
    RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
