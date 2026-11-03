"""Which company filed for each S&P 500 member: its SEC number (CIK) over its own years.

Before a study of what companies file (research/filings/README.md). EDGAR's ticker list knows
only today's tickers, so the members that have left, or changed their ticker, are not in it:
172 of the 768 tickers since 2015, 22% of 2016's member-days. A study without them would hold
only the companies that lasted.

SEC's Financial Statement Data Sets list each filing with the file name of its XBRL instance,
which usually begins with the ticker the company then used (yhoo-20151231.xml). A member's
company is decided, first rule first:

- BY_HAND: the few cases read and checked against SEC's own names;
- a ticker still a member on the last day of the prices: today's list, since that company is
  the one trading under it. Others may use the same file names (CPT's are also those of Copper
  Property CTL Pass Through Trust);
- otherwise the company with the most 10-K and 10-Q filings named after the ticker, filed from
  a year before the member joined to 120 days after it left. A ticker that has since passed to
  another company (BBT, now Beacon Financial) stays with the company of its years;
- otherwise today's list.

ALSO lists the other registrants of one business, whose reports continue the chain: a new
holding company files under a new number (Disney in 2019, Exxon in 2026). Subsidiaries that
file for themselves (Allstate Life, MPLX) are not in it.

Downloaded: 2014q4 and each first quarter of 2015-2026, enough since every company files at
least yearly. Each set's zip is deleted once the columns used are kept. Nothing here reads a
return.

Output: data/filings/members.parquet (ticker, first, last, cik, name, how, also), one row a
member ticker; the coverage each year, every disagreement and what is left without a company, printed.

Usage: uv run python research/filings/members.py   (with SEC_USER_AGENT set; it reads .env's)
"""

import json
import os
import re
import time
import urllib.request
import zipfile
from datetime import timedelta
from pathlib import Path

import polars as pl

OUT = Path("data/filings")
SETS = OUT / "sets"
PRICES = Path("data/raw/prices.parquet")
SET_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{}.zip"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
QUARTERS = ("2014q4", *(f"{year}q1" for year in range(2015, 2027)))
COLUMNS = ["adsh", "cik", "name", "form", "filed", "instance"]
BEFORE, AFTER = timedelta(days=400), timedelta(days=120)
FIRST_MEMBER_DAY = "2015-01-01"
SPACING = 0.125  # seconds between requests: SEC asks for at most 10 a second
# Checked against the names in SEC's data sets, on 3 November 2026.
BY_HAND = {
    "AABA": 1011006,  # Yahoo, renamed Altaba: its files say yhoo
    "CMCSK": 1166691,  # Comcast's other class of shares
    "CTRA": 858470,  # Cabot Oil & Gas, renamed Coterra: cog
    "DISCK": 1437107,  # Discovery's other class of shares
    "JEF": 96223,  # Jefferies Financial, once Leucadia; Jefferies Group is its subsidiary
    "NLSN": 1492633,  # Nielsen: nlsnnv
    "SATS": 1415404,  # EchoStar, whose ticker became ECHO; today's SATS is Strive
    "SNI": 1430602,  # Scripps Networks; Summit Networks uses its file names
    "XL_old": 875159,  # XL Group: xlgroup
}
ALSO = {
    "APA": (6769,),  # Apache, before APA Corp in 2021
    "AVGO": (1441634, 1649338),  # Avago, then Broadcom Ltd, before Broadcom Inc.
    "BLK": (1364742,),  # BlackRock's new holding company, 2024
    "CI": (701221,),  # Cigna's new holding company, 2018
    "DD": (30554,),  # E. I. du Pont, before DowDuPont
    "DIS": (1001039,),  # Disney's new holding company, 2019
    "FTI": (1135152,),  # FMC Technologies, before TechnipFMC
    "GOOG": (1288776,),  # Google, before Alphabet
    "GOOGL": (1288776,),
    "JCI": (53669,),  # Johnson Controls, before it merged into Tyco
    "LLL_old": (1039101,),  # L-3 Communications Holdings, then L3 Technologies
    "WRK": (1636023,),  # WestRock before its 2018 merger with KapStone
    "XOM": (34088,),  # Exxon's new holding company, 2026
}


def get(url: str, path: Path | None = None) -> bytes | None:
    """One request, spaced and retried; into `path` when given."""
    agent = os.environ["SEC_USER_AGENT"]
    for attempt in range(5):
        time.sleep(SPACING)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": agent})
            with urllib.request.urlopen(request, timeout=120) as response:
                if path is None:
                    return response.read()
                with path.open("wb") as file:
                    while chunk := response.read(1 << 20):
                        file.write(chunk)
                return None
        except OSError as error:
            if attempt == 4:
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError


def named_after(instance: str | None) -> str | None:
    """The ticker a filing's XBRL instance is named after: aapl-20150926.xml gives aapl."""
    found = re.match(r"([A-Za-z0-9.\-]+?)-\d{6,8}", instance or "")
    return plain(found.group(1)) if found else None


def plain(ticker: str) -> str:
    """A ticker without our _old mark, its punctuation or its case: BRK.B gives brkb."""
    return re.sub(r"[.\-/]", "", re.sub(r"_old\d*$", "", ticker)).lower()


def members(prices: pl.DataFrame) -> pl.DataFrame:
    """(ticker, first, last): each member ticker's first and last day as a member."""
    return (
        prices.filter(pl.col("sp500"))
        .group_by("ticker")
        .agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
        .sort("ticker")
    )


def filings_named_after(filings: pl.DataFrame) -> pl.DataFrame:
    """10-K and 10-Q filings with the ticker their instance is named after."""
    return (
        filings.filter(
            pl.col("form").is_in(["10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT"])
        )
        .with_columns(
            pl.col("instance")
            .map_elements(named_after, return_dtype=pl.String)
            .alias("named"),
            pl.col("filed").cast(pl.String).str.to_date("%Y%m%d").alias("filed"),
            pl.col("cik").cast(pl.Int64),
        )
        .drop_nulls("named")
    )


def company_in_its_years(row: dict, filings: pl.DataFrame) -> tuple[int, str] | None:
    """The company with the most filings named after the ticker around its member years."""
    found = filings.filter(
        (pl.col("named") == plain(row["ticker"]))
        & pl.col("filed").is_between(row["first"] - BEFORE, row["last"] + AFTER)
    )
    if found.is_empty():
        return None
    best = (
        found.group_by("cik")
        .agg(
            pl.len().alias("n"), pl.col("name").last(), pl.col("filed").max().alias("t")
        )
        .sort(["n", "t", "cik"], descending=[True, True, False])
        .row(0, named=True)
    )
    return best["cik"], best["name"]


def match(
    rows: pl.DataFrame, filings: pl.DataFrame, today: dict[str, tuple[int, str]]
) -> pl.DataFrame:
    """Each member ticker's company, by the module's rules, first rule first."""
    latest = rows["last"].max()
    names = dict(
        filings.sort("filed")
        .select("cik", "name")
        .unique("cik", keep="last")
        .iter_rows()
    )
    out = []
    for row in rows.iter_rows(named=True):
        then = company_in_its_years(row, filings)
        now = None if "_old" in row["ticker"] else today.get(plain(row["ticker"]))
        if row["ticker"] in BY_HAND:
            cik = BY_HAND[row["ticker"]]
            name, how = names.get(cik), "by hand"
        elif now and (row["last"] == latest or not then):
            cik, name, how = *now, "today's list"
        elif then:
            cik, name = then
            how = (
                "filings, against today's list" if now and now[0] != cik else "filings"
            )
        else:
            cik, name, how = None, None, "none"
        also = list(ALSO.get(row["ticker"], ()))
        out.append(row | {"cik": cik, "name": name, "how": how, "also": also})
    return pl.DataFrame(
        out,
        schema={
            "ticker": pl.String,
            "first": pl.Date,
            "last": pl.Date,
            "cik": pl.Int64,
            "name": pl.String,
            "how": pl.String,
            "also": pl.List(pl.Int64),
        },
    )


def load_set(quarter: str) -> pl.DataFrame:
    """A quarter's filings, keeping only COLUMNS; its zip is fetched once and deleted."""
    kept = SETS / f"{quarter}.parquet"
    if not kept.exists():
        SETS.mkdir(parents=True, exist_ok=True)
        archive = SETS / f"{quarter}.zip"
        get(SET_URL.format(quarter), archive)
        with zipfile.ZipFile(archive) as z, z.open("sub.txt") as sub:
            frame = pl.read_csv(
                sub.read(),
                separator="\t",
                quote_char=None,
                encoding="utf8-lossy",
                infer_schema_length=0,
                columns=COLUMNS,
            )
        frame.write_parquet(kept)
        archive.unlink()
        print(f"{quarter}: {frame.height:,} filings", flush=True)
    return pl.read_parquet(kept)


def main() -> None:
    filings = filings_named_after(pl.concat([load_set(q) for q in QUARTERS]))
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "sp500"]).filter(
        pl.col("date") >= pl.lit(FIRST_MEMBER_DAY).str.to_date()
    )
    listed = json.loads(get(TICKERS_URL))
    today = {
        plain(v["ticker"]): (int(v["cik_str"]), v["title"]) for v in listed.values()
    }
    table = match(members(prices), filings, today)
    table.write_parquet(OUT / "members.parquet")

    print(f"\n{table.height} member tickers since 2015:")
    print(table.group_by("how").len().sort("len", descending=True))
    by_year = (
        prices.filter(pl.col("sp500"))
        .join(table.select("ticker", "cik"), on="ticker", how="left")
        .group_by(pl.col("date").dt.year().alias("year"))
        .agg(
            pl.col("cik")
            .is_not_null()
            .mean()
            .round(4)
            .alias("member_days_with_a_company")
        )
        .sort("year")
    )
    print(by_year)
    against = table.filter(pl.col("how") == "filings, against today's list")
    print(
        f"\nLeft the index, and today's list names another company, {against.height}:"
    )
    for row in against.iter_rows(named=True):
        print(
            f"  {row['ticker']} {row['first']} to {row['last']}: {row['name']} ({row['cik']})"
        )
    left = table.filter(pl.col("how") == "none")
    print(f"\nWithout a company, {left.height}:")
    for row in left.iter_rows(named=True):
        print(f"  {row['ticker']} {row['first']} to {row['last']}")


if __name__ == "__main__":
    main()
