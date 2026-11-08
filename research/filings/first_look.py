"""A first look at "lazy prices" among S&P 500 members, on 2016-2021 only. No model.

Does a member whose report changed little from a year before do better, over the next month,
than one whose report changed much? 2022-2026 stay unread: no price and no report after
LAST_SEEN reaches anything here, so a rule found on these years can still be tested there, once.

THE LOOK, fixed in this file before it was first run:
- At each month's last session t, the members that day, each with its latest report filed in
  the three months before t, filed before t. A company with two share classes counts once. A
  report of under MIN_WORDS words, or set against one, is left out: such a document usually
  leaves its contents to an exhibit.
- Its score is the report's cosine (change.py), ranked within its form among that day's reports,
  since a 10-Q and a 10-K do not change alike.
- Long the fifth that changed least, short the fifth that changed most, equal weights, from t's
  close to the next month's last close. A stock that stops trading is held at its last close.
- Costs as the news tests': 1 bp a round trip on each side, on the part of the side that changed.
- THE NUMBER: the mean a month after costs and its t, over 2016-2021.

Reported beside it, not to be chosen among: each year; each side against the mean of the members
ranked; before costs; each form; Jaccard for cosine; holding three and six months (overlapping,
as a share of each month's positions).

These years' results decide only whether a test on 2022-2026 is worth writing down.

Usage: uv run python research/filings/first_look.py
"""

from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("data/filings")
PRICES = Path("data/raw/prices.parquet")
RESULTS = Path(__file__).parent / "results_first_look.txt"
LAST_SEEN = date(2021, 12, 31)  # nothing after it is read
FIRST_MONTH = date(2016, 1, 1)
RECENT = timedelta(days=92)
MIN_WORDS = 2_000
MIN_REPORTS = 5  # of one form on one day, to rank them
FIFTH = 0.2
COST = 0.0001  # a round trip, each side


def seen(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Only what is dated by LAST_SEEN: the one door through which any data comes in."""
    return frame.filter(pl.col(column) <= LAST_SEEN)


def month_ends(days: list[date]) -> list[date]:
    """Each month's last session from FIRST_MONTH on, of the sorted sessions given."""
    last: dict[tuple[int, int], date] = {}
    for day in days:
        last[(day.year, day.month)] = day
    return [d for d in sorted(last.values()) if d >= FIRST_MONTH]


def chain_of(members: pl.DataFrame) -> dict[str, tuple[int, ...]]:
    """Each member ticker's registrants, its main one first."""
    return {
        r["ticker"]: (r["cik"], *r["also"])
        for r in members.filter(pl.col("cik").is_not_null()).iter_rows(named=True)
    }


def reports_by_cik(change: pl.DataFrame) -> dict[int, list[dict]]:
    """Each registrant's scored reports of enough words, by filing date."""
    usable = change.filter(
        (pl.col("words") >= MIN_WORDS) & (pl.col("earlier_words") >= MIN_WORDS)
    ).sort("filed")
    out: dict[int, list[dict]] = {}
    for row in usable.iter_rows(named=True):
        out.setdefault(row["cik"], []).append(row)
    return out


def formation(
    t: date,
    tickers: list[str],
    chains: dict[str, tuple[int, ...]],
    reports: dict[int, list[dict]],
    measure: str = "cosine",
) -> pl.DataFrame:
    """Day t's members with a recent report, ranked within form: (ticker, form, rank), where a
    rank of 1 is the least changed. A company counts once, under its first ticker."""
    rows, taken = [], set()
    for ticker in sorted(tickers):
        chain = chains.get(ticker)
        if chain is None or chain[0] in taken:
            continue
        recent = [
            r
            for cik in chain
            for r in reports.get(cik, [])
            if t - RECENT <= r["filed"] < t
        ]
        if not recent:
            continue
        latest = max(recent, key=lambda r: (r["filed"], r["accession"]))
        rows.append(
            {"ticker": ticker, "form": latest["form"], "score": latest[measure]}
        )
        taken.add(chain[0])
    schema = {"ticker": pl.String, "form": pl.String, "score": pl.Float64}
    return (
        pl.DataFrame(rows, schema=schema)
        .filter(pl.len().over("form") >= MIN_REPORTS)
        .with_columns(
            rank=pl.col("score").rank("average").over("form") / pl.len().over("form")
        )
    )


def held_return(closes: pl.DataFrame, t: date, until: date) -> dict[str, float]:
    """Each ticker's return from t's close to its last close by `until`."""
    part = closes.filter(pl.col("date").is_between(t, until)).sort("date")
    ends = part.group_by("ticker", maintain_order=True).agg(
        pl.col("date").first().alias("a"),
        pl.col("close").first().alias("start"),
        pl.col("close").last().alias("end"),
    )
    ends = ends.filter(pl.col("a") == t)
    return dict(
        zip(ends["ticker"], (ends["end"] / ends["start"] - 1).to_list(), strict=True)
    )


def sides(
    ranked: pl.DataFrame, form: str | None = None
) -> tuple[set[str], set[str], set[str]]:
    """The tickers long (changed least), short (changed most), and all those ranked."""
    part = ranked if form is None else ranked.filter(pl.col("form") == form)
    long = set(part.filter(pl.col("rank") > 1 - FIFTH)["ticker"].to_list())
    short = set(part.filter(pl.col("rank") <= FIFTH)["ticker"].to_list())
    return long, short, set(part["ticker"].to_list())


def side_mean(names: set[str], returns: dict[str, float]) -> float:
    values = [returns[n] for n in names if n in returns]
    return float(np.mean(values)) if values else 0.0


def monthly(
    prices: pl.DataFrame,
    change: pl.DataFrame,
    members: pl.DataFrame,
    measure: str = "cosine",
    form: str | None = None,
    months_held: int = 1,
) -> pl.DataFrame:
    """Each month's long-short, from the data as seen: (month, long, short, everyone, before,
    after, turnover, names)."""
    prices = seen(prices, "date")
    change = seen(change, "filed")
    closes = prices.select("date", "ticker", "close")
    ends = month_ends(sorted(prices["date"].unique().to_list()))
    chains, reports = chain_of(members), reports_by_cik(change)
    held: list[tuple[set[str], set[str], set[str]]] = []
    rows = []
    for t, until in pairwise(ends):
        tickers = prices.filter((pl.col("date") == t) & pl.col("sp500"))[
            "ticker"
        ].to_list()
        now = sides(formation(t, tickers, chains, reports, measure), form)
        before = held[-1] if held else (set(), set(), set())
        held.append(now)
        cohorts = [c for c in held[-months_held:] if c[0] and c[1]]
        if not cohorts:
            continue
        returns = held_return(closes, t, until)
        long = float(np.mean([side_mean(c[0], returns) for c in cohorts]))
        short = float(np.mean([side_mean(c[1], returns) for c in cohorts]))
        everyone = float(np.mean([side_mean(c[2], returns) for c in cohorts]))
        changed = [
            len(a - b) / max(len(a), 1)
            for a, b in ((now[0], before[0]), (now[1], before[1]))
        ]
        # a cohort of the months held turns over each month: a share of the part replaced
        turnover = sum(changed) / months_held
        cost = COST * turnover
        rows.append(
            {
                "month": t,
                "long": long,
                "short": short,
                "everyone": everyone,
                "before": long - short,
                "after": long - short - cost,
                "turnover": sum(changed) / 2,
                "names": len(now[2]),
            }
        )
    return pl.DataFrame(rows)


def mean_t(values) -> tuple[float, float, int]:
    v = np.asarray(values, dtype=float)
    spread = v.std(ddof=1) if len(v) > 1 else 0.0
    t = float(v.mean() / spread * np.sqrt(len(v))) if spread > 0 else 0.0
    return float(v.mean() * 1e4), t, len(v)


def line(label: str, values) -> str:
    bps, t, n = mean_t(values)
    return f"{label:<34}{bps:>+9.1f} bps a month  t {t:>+5.2f}  months {n:>3}"


def main() -> None:
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close", "sp500"])
    change = pl.read_parquet(OUT / "change.parquet")
    members = pl.read_parquet(OUT / "members.parquet")
    base = monthly(prices, change, members)
    last = seen(prices, "date")["date"].max()
    sharpe = mean_t(base["after"])[1] / np.sqrt(base.height / 12)
    out = [
        "first_look.py: lazy prices among S&P 500 members, on 2016-2021 only.",
        f"Nothing after {LAST_SEEN} was read. Last month held to {last}.",
        "",
        "THE NUMBER: long the fifth that changed least, short the fifth that changed most.",
        line("After costs", base["after"]),
        (
            f"Annual Sharpe ratio {sharpe:+.2f}; each side's turnover a month "
            f"{base['turnover'].mean():.0%}; {base['names'].mean():.0f} names ranked on average."
        ),
        "",
        "Beside it, not to be chosen among:",
        line("Before costs", base["before"]),
        line("Long side, against everyone ranked", base["long"] - base["everyone"]),
        line("Short side, against everyone ranked", base["everyone"] - base["short"]),
    ]
    for (year,), part in base.group_by(pl.col("month").dt.year(), maintain_order=True):
        out.append(line(f"{year}", part["after"]))
    for label, kwargs in (
        ("10-K reports only", {"form": "10-K"}),
        ("10-Q reports only", {"form": "10-Q"}),
        ("Jaccard for cosine", {"measure": "jaccard"}),
        ("Held three months", {"months_held": 3}),
        ("Held six months", {"months_held": 6}),
    ):
        out.append(line(label, monthly(prices, change, members, **kwargs)["after"]))
    RESULTS.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
