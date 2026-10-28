"""What kinds of member news move prices, and when (seen years, no model; before N7).

Each member-day of N3 to N6 (n6/days.parquet, pairs.parquet) is tagged by the kinds of its
headlines, found by their wording (KINDS): analysts' upgrades and downgrades, price targets
raised and lowered, earnings beating and missing estimates, guidance raised and lowered. A
member-day whose headlines point both ways in a pair is left out of that pair.

For each kind, the member's return beyond the mean of all S&P 500 members that day, over:
- the day itself, from the close before to its close (the news is known by 15:30, so most of
  any reaction is in it already);
- the night after, from the day's close to the next open;
- the next session, from its open to its close;
- the next day, close to close (what N2 to N6 hold);
- five sessions, from the day's close to the fifth close after.
Each in bps: the mean over days of the day's mean over its member-days, and its t over days;
for 2016-2021 and 2022-2026 apart. Then, for the kinds that point a way, the next day and the
five sessions for headlines that came in the session (known within FRESH hours of 15:30)
against those known before it. Every one of these years has been seen: nothing here is a
test, and N7, if it comes, will say so.

Usage: uv run python research/news_reader/event_study.py
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

N6 = Path("data/news_reader/n6")
NEWS = Path("data/frontier/news_all")
PRICES = Path("data/raw/prices.parquet")
OUT = Path(__file__).parent / "results_event_study.txt"
FRESH = 6.0  # hours before 15:30: known from 09:30 on
TARGET = r"(price target|\bPT\b)"
KINDS = {
    "upgrade": r"\bupgrade[sd]?\b",
    "downgrade": r"\bdowngrade[sd]?\b",
    "target raised": rf"\b(raises?|boosts?|lifts?|increases?)\b.{{0,40}}{TARGET}|{TARGET}.{{0,20}}\b(raised|boosted|increased)\b",
    "target lowered": rf"\b(lowers?|cuts?|reduces?|trims?)\b.{{0,40}}{TARGET}|{TARGET}.{{0,20}}\b(lowered|cut|reduced)\b",
    "earnings beat": r"\bbeats?\b.{0,60}\b(estimate|consensus|eps)\b|\b(eps|sales|revenue)\b.{0,60}\bbeats?\b",
    "earnings miss": r"\bmiss(es)?\b.{0,60}\b(estimate|consensus|eps)\b|\b(eps|sales|revenue)\b.{0,60}\bmiss(es)?\b",
    "guidance raised": r"\b(raises?|boosts?|lifts?|ups)\b.{0,40}\b(guidance|outlook|forecast)\b",
    "guidance lowered": r"\b(lowers?|cuts?|reduces?)\b.{0,40}\b(guidance|outlook|forecast)\b",
}
PAIRS = (
    ("upgrade", "downgrade"),
    ("target raised", "target lowered"),
    ("earnings beat", "earnings miss"),
    ("guidance raised", "guidance lowered"),
)
MEASURES = {
    "same": "the day",
    "night": "night after",
    "session": "next session",
    "next": "next day",
    "five": "5 sessions",
}
ERAS = {"2016-2021": (2016, 2021), "2022-2026": (2022, 2026)}


def kinds_of(headline: str) -> set[str]:
    return {
        k for k, pattern in KINDS.items() if re.search(pattern, headline, re.IGNORECASE)
    }


def headlines(wanted: set[int]) -> dict[int, str]:
    out = {}
    for path in sorted(NEWS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            a = json.loads(line)
            if a["id"] in wanted:
                out[a["id"]] = a["headline"]
    return out


def excess_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """(day, ticker) with each measure's return less the day's mean over S&P 500 members."""
    p = prices.sort("ticker", "date")

    def ahead(c: str, n: int) -> pl.Expr:
        return pl.col(c).shift(-n).over("ticker")

    p = p.with_columns(
        same=pl.col("close") / pl.col("close").shift(1).over("ticker") - 1,
        night=ahead("open", 1) / pl.col("close") - 1,
        session=ahead("close", 1) / ahead("open", 1) - 1,
        next=ahead("close", 1) / pl.col("close") - 1,
        five=ahead("close", 5) / pl.col("close") - 1,
    )
    members = p.filter(pl.col("sp500"))
    return members.select(
        pl.col("date").alias("day"),
        "ticker",
        *[(pl.col(m) - pl.col(m).mean().over("date")).alias(m) for m in MEASURES],
    )


def daily_mean_t(frame: pl.DataFrame, measure: str) -> tuple[float, float, int]:
    """Bps: the mean over days of each day's mean, its t over days, and the member-days."""
    by_day = (
        frame.drop_nulls(measure)
        .group_by("day")
        .agg(pl.col(measure).mean())[measure]
        .to_numpy()
    )
    n = frame.drop_nulls(measure).height
    if len(by_day) < 3:
        return float("nan"), float("nan"), n
    t = by_day.mean() / by_day.std(ddof=1) * np.sqrt(len(by_day))
    return float(by_day.mean() * 1e4), float(t), n


def main() -> None:
    days = pl.read_parquet(
        N6 / "days.parquet", columns=["day", "ticker", "start", "stop"]
    )
    pairs = pl.read_parquet(N6 / "pairs.parquet", columns=["id", "age_hours"])
    text = headlines(set(pairs["id"].to_list()))
    group = np.repeat(np.arange(days.height), (days["stop"] - days["start"]).to_numpy())
    tagged = pairs.with_columns(
        group=pl.Series(group, dtype=pl.UInt32),
        kinds=pl.Series([sorted(kinds_of(text[i])) for i in pairs["id"].to_list()]),
    ).explode("kinds", empty_as_null=True)
    tagged = tagged.drop_nulls("kinds")
    per_day = tagged.group_by("group", "kinds").agg(
        pl.col("age_hours").min().alias("age")
    )
    days = days.with_row_index("group")
    returns = excess_returns(
        pl.read_parquet(PRICES, columns=["date", "ticker", "open", "close", "sp500"])
    )
    base = days.join(returns, on=["day", "ticker"], how="left")
    lines = [
        "event_study.py: member-days by the kinds of their headlines, returns beyond all",
        "S&P 500 members' mean that day, in bps a day (t over days; member-days).",
        "",
    ]
    kinds = per_day.pivot(on="kinds", index="group", values="age")
    for era, (first, last) in ERAS.items():
        lines.append(f"{era}")
        head = f"{'kind':<18}" + "".join(f"{label:>24}" for label in MEASURES.values())
        lines.append(head)
        era_days = base.filter(pl.col("day").dt.year().is_between(first, last))
        rows = [("all with news", era_days)]
        for up, down in PAIRS:
            for k, other in ((up, down), (down, up)):
                chosen = kinds.filter(pl.col(k).is_not_null())
                if other in kinds.columns:
                    chosen = chosen.filter(pl.col(other).is_null())
                rows.append((k, era_days.join(chosen.select("group"), on="group")))
        for label, frame in rows:
            cells = ""
            for m in MEASURES:
                bps, t, n = daily_mean_t(frame, m)
                cells += f"{bps:>+9.1f} ({t:>+5.1f}; {n:>5})"
            lines.append(f"{label:<18}{cells}")
        lines.append("")
    lines.append(
        f"The next day and 5 sessions by when the headline came: in the session (within {FRESH:g} "
        "hours of 15:30) or before it; 2016-2026."
    )
    for up, down in PAIRS:
        for k, other in ((up, down), (down, up)):
            chosen = kinds.filter(pl.col(k).is_not_null())
            if other in kinds.columns:
                chosen = chosen.filter(pl.col(other).is_null())
            for when, keep in (
                ("in the session", pl.col(k) <= FRESH),
                ("before it", pl.col(k) > FRESH),
            ):
                frame = base.join(chosen.filter(keep).select("group"), on="group")
                cells = ""
                for m in ("next", "five"):
                    bps, t, n = daily_mean_t(frame, m)
                    cells += f"  {MEASURES[m]} {bps:+6.1f} (t {t:+5.1f}; {n:>5})"
                lines.append(f"{k:<18}{when:<16}{cells}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
