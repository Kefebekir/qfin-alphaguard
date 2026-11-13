"""The last look at lazy prices among S&P 500 members: the risk factors and the managers'
discussion alone, on 2016-2021 only. No model.

The first look (first_look.py) read whole reports and made +14.2 bps a month after costs, t
+0.61. Its word counts were mostly common words, and the study's strongest results are in
single sections. This is the one more look Efe chose on 13 November 2026. Only the text read
differs from the first look's, and THE RULE was committed before the first run.

THE RULE:
- Each report's risk factors and MD&A (sections.py), pooled, set against the same report's a
  year before: the cosine of their word counts. A report counts when its pooled sections and
  the earlier report's both have MIN_WORDS words or more.
- Everything else is the first look's (first_look.monthly):
  - at each month's last session, the members with a report filed in the three months before,
    a company once, ranked within form;
  - long the fifth that changed least, short the fifth that changed most, equal weights, to
    the next month's last close;
  - costs of 1 bp a round trip on the part of each side that changed.
- THE NUMBER: the mean a month after costs, and its t, over 2016-2021.
- THE GATE: with a t of 2.0 or more, a test on 2022-2026 is written down for Efe's approval
  and run once. Below 2.0, B is closed and 2022-2026 stay unread.

Reported beside it, not to be chosen among:
- before costs; each side against everyone ranked; each year;
- the risk factors alone and the MD&A alone, on reports with that section in both years;
- each form; Jaccard for cosine; held three and six months.

Nothing after 2021 is read: first_look.monthly takes prices and reports through its one door,
first_look.seen.

Usage: uv run python research/filings/section_look.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import first_look as fl

RESULTS = Path(__file__).parent / "results_section_look.txt"
SECTIONS = fl.OUT / "sections.parquet"
MIN_WORDS = 1_000  # two sections pooled; a 10-Q's MD&A alone runs to about 6,000
fl.MIN_WORDS = MIN_WORDS  # the first look's filter, set to the sections' length
SECTION_FOUND = 100  # words, for a section scored alone
GATE = 2.0
ALONE = {"Risk factors alone": "risk", "MD&A alone": "mdna"}


def scored(sections: pl.DataFrame) -> pl.DataFrame:
    """The reports with a pooled score: a report a year before and words in both."""
    return sections.filter(
        pl.col("cosine").is_not_null() & pl.col("cosine").is_not_nan()
    )


def alone(sections: pl.DataFrame, name: str) -> pl.DataFrame:
    """The reports whose section `name` is found in both years, scored on it alone."""
    return scored(sections).filter(
        (pl.col(f"{name}_words") >= SECTION_FOUND)
        & (pl.col(f"earlier_{name}_words") >= SECTION_FOUND)
        & pl.col(f"{name}_cosine").is_not_nan()
    )


def line(label: str, months: pl.DataFrame) -> str:
    """first_look.line on the months' results after costs; a form with no reports has none."""
    if months.is_empty():
        return f"{label:<34}no months"
    return fl.line(label, months["after"])


def verdict(t: float) -> str:
    if t >= GATE:
        return f"t {t:+.2f} reaches the gate of {GATE}: a test on 2022-2026 is worth writing down."
    return f"t {t:+.2f} is below the gate of {GATE}: B is closed, and 2022-2026 stay unread."


def look(
    prices: pl.DataFrame, sections: pl.DataFrame, members: pl.DataFrame
) -> list[str]:
    """The rule's lines: THE NUMBER, its verdict, and the lines beside it."""
    pooled = scored(sections)
    base = fl.monthly(prices, pooled, members)
    _, t, _ = fl.mean_t(base["after"])
    last = fl.seen(prices, "date")["date"].max()
    sharpe = t / np.sqrt(base.height / 12)
    out = [
        "section_look.py: lazy prices in the risk factors and MD&A, on 2016-2021 only.",
        f"Nothing after {fl.LAST_SEEN} was read. Last month held to {last}.",
        "",
        (
            "THE NUMBER: long the fifth whose two sections changed least, short the fifth "
            "that changed most."
        ),
        fl.line("After costs", base["after"]),
        (
            f"Annual Sharpe ratio {sharpe:+.2f}; each side's turnover a month "
            f"{base['turnover'].mean():.0%}; {base['names'].mean():.0f} names ranked on average."
        ),
        verdict(t),
        "",
        "Beside it, not to be chosen among:",
        fl.line("Before costs", base["before"]),
        fl.line("Long side, against everyone ranked", base["long"] - base["everyone"]),
        fl.line(
            "Short side, against everyone ranked", base["everyone"] - base["short"]
        ),
    ]
    for (year,), part in base.group_by(pl.col("month").dt.year(), maintain_order=True):
        out.append(fl.line(f"{year}", part["after"]))
    for label, name in ALONE.items():
        found = alone(sections, name)
        out.append(
            line(label, fl.monthly(prices, found, members, measure=f"{name}_cosine"))
        )
    for label, kwargs in (
        ("10-K reports only", {"form": "10-K"}),
        ("10-Q reports only", {"form": "10-Q"}),
        ("Jaccard for cosine", {"measure": "jaccard"}),
        ("Held three months", {"months_held": 3}),
        ("Held six months", {"months_held": 6}),
    ):
        out.append(line(label, fl.monthly(prices, pooled, members, **kwargs)))
    return out


def main() -> None:
    prices = pl.read_parquet(fl.PRICES, columns=["date", "ticker", "close", "sp500"])
    sections = pl.read_parquet(SECTIONS)
    members = pl.read_parquet(fl.OUT / "members.parquet")
    out = look(prices, sections, members)
    RESULTS.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
