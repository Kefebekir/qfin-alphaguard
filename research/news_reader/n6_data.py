"""N6's data: each member-day's headlines one by one, its context, and its next return.

The member-days and their articles are N3's and N4's (members.py, features_n3.member_days): an
article known from 16:00 on the session before to 15:30, tagging one or two stocks, counts for
each S&P 500 member it names that day; one naming a member twice counts twice. For each
member-day:

- its headlines in the order they became known, each a row of N6's eyes (n6/headlines.npy,
  n6_eyes.py), with how many hours before 15:30 it became known, whether it counts from an
  edit, and how many stocks it tags;
- its context (n6/context.parquet, n6_context.py), from the closes of the session before;
- its return from the day's close to the next session's (harness.next_day_returns, as N4's),
  null where there is none.

Output, under data/news_reader/n6/:
- days.parquet: day, ticker, count, start and stop (its headlines' rows in pairs.parquet),
  next_return and the context's columns, sorted by day and ticker as n3_rows.parquet is;
- pairs.parquet: id, row (in headlines.npy), age_hours, edited, tags, member-day by member-day.

Checks, printed: the member-days and counts are N3's, row for row; every headline is in the
store, read by the checkpoint N3's rule gives its year; on 2,000 member-days read by one
checkpoint, the mean of their headlines in the store is N3's feature (n3_base-last) in the
same scale; vol20 is N4's volatility (risk.parquet); and the member-days with a next return to
9 October 2026 are as many as the frozen N4 was fitted on.

Usage: uv run python research/news_reader/n6_data.py
"""

import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from eyes_check import CHECKPOINTS
from features_n3 import member_days
from members import member_articles

NEW_YORK = ZoneInfo("America/New_York")
INDEX = Path("data/news_reader/emb/index.parquet")
PRICES = Path("data/raw/prices.parquet")
N6 = Path("data/news_reader/n6")
PIT_EMB = Path("data/news_reader/pit_emb")
RISK = Path("data/news_reader/risk/risk.parquet")
EYES = Path("data/models/n6-eyes.npz")
N4_FITTED = 259_967  # member-days the frozen N4 was fitted on, 2016-01-05 to 2026-10-09


def decision(day: date) -> datetime:
    """15:30 New York time on `day`, in UTC."""
    return datetime.combine(day, time(15, 30), tzinfo=NEW_YORK).astimezone(UTC)


def reader(known: pl.Expr) -> pl.Expr:
    """The checkpoint's place in CHECKPOINTS that reads an article known at `known` (UTC),
    by N3's rule (members.checkpoint): the year before's, by New York's calendar."""
    year = known.dt.convert_time_zone("America/New_York").dt.year()
    return (year.clip(2016, 2025) - 2016).cast(pl.Int8)


def arrange(
    groups: pl.DataFrame,
    rows: pl.DataFrame,
    index: pl.DataFrame,
    store_ids: np.ndarray,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The pairs, member-day by member-day in the order they became known, with their rows
    in the store; and the member-days with start and stop into them.

    `groups` holds each pair's id and member-day (group, the member-day's place in `rows`);
    `rows` the member-days (ticker, day, count); `index` each article's id, created, known
    and symbols; `store_ids` the article of each row of headlines.npy."""
    rows = rows.with_row_index("group")
    store = pl.DataFrame(
        {"id": store_ids, "row": np.arange(len(store_ids), dtype=np.int32)}
    )
    ends = {d: decision(d) for d in rows["day"].unique().to_list()}
    pairs = (
        groups.join(rows.select("group", "day"), on="group")
        .join(
            index.select(
                "id",
                "created",
                "known",
                tags=pl.col("symbols").list.len().cast(pl.Int16),
            ),
            on="id",
            how="left",
        )
        .join(store, on="id", how="left")
        .with_columns(
            end=pl.col("day").replace_strict(
                ends, return_dtype=pl.Datetime("us", "UTC")
            )
        )
        .with_columns(
            age_hours=(
                (pl.col("end") - pl.col("known")).dt.total_microseconds() / 3.6e9
            ).cast(pl.Float32),
            edited=pl.col("known") != pl.col("created"),
        )
        .sort("group", "known", "id")
    )
    if pairs["row"].null_count():
        raise ValueError(f"{pairs['row'].null_count()} headlines are not in the store")
    if (pairs["age_hours"] < 0).any():
        raise ValueError("a headline became known after 15:30")
    n = np.bincount(pairs["group"].to_numpy(), minlength=rows.height)
    if not (n == rows["count"].to_numpy()).all():
        raise ValueError("a member-day's count is not its number of headlines")
    stop = np.cumsum(n)
    days = rows.with_columns(start=pl.Series(stop - n), stop=pl.Series(stop))
    return (
        pairs.select("id", "row", "age_hours", "edited", "tags", "known"),
        days.drop("group"),
    )


def main() -> None:
    import harness
    from n6_eyes import aligned

    index = pl.read_parquet(INDEX)
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "sp500"])
    rows, groups = member_days(member_articles(index, prices))
    n3 = pl.read_parquet(PIT_EMB / "n3_rows.parquet")
    print(f"member-days and counts the same as N3's, row for row: {rows.equals(n3)}")

    store_ids = np.load(N6 / "ids.npy")
    which = np.load(N6 / "checkpoint.npy")
    pairs, days = arrange(groups, rows, index, store_ids)
    rule = pairs.select(reader(pl.col("known")))["known"].to_numpy()
    same = bool((which[pairs["row"].to_numpy()] == rule).all())
    print(
        f"{pairs.height:,} headlines, every one in the store, read by the checkpoint "
        f"N3's rule gives: {same}"
    )

    store = np.load(N6 / "headlines.npy", mmap_mode="r")
    n3_last = np.load(PIT_EMB / "n3_base-last.npy", mmap_mode="r")
    arrays = dict(np.load(EYES))
    one = (
        pairs.with_columns(
            group=pl.Series(
                np.repeat(np.arange(days.height), days["count"].to_numpy())
            ),
            reader=reader(pl.col("known")),
        )
        .group_by("group")
        .agg(pl.col("reader").n_unique().alias("readers"), pl.col("reader").first())
        .filter(pl.col("readers") == 1)
        .sort(
            "group"
        )  # group_by's order varies: sorted, the sample is the same each run
    )
    sample = one.sample(2_000, seed=0).sort("group")
    rows_of = pairs["row"].to_numpy()
    start, stop = days["start"].to_numpy(), days["stop"].to_numpy()
    worst = 1.0
    for group, k in sample.select("group", "reader").iter_rows():
        mine = np.asarray(store[rows_of[start[group] : stop[group]]], np.float32)
        theirs = aligned(arrays, CHECKPOINTS[k], n3_last[group : group + 1])[0]
        a = mine.mean(0)
        worst = min(
            worst, float(a @ theirs / np.linalg.norm(a) / np.linalg.norm(theirs))
        )
    print(
        f"2,000 member-days read by one checkpoint: their headlines' mean in the store "
        f"against N3's feature in the same scale, cosine min {worst:.6f}"
    )

    context = pl.read_parquet(N6 / "context.parquet")
    table = (
        days.join(harness.next_day_returns(), on=["day", "ticker"], how="left")
        .join(context, on=["day", "ticker"], how="left")
        .sort("day", "ticker")
    )
    risk = pl.read_parquet(RISK).select("day", "ticker", "vol")
    both = table.select("day", "ticker", "vol20").join(risk, on=["day", "ticker"])
    gap = both.select((pl.col("vol20") - pl.col("vol")).abs().max()).item()
    print(
        f"vol20 against N4's volatility on {both.height:,} member-days: largest gap {gap:.2e}"
    )
    fitted = table.filter(
        pl.col("next_return").is_not_null() & (pl.col("day") <= date(2026, 10, 9))
    ).height
    print(
        f"member-days with a next return to 9 October 2026: {fitted:,} "
        f"(the frozen N4 was fitted on {N4_FITTED:,})"
    )
    table.write_parquet(N6 / "days.parquet")
    pairs.drop("known").write_parquet(N6 / "pairs.parquet")
    print(
        f"{N6 / 'days.parquet'}: {table.height:,} member-days, {table.width} columns; "
        f"{N6 / 'pairs.parquet'}: {pairs.height:,} headlines"
    )


if __name__ == "__main__":
    main()
