"""Can Alpaca's daily bars continue EODHD's history? (decision 0022)

EODHD's subscription ended on 11 November 2026, and with it the nightly prices. `qfin
ingest` now continues EODHD's last file with Alpaca's bars as traded and Alpaca's
corporate actions (ingest.continue_prices). Two questions, on real data:

1. The past, rebuilt. EODHD's history is cut at the end of 2024 and continued by Alpaca
   to 10 November 2026, as the nightly job will continue it from now on; the rebuilt
   bars are set beside EODHD's own. Do the closes, the daily returns and the splits and
   dividends agree?
2. The switch itself. EODHD's whole history is continued to the last session over. Is
   every return to 10 November unchanged, does the result pass validation, and do N5's
   and N6's records see the same members and returns on every day they scored?

Usage: uv run --env-file .env python research/phase3/prices_from_alpaca.py \\
    data/raw/prices_eodhd.parquet
"""

import importlib.util
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.corporate import corporate_actions
from qfin_alphaguard.data.ingest import continue_prices
from qfin_alphaguard.data.universe import sp500
from qfin_alphaguard.data.validate import validate_prices
from qfin_alphaguard.events import Split

RESULTS = Path(__file__).parent / "results" / "05_prices_from_alpaca.txt"
CUT = date(2024, 12, 31)
LAST_EODHD = date(2026, 11, 10)
RECORD = Path(__file__).parents[1] / "news_reader" / "forward_record.py"


def returns(prices: pl.DataFrame) -> pl.DataFrame:
    return prices.sort("ticker", "date").with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("r")
    )


def share(mask: pl.Series) -> str:
    return f"{mask.mean():.2%}" if len(mask) else "none"


def actions_after(prices: pl.DataFrame, first: date, last: date) -> dict:
    """The splits and dividends after `first`, by `last`. EODHD's file holds 33
    adjustments that corporate_actions refuses, all but one before 2026; a ticker
    with one is left out, and counted under the key "refused"."""
    found, refused = {}, []
    for (ticker,), bars in prices.partition_by("ticker", as_dict=True).items():
        try:
            actions = corporate_actions(bars)
        except ValueError:
            refused.append(ticker)
            continue
        for a in actions:
            if first < a.day <= last:
                size = a.ratio if isinstance(a, Split) else a.amount
                found[(type(a).__name__, a.ticker, a.day)] = size
    found["refused"] = sorted(refused)
    return found


def as_of(prices: pl.DataFrame, day: date) -> pl.DataFrame:
    """The bars to `day`, as EODHD would have sent them then: adjusted only for the
    splits and dividends to that day, so each ticker's factors are 1 on its last bar.
    The stored file was adjusted on 11 November 2026, for every event before it."""
    part = prices.filter(pl.col("date") <= day).sort("ticker", "date")
    last = pl.col("adjustment").last().over("ticker")
    split = pl.col("split_factor").last().over("ticker")
    return part.with_columns(
        *(pl.col(c) / last for c in ("open", "high", "low", "close")),
        (pl.col("volume") / split).round().cast(pl.Int64),
        pl.col("adjustment") / last,
        pl.col("split_factor") / split,
    )


def same_actions(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        a[k] == b[k] if k == "refused" else abs(a[k] / b[k] - 1) < 1e-9 for k in a
    )


def the_past(eodhd: pl.DataFrame, client: AlpacaClient, out: list[str]) -> None:
    cut = as_of(eodhd, CUT)
    now = datetime(2026, 11, 11, 6, 0, tzinfo=UTC)
    rebuilt = continue_prices(cut, client, sp500(), LAST_EODHD, now)
    after = (pl.col("date") > CUT) & (pl.col("date") <= LAST_EODHD)
    ours = returns(rebuilt).filter(after)
    theirs = returns(eodhd).filter(after)
    both = ours.join(theirs, on=["ticker", "date"], suffix="_e")
    members = both.filter(pl.col("sp500_e"))
    out += [
        "1. THE PAST, REBUILT: EODHD cut at 2024-12-31, continued by Alpaca to 2026-11-10",
        (
            f"Rows after the cut: rebuilt {ours.height:,}, EODHD {theirs.height:,}, "
            f"in both {both.height:,}; member-days in both {members.height:,} of "
            f"EODHD's {theirs.filter(pl.col('sp500')).height:,}."
        ),
        f"Members agree on {share(members['sp500'] == members['sp500_e'])} of member-days.",
    ]
    traded = (members["close"] / members["adjustment"]) / (
        members["close_e"] / members["adjustment_e"]
    ) - 1
    out.append(
        "Traded close, Alpaca's against EODHD's, member-days: within 1 bp "
        f"{share(traded.abs() <= 1e-4)}, within 10 bps {share(traded.abs() <= 1e-3)}."
    )
    gap = (members["r"] - members["r_e"]).drop_nulls()
    out.append(
        "Daily return, rebuilt against EODHD's, member-days: within 1 bp "
        f"{share(gap.abs() <= 1e-4)}, within 10 bps {share(gap.abs() <= 1e-3)}, "
        f"mean gap {gap.mean() * 1e4:+.3f} bps, mean absolute gap "
        f"{gap.abs().mean() * 1e4:.3f} bps."
    )
    worst = (
        members.with_columns((pl.col("r") - pl.col("r_e")).alias("gap"))
        .drop_nulls("gap")
        .sort(pl.col("gap").abs(), descending=True)
        .head(10)
    )
    out.append("The ten largest gaps (ticker, day, rebuilt, EODHD, in %):")
    out += [
        f"  {t} {d} {r * 100:+.2f} {e * 100:+.2f}"
        for t, d, r, e in worst.select("ticker", "date", "r", "r_e").rows()
    ]
    adjustment = members["adjustment"] / members["adjustment_e"] - 1
    out.append(
        "Adjustment factor, rebuilt against EODHD's: within 0.1% "
        f"{share(adjustment.abs() <= 1e-3)}, within 1% {share(adjustment.abs() <= 1e-2)}."
    )
    mine = actions_after(rebuilt, CUT, LAST_EODHD)
    known = actions_after(eodhd, CUT, LAST_EODHD)
    refused = set(mine.pop("refused")) | set(known.pop("refused"))
    kept = set(both["ticker"]) - refused
    out.append(
        f"Splits and dividends compared on {len(kept)} tickers; left out, for an "
        f"adjustment corporate_actions refuses: {', '.join(sorted(refused & set(both['ticker'])))}."
    )
    mine = {k: v for k, v in mine.items() if k[1] in kept}
    known = {k: v for k, v in known.items() if k[1] in kept}
    for kind in ("Split", "Dividend"):
        a = {k: v for k, v in mine.items() if k[0] == kind}
        b = {k: v for k, v in known.items() if k[0] == kind}
        same = [k for k in a if k in b]
        close = [k for k in same if abs(a[k] / b[k] - 1) <= 0.01]
        out.append(
            f"{kind}s after the cut: rebuilt {len(a)}, EODHD {len(b)}, on the same day "
            f"{len(same)}, of those within 1% {len(close)}; only rebuilt {len(a) - len(same)}, "
            f"only EODHD {len(b) - len(same)}."
        )
        odd = sorted(set(a) ^ set(b), key=lambda k: (k[2], k[1]))[:8]
        out += [
            f"  only {'rebuilt' if k in a else 'EODHD'}: {k[1]} {k[2]} "
            f"{(a if k in a else b)[k]:.4g}"
            for k in odd
        ]
    shares = (members["volume"] / members["split_factor"]) / (
        members["volume_e"] / members["split_factor_e"]
    )
    out.append(
        f"Shares traded, Alpaca's over EODHD's: median {shares.median():.3f}, "
        f"within 5% {share((shares - 1).abs() <= 0.05)}."
    )


def the_switch(eodhd: pl.DataFrame, client: AlpacaClient, out: list[str]) -> None:
    now = datetime.now(UTC)
    switched = continue_prices(eodhd, client, sp500(), now.date(), now)
    before = pl.col("date") <= LAST_EODHD
    old = returns(eodhd).filter(before)
    new = returns(switched).filter(before)
    both = old.join(new, on=["ticker", "date"], suffix="_n")
    gap = (both["r"] - both["r_n"]).drop_nulls().abs()
    days = sorted(switched.filter(~before)["date"].unique().to_list())
    report = validate_prices(switched)
    since = date(2014, 1, 1)
    unchanged = same_actions(
        actions_after(eodhd, since, LAST_EODHD),
        actions_after(switched, since, LAST_EODHD),
    )
    out += [
        "",
        f"2. THE SWITCH: EODHD's history continued by Alpaca, run at {now:%Y-%m-%d %H:%M} UTC",
        (
            f"New sessions: {', '.join(map(str, days))}; rows {switched.height:,} "
            f"against {eodhd.height:,}."
        ),
        (
            f"Rows to {LAST_EODHD}: EODHD {old.height:,}, after the switch "
            f"{new.height:,}; largest change in a daily return {gap.max():.2e}."
        ),
        f"Splits and dividends to {LAST_EODHD} the same, to 1e-9: {unchanged}.",
        f"Validation: {'passed' if report.ok else 'FAILED'}; "
        + "; ".join(f"[{i.severity}] {i.check}: {i.detail}" for i in report.issues),
    ]
    final = set(eodhd.filter(pl.col("date") == LAST_EODHD)["ticker"])
    spy = returns(switched).filter(pl.col("ticker") == "SPY")
    for day in days:
        on = switched.filter(pl.col("date") == day)
        r = spy.filter(pl.col("date") == day)["r"][0]
        out.append(
            f"  {day}: {on.filter(pl.col('sp500')).height} members with a bar, "
            f"{on.height} tickers; SPY {r * 1e4:+.1f} bps; with a bar on "
            f"{LAST_EODHD} and none now: {', '.join(sorted(final - set(on['ticker']))) or 'none'}"
        )
    after = actions_after(switched, LAST_EODHD, days[-1] if days else LAST_EODHD)
    after.pop("refused")
    out.append(
        "Splits and dividends since: "
        + (
            ", ".join(
                f"{k[0]} {k[1]} {k[2]} {v:.4g}"
                for k, v in sorted(after.items(), key=lambda kv: (kv[0][2], kv[0][1]))
            )
            or "none"
        )
    )

    spec = importlib.util.spec_from_file_location("forward_record", RECORD)
    record = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RECORD.parent))
    spec.loader.exec_module(record)
    scored = pl.read_parquet(Path("data/news_reader/forward/days.parquet"))["day"]
    same = 0
    for day in scored.to_list():
        a, b = record.next_returns(eodhd, day), record.next_returns(switched, day)
        if (
            record.members_on(eodhd, day) == record.members_on(switched, day)
            and a.keys() == b.keys()
            and max(abs(a[k] - b[k]) for k in a) < 1e-12
        ):
            same += 1
    out.append(
        f"N5's and N6's scored days ({scored.min()} to {scored.max()}): members and next "
        f"returns the same on {same} of {len(scored)}."
    )
    following = [d for d in days[:-1]] if days else []
    for day in [LAST_EODHD, *following]:
        r = record.next_returns(switched, day)
        members = record.members_on(switched, day)
        if r is not None:
            covered = sum(1 for k in members if any(t in r for t in members[k]))
            out.append(
                f"  {day}: next returns for {covered} of its {len(members)} members, "
                f"mean {np.mean([r[t] for k in members for t in members[k] if t in r]) * 1e4:+.1f} bps"
            )


def main(path: str) -> None:
    eodhd = pl.read_parquet(path)
    client = AlpacaClient(*api_keys())
    out = [
        f"prices_from_alpaca.py on {path}, {eodhd['date'].min()} to {eodhd['date'].max()}",
        "",
    ]
    the_past(eodhd, client, out)
    the_switch(eodhd, client, out)
    RESULTS.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
