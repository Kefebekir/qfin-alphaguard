"""N6's second look, once: the chosen variant on 2022 to 9 October 2026 (pre-registration N6).

Efe runs it, after variants/N6_chosen.json is committed:

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_second.py

The chosen variant (its settings and seeds) runs on each year 2022-2026 as on development: one
model a seed trained on every member-day from 2016 that ends five sessions before the year's
first, their forecasts averaged, with its two placebos, scored with N4's variant 2's portfolio.
These years have been seen through N2's to N4's holdouts, so nothing here is a test. The
pre-registration asks:

- that its mean a day after costs be above 0, or N6 fails here and no card is made;
- the clip of its forward test: its largest absolute daily score, rounded up to 0.1%;
- reported beside it: each year, before costs, the long and the short side apart (against the
  mean of the day's members with news), the placebos, and N6 less N4's variant 2 day by day
  (data/news_reader/dev/days_N4.parquet) with its t.

It runs once: the result is locked in data/news_reader/n6/second_look.json with the commit,
and added to LEDGER.md. Run again, it runs nothing and writes the page from the lock:
data/news_reader/n6/second_look.html.
"""

import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from dev_report import PAGE, signed

N6 = Path("data/news_reader/n6")
LOCK = N6 / "second_look.json"
CHOSEN = Path("research/news_reader/variants/N6_chosen.json")
LEDGER = Path("research/news_reader/LEDGER.md")
N4_DAYS = Path("data/news_reader/dev/days_N4.parquet")


def mean_t(values) -> dict:
    v = np.asarray(values, dtype=float)
    spread = v.std(ddof=1) if len(v) > 1 else 0.0
    t = float(v.mean() / spread * math.sqrt(len(v))) if spread > 0 else 0.0
    return {"bps": float(v.mean() * 1e4) if len(v) else 0.0, "t": t, "days": len(v)}


def clip_of(scores) -> float:
    """The largest absolute daily score, rounded up to 0.1%."""
    return math.ceil(float(np.max(np.abs(scores))) * 1000 - 1e-9) / 1000


def report(days: list[dict], n4: pl.DataFrame) -> dict:
    """What the pre-registration asks of the second look, from the variant's days."""
    frame = pl.DataFrame(days).with_columns(pl.col("day").str.to_date())
    score, long, short, everyone = (
        frame[c].to_numpy() for c in ("score", "long", "short", "everyone")
    )
    both = frame.select("day", "score").join(
        n4.select("day", pl.col("score").alias("n4")), on="day", how="inner"
    )
    years = {
        str(y): mean_t(part["score"].to_numpy())
        for (y,), part in frame.group_by(pl.col("day").dt.year(), maintain_order=True)
    }
    after = mean_t(score)
    return {
        "after": after,
        "before": mean_t(long - short),
        "long": mean_t(long - everyone),
        "short": mean_t(everyone - short),
        "years": dict(sorted(years.items())),
        "less_n4": mean_t((both["score"] - both["n4"]).to_numpy()),
        "n4": mean_t(both["n4"].to_numpy()),
        "clip": clip_of(score),
        "passes": after["bps"] > 0,
    }


def committed(path: Path) -> str:
    """The commit, refusing a choice that is not committed or has changed since."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        capture_output=True,
        check=False,
    ).returncode
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path), "research/news_reader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if tracked != 0 or status.strip():
        raise SystemExit(f"commit {path} and the harness before the second look")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def section(lock: dict) -> str:
    r = lock["report"]

    def line(v: dict) -> str:
        return f"{signed(v['bps'])} bps a day (t {signed(v['t'])})"

    years = ", ".join(f"{y} {signed(v['bps'])}" for y, v in r["years"].items())
    placebos = lock["result"]["all"]
    verdict = (
        "**Above 0: a card may be made.**"
        if r["passes"]
        else "**Not above 0: N6 fails here; no card is made.**"
    )
    return f"""
## N6's second look (run once, {lock["run_utc"][:10]}, commit {lock["commit"][:7]}; seen years, not a test)

Variant {lock["chosen"]["variant"]}, `{lock["chosen"]["name"]}`, on 2022 to 9 October 2026:
{line(r["after"])} after costs over {r["after"]["days"]:,} days. {verdict}

- Each year: {years}.
- Before costs {line(r["before"])}; the long side {line(r["long"])} and the short side
  {line(r["short"])}, each against the mean of the day's members with news.
- The placebos: headlines dealt within each day {signed(placebos["dealt"]["bps"])}, the second
  {signed(placebos["silent"]["bps"])}.
- N6 less N4's variant 2, day by day: {line(r["less_n4"])} over {r["less_n4"]["days"]:,} days
  (N4's variant 2 {signed(r["n4"]["bps"])} on them).
- The forward test's clip, the largest absolute daily score rounded up to 0.1%:
  {r["clip"]:.3f}.
"""


def page(lock: dict) -> str:
    r = lock["report"]
    rows = "".join(
        f"<tr><td>{label}</td><td class='num'>{signed(v['bps'])}</td>"
        f"<td class='num'>{signed(v['t'])}</td><td class='num'>{v['days']:,}</td></tr>"
        for label, v in [
            ("After costs", r["after"]),
            ("Before costs", r["before"]),
            ("Long side, against the mean", r["long"]),
            ("Short side, against the mean", r["short"]),
            *[(y, v) for y, v in r["years"].items()],
            ("N6 less N4's variant 2", r["less_n4"]),
            ("N4's variant 2, the same days", r["n4"]),
        ]
    )
    body = f"""
<h2>The second look, 2022 to 9 October 2026 (bps a day; t)</h2>
<div class="tables"><table><thead><tr><th></th><th class="num">Bps a day</th>
<th class="num">t</th><th class="num">Days</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">Seen years: not a test. The forward test's clip: {r["clip"]:.3f}.</p>
"""
    out = PAGE.format(
        title="N6 second look",
        heading="N6's second look",
        updated=lock["run_utc"] + " UTC",
        summary=f"<li>{'Above 0: a card may be made.' if r['passes'] else 'Not above 0: no card.'}</li>",
        chart_title="",
        chart="",
        table_title="",
        head="",
        rows="",
        note="",
    )
    start, end = out.index("<h2></h2>"), out.index("</main>")
    return out[:start] + body + out[end:]


def main() -> None:
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
        (N6 / "second_look.html").write_text(page(lock))
        print(
            f"the second look has run ({lock['run_utc']}): nothing runs again; the page is written"
        )
        return
    import n6_harness as nh
    import n6_model as nm
    import n6_run as nr

    if not CHOSEN.exists():
        raise SystemExit(f"{CHOSEN} is not written: run n6_run.py choose and commit it")
    commit = committed(CHOSEN)
    chosen = json.loads(CHOSEN.read_text())
    s = nm.Settings(**chosen["settings"])
    result = nr.run(
        chosen["name"],
        s,
        seeds=tuple(chosen["seeds"]),
        years_run=nh.SECOND_LOOK,
        keep_days=True,
    )
    lock = {
        "chosen": chosen,
        "commit": commit,
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "result": {k: v for k, v in result.items() if k != "days"},
        "days": result["days"],
        "report": report(result["days"], pl.read_parquet(N4_DAYS)),
    }
    LOCK.write_text(json.dumps(lock, indent=1))
    LEDGER.write_text(LEDGER.read_text().rstrip("\n") + "\n" + section(lock))
    (N6 / "second_look.html").write_text(page(lock))
    r = lock["report"]
    print(
        f"second look: {r['after']['bps']:+.2f} bps a day (t {r['after']['t']:+.2f}) over "
        f"{r['after']['days']:,} days; less N4 {r['less_n4']['bps']:+.2f}; clip {r['clip']:.3f}; "
        f"{'a card may be made' if r['passes'] else 'N6 fails here'}"
    )


if __name__ == "__main__":
    main()
