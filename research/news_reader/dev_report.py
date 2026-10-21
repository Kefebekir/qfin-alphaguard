"""Efe's development runs of the news tests: fit a variant, count it, show every variant.

Efe runs it, with the settings he chose:

    uv run python research/news_reader/dev_report.py N2 16 10.0
    uv run python research/news_reader/dev_report.py N3 16 10.0 base-last
    uv run python research/news_reader/dev_report.py N4 64 10.0 base-last target=scaled side=fifth

N4's levers (target, side, weights, sectors) left out take N3's values; all four go in the
ledger. Once a test's choice is committed, no more of its variants are run.

It fits and scores efe_model.make(components, penalty) on the development years
(harness.run), then:
- adds the variant's row to LEDGER.md, numbered within its test. It refuses an eleventh
  variant of a test, and runs nothing for settings already tried, unless they were scored
  under an earlier rule (harness.SCORING): then the same variant is scored again and its
  row says so.
- keeps every result in data/news_reader/dev/<test>.json, with counting alone, the
  pre-registered comparison, computed once.
- writes data/news_reader/dev/<test>.html: each variant's mean a day after costs against
  counting alone, and each development year.

The holdout, once, after the choice is committed in variants/<test>_chosen.json:

    uv run python research/news_reader/dev_report.py N2 holdout

It runs the committed choice through harness.run_holdout, adds the result to the end of
LEDGER.md and writes data/news_reader/dev/<test>_holdout.html. Run again, it runs nothing:
it writes the page from the stored result.

Exploration, by amendment 6, after the test's own holdout:

    uv run python research/news_reader/dev_report.py N4 explore 6 10 8

It runs those development variants, and only those, on the holdout years, each once, keeps
them in data/news_reader/explore_<test>.json, records them in the ledger as exploration, not
tests, beside the bar four tries would need, and writes <test>_explore.html.

The reports the pre-registrations ask for beside the pass, after N2's, N3's and N4's holdouts:

    uv run python research/news_reader/dev_report.py reports

It runs each committed choice again on the holdout years, day by day (harness.daily), and
stops unless each reproduces its locked mean and days. Then: before costs; the long and the
short side apart, each against the mean of the day's members with news; N3 less N2 and N4
less N3, day by day, with their t; and, as a diagnostic, each month of 2026 and how much of
the result its ten best days carry. To the ledger and to reports.html; the days to
data/news_reader/dev/days_<test>.parquet.
"""

import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

LEDGER = Path("research/news_reader/LEDGER.md")
DEV = Path("data/news_reader/dev")
LIMIT = 10
VARIANTS_HEADER = "| # | Test | Run (UTC) | Settings (`efe_model.make`) |"


def signed(x: float, digits: int = 2) -> str:
    """+1.23 or −1.23, with a true minus sign, as the ledger writes numbers."""
    return f"{x:+.{digits}f}".replace("-", "−")


LEVERS = (
    "target",
    "side",
    "weights",
    "sectors",
)  # N4's, in the pre-registration's order


def settings_text(settings: dict) -> str:
    text = f"components {settings['components']}, penalty {settings['penalty']}"
    if settings.get("features"):
        text += f", features {settings['features']}"
    if "target" in settings:
        text += ", " + ", ".join(f"{lever} {settings[lever]}" for lever in LEVERS)
    return text


def chart_label(name: str, settings: dict) -> str:
    """A variant's name on the chart: in full, or shortened when it carries N4's levers."""
    if "target" not in settings:
        return f"{name} · {settings_text(settings)}"
    levers = "/".join(settings[lever] for lever in LEVERS)
    return (
        f"{name} · {settings['components']} comp, pen {settings['penalty']:g}, "
        f"{settings['features']}, {levers}"
    )


def ledger_row(
    n: int,
    test: str,
    run_utc: str,
    settings: dict,
    result: dict,
    bar: dict,
    before: dict | None = None,
) -> str:
    years = " / ".join(signed(y["mean_bps"]) for y in result["years"].values())
    gap = result["mean_bps"] - bar["mean_bps"]
    side = "above" if gap > 0 else "below"
    note = f"{signed(gap)} bps a day {side} counting alone ({signed(bar['mean_bps'])})"
    if before:
        note += (
            f"; scored again, {result['scoring']}: it was {signed(before['mean_bps'])} bps "
            f"(t {signed(before['t'])})"
        )
    return (
        f"| {n} | {test} | {run_utc} | {settings_text(settings)} | "
        f"{signed(result['mean_bps'])} bps | {signed(result['t'])} | {years} | {note} |"
    )


def add_row(ledger: str, row: str) -> str:
    """The ledger with `row` after the last row of the variants table."""
    lines = ledger.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(VARIANTS_HEADER))
    end = start + 1
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return "\n".join([*lines[:end], row, *lines[end:]])


def replace_row(ledger: str, test: str, n: int, row: str) -> str:
    """The ledger with variant n of `test` replaced by `row`."""
    lines = ledger.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(VARIANTS_HEADER))
    for i in range(start + 2, len(lines)):
        if not lines[i].startswith("|"):
            break
        cells = lines[i].split("|")
        if cells[1].strip() == str(n) and cells[2].strip() == test:
            lines[i] = row
            return "\n".join(lines)
    raise ValueError(f"no row for {test}'s variant {n}")


def variants_in_ledger(ledger: str, test: str) -> int:
    lines = ledger.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(VARIANTS_HEADER))
    count = 0
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        if line.split("|")[2].strip() == test:
            count += 1
    return count


class AlreadyTried(SystemExit):
    """These settings were run under the current rule: nothing is run again."""


def stored(result: dict) -> dict:
    """A result as the store keeps it: through JSON, whose keys are strings (the years)."""
    return json.loads(json.dumps(result))


def check_new(
    store: dict, ledger: str, test: str, settings: dict, scoring: str
) -> tuple[int, bool]:
    """The variant's number within its test, and whether it is scored again (settings tried
    under an earlier rule). Refuses a repeat under the same rule, or an eleventh."""
    for v in store["variants"]:
        if v["settings"] == settings:
            if v["result"].get("scoring") == scoring:
                raise AlreadyTried(
                    f"{test}: {settings_text(settings)} is variant {v['n']} already; "
                    f"its result is on {DEV / (test + '.html')}, written again"
                )
            return v["n"], True
    n = variants_in_ledger(ledger, test)
    if n >= LIMIT:
        raise SystemExit(
            f"{test} has its {LIMIT} variants; the next step is the choice"
        )
    return n + 1, False


def _cell(value: float, scale: float) -> str:
    share = min(abs(value) / scale, 1.0) * 70 if scale else 0.0
    pole = "var(--pos)" if value > 0 else "var(--neg)"
    return (
        f'<td class="num" style="background: color-mix(in oklab, {pole} {share:.0f}%, '
        f'var(--mid))">{signed(value)}</td>'
    )


def _bars(rows: list[tuple[str, dict]], bar: dict) -> str:
    """A horizontal bar a variant: its mean a day after costs, from zero; counting alone
    dashed. Labels and values sit outside the plot, so nothing overlaps."""
    width, right, values_x, row_h, top = 1000, 790, 805, 34, 30
    # Room on the left for the longest label (12px text, about 6.6 units a character).
    left = max(230, min(460, round(14 + 6.6 * max(len(label) for label, _ in rows))))
    height = top + row_h * len(rows) + 30
    means = [r["mean_bps"] for _, r in rows] + [bar["mean_bps"], 0.0]
    lo, hi = min(means), max(means)
    pad = max((hi - lo) * 0.08, 1.0)
    lo, hi = lo - pad, hi + pad

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (right - left)

    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Each variant\'s '
            'mean a day after costs, against counting alone">'
        )
    ]
    step = 1 if hi - lo <= 12 else 2 if hi - lo <= 24 else 5 if hi - lo <= 60 else 10
    tick = int(lo // step) * step
    while tick <= hi:
        if tick >= lo:
            parts.append(
                f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 6}" '
                f'y2="{height - 24}" stroke="var(--grid)"/>'
                f'<text x="{x(tick):.1f}" y="{height - 8}" text-anchor="middle" '
                f'class="axis">{signed(tick, 0) if tick else "0"}</text>'
            )
        tick += step
    parts.append(
        f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 6}" y2="{height - 24}" '
        f'stroke="var(--baseline)" stroke-width="1.5"/>'
    )
    for i, (label, r) in enumerate(rows):
        y0 = top + i * row_h + 6
        h = row_h - 12
        a, b = x(0), x(r["mean_bps"])
        r4 = min(4.0, abs(b - a))
        if b >= a:
            d = (
                f"M{a:.1f},{y0} H{b - r4:.1f} Q{b:.1f},{y0} {b:.1f},{y0 + r4} "
                f"V{y0 + h - r4} Q{b:.1f},{y0 + h} {b - r4:.1f},{y0 + h} H{a:.1f} Z"
            )
        else:
            d = (
                f"M{a:.1f},{y0} H{b + r4:.1f} Q{b:.1f},{y0} {b:.1f},{y0 + r4} "
                f"V{y0 + h - r4} Q{b:.1f},{y0 + h} {b + r4:.1f},{y0 + h} H{a:.1f} Z"
            )
        years = ", ".join(
            f"{y}: {signed(v['mean_bps'])}" for y, v in r["years"].items()
        )
        tip = f"{label}: {signed(r['mean_bps'])} bps a day, t {signed(r['t'])}; {years}"
        parts.append(
            f'<g class="bar"><title>{html.escape(tip)}</title>'
            f'<rect x="0" y="{y0 - 6}" width="{width}" height="{row_h}" fill="transparent"/>'
            f'<path d="{d}" fill="var(--series-1)"/>'
            f'<text x="{left - 10}" y="{y0 + h / 2 + 4}" text-anchor="end" '
            f'class="label">{html.escape(label)}</text>'
            f'<text x="{values_x}" y="{y0 + h / 2 + 4}" class="value">'
            f"{signed(r['mean_bps'])} bps, t {signed(r['t'])}</text></g>"
        )
    xb = x(bar["mean_bps"])
    parts.append(
        f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{top - 6}" y2="{height - 24}" '
        f'stroke="var(--band)" stroke-width="2" stroke-dasharray="5 4"/>'
        f'<text x="{xb:.1f}" y="{top - 12}" text-anchor="middle" class="axis">'
        f"counting alone {signed(bar['mean_bps'])}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _table(rows: list[tuple[str, dict]], bar: dict) -> tuple[str, str]:
    """The year table's head and body: each row's years, all of them, t and days, then
    counting alone; cells coloured by gain and loss on one scale."""
    years = list(bar["years"])
    cells = [r["years"][y]["mean_bps"] for _, r in rows for y in years]
    cells += [bar["years"][y]["mean_bps"] for y in years]
    scale = max([abs(c) for c in cells] + [5.0])
    body_rows = []
    for label, r in [*rows, ("counting alone (the comparison)", bar)]:
        body_rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            + "".join(_cell(r["years"][y]["mean_bps"], scale) for y in years)
            + _cell(r["mean_bps"], scale)
            + f'<td class="num">{signed(r["t"])}</td><td class="num">{r["days"]:,}</td></tr>'
        )
    head = "".join(f"<th class='num'>{y}</th>" for y in years)
    return head, "".join(body_rows)


def page(test: str, store: dict) -> str:
    bar = store["count_only"]
    variants = store["variants"]
    rows = [
        (f"{v['n']} · {settings_text(v['settings'])}", v["result"]) for v in variants
    ]
    head, body = _table(rows, bar)
    chart_rows = [
        (chart_label(str(v["n"]), v["settings"]), v["result"]) for v in variants
    ]
    best = max(variants, key=lambda v: v["result"]["mean_bps"]) if variants else None
    summary = [
        f"Variants tried: <b>{len(variants)} of {LIMIT}</b>.",
        (
            f"The bar to clear: counting alone, <b>{signed(bar['mean_bps'])} bps a day</b> "
            f"after costs (t {signed(bar['t'])})."
        ),
    ]
    if best:
        r = best["result"]
        summary.append(
            f"Highest so far: variant {best['n']}, {html.escape(settings_text(best['settings']))}: "
            f"<b>{signed(r['mean_bps'])} bps a day</b> (t {signed(r['t'])}), "
            f"{signed(r['mean_bps'] - bar['mean_bps'])} against counting alone. By the "
            "pre-registration the highest goes to the holdout, whatever it is."
        )
    summary.append(
        "Means are bps a day after costs of 2 bps a day (1 bp a round trip on each side). "
        "The pass, t &ge; 2.50, applies only to the holdout, run once after the choice."
    )
    chart = _bars(chart_rows, bar) if rows else "<p>No variant yet.</p>"
    return PAGE.format(
        title=f"{html.escape(test)} development",
        heading=f"{html.escape(test)}: the variants on the development years",
        updated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        summary="".join(f"<li>{s}</li>" for s in summary),
        chart_title="Mean a day after costs, 2018-2021 (bps)",
        chart=chart,
        table_title="Each development year (bps a day after costs)",
        head=head,
        rows=body,
        note="A variant worth choosing beats counting alone in most years, not in one.",
    )


def holdout_page(test: str, result: dict) -> str:
    """The holdout, run once: the chosen variant against the threshold and counting alone."""
    bar = result["count_only"]
    v = result["variant"]
    label = f"variant {v['variant']} · {settings_text(v)}"
    head, body = _table([(label, result)], bar)
    verdict = "Passed" if result["passes"] else "Failed"
    mark = "&#10003;" if result["passes"] else "&#10007;"
    summary = [
        (
            f"<b>{mark} {verdict}:</b> t {signed(result['t'])} against the pre-registered "
            f"{result['threshold']:.2f}, one-sided, over {result['days']:,} days."
        ),
        (
            f"The chosen variant, {html.escape(label)}: <b>{signed(result['mean_bps'])} bps a "
            f"day</b> after costs; counting alone {signed(bar['mean_bps'])} "
            f"(t {signed(bar['t'])})."
        ),
        (
            f"Run once, at commit <code>{html.escape(result['commit'][:7])}</code>; "
            "a second run is refused."
        ),
    ]
    return PAGE.format(
        title=f"{html.escape(test)} holdout",
        heading=f"{html.escape(test)}: the holdout, 2022 to 9 October 2026",
        updated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        summary="".join(f"<li>{s}</li>" for s in summary),
        chart_title="Mean a day after costs, 2022-2026 (bps)",
        chart=_bars([(chart_label(f"variant {v['variant']}", v), result)], bar),
        table_title="Each holdout year (bps a day after costs)",
        head=head,
        rows=body,
        note="The holdout is the test; development years only chose the variant.",
    )


def holdout_section(test: str, result: dict, day: str) -> str:
    """The ledger's record of a holdout, as N1's."""
    bar = result["count_only"]
    v = result["variant"]

    def line(label: str, r: dict) -> str:
        years = " / ".join(signed(y["mean_bps"]) for y in r["years"].values())
        return (
            f"| {label} | {r['days']:,} | {signed(r['mean_bps'])} bps | {signed(r['t'])} | "
            f"{years} |"
        )

    verdict = "passed" if result["passes"] else "failed"
    sign = "\u2265" if result["passes"] else "<"
    return (
        f"\n## {test}'s holdout (run once, {day}, commit {result['commit'][:7]})\n\n"
        f"`data/news_reader/holdout_{test}.json`; a second run is refused.\n\n"
        "| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |\n"
        "|---|---|---|---|---|\n"
        + line(f"Variant {v['variant']}: {settings_text(v)}", result)
        + "\n"
        + line("The count alone (the comparison)", bar)
        + f"\n\n**{test} {verdict} on its holdout: t {signed(result['t'])} {sign} "
        f"{result['threshold']:.2f}.**\n"
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ef; --grid: #e4e3df; --baseline: #c3c2b7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77766f;
  --series-1: #2a78d6; --band: #8d8c86;
  --pos: #2a78d6; --neg: #e34948; --mid: #f0efec;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322; --grid: #33332f; --baseline: #4a4a45;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-1: #3987e5; --band: #8d8c86;
    --pos: #3987e5; --neg: #e66767; --mid: #383835;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #232322; --grid: #33332f; --baseline: #4a4a45;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --series-1: #3987e5; --band: #8d8c86;
  --pos: #3987e5; --neg: #e66767; --mid: #383835;
}}
body {{ margin: 0; background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 28px 0 8px; }}
.updated {{ color: var(--text-muted); font-size: 12px; margin: 0 0 12px; }}
ul.summary {{ padding-left: 18px; color: var(--text-secondary); }}
svg {{ width: 100%; height: auto; display: block; }}
svg .axis {{ font-size: 11px; fill: var(--text-muted); }}
svg .label {{ font-size: 12px; fill: var(--text-secondary); }}
svg .value {{ font-size: 12px; fill: var(--text-primary);
  font-variant-numeric: tabular-nums; }}
svg .bar:hover path {{ opacity: 0.8; }}
.tables {{ overflow-x: auto; }}
table {{ border-collapse: separate; border-spacing: 2px; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 4px 8px; }}
th {{ color: var(--text-secondary); font-weight: 600; border-bottom: 1px solid var(--grid); }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.num {{ border-radius: 3px; }}
.note {{ color: var(--text-muted); font-size: 12px; }}
</style>
</head>
<body>
<main>
<h1>{heading}</h1>
<p class="updated">Updated {updated}</p>
<ul class="summary">{summary}</ul>
<h2>{chart_title}</h2>
{chart}
<h2>{table_title}</h2>
<div class="tables"><table><thead><tr><th>Variant</th>{head}<th class="num">All</th>
<th class="num">t</th><th class="num">Days</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">Blue cells earned, red cells lost; the stronger the colour, the larger the
amount. {note}</p>
</main>
</body>
</html>
"""


def record(test: str, result: dict, day: str) -> None:
    """The holdout's record at the end of the ledger."""
    LEDGER.write_text(
        LEDGER.read_text().rstrip("\n") + "\n" + holdout_section(test, result, day)
    )


def holdout(test: str, harness, efe_model) -> None:
    """The chosen variant on the holdout, once: harness.run_holdout checks the choice is
    committed and refuses a second run. The settings come from the committed choice. The
    ledger's record is written before the page, so a page that fails loses nothing."""
    if test not in ("N2", "N3", "N4"):
        raise SystemExit("dev_report.py runs N2, N3 and N4")
    DEV.mkdir(parents=True, exist_ok=True)
    out = DEV / f"{test}_holdout.html"
    lock = harness.LOCKS / f"holdout_{test}.json"
    if lock.exists():
        # Run already: its result is in the lock. Write the page, and the ledger's record if
        # it is missing, from there; nothing runs again.
        result = json.loads(lock.read_text())
        result["threshold"] = harness.THRESHOLDS[test]
        out.write_text(holdout_page(test, result))
        if f"## {test}'s holdout" not in LEDGER.read_text():
            day = datetime.fromtimestamp(lock.stat().st_mtime, UTC).strftime(
                "%-d %B %Y"
            )
            record(test, result, day)
        print(f"{test}'s holdout has been run; its page is written again: {out}")
        return
    chosen = harness.VARIANTS / f"{test}_chosen.json"
    if not chosen.exists():
        raise SystemExit(f"write and commit {chosen} with the chosen variant first")
    settings = json.loads(chosen.read_text())
    print(f"{test}'s holdout, once: {settings_text(settings)} ...", flush=True)
    result = stored(
        harness.run_holdout(
            test, efe_model.make(settings["components"], settings["penalty"])
        )
    )
    result["threshold"] = harness.THRESHOLDS[test]
    record(test, result, datetime.now(UTC).strftime("%-d %B %Y"))
    out.write_text(holdout_page(test, result))
    verdict = "passed" if result["passes"] else "failed"
    print(
        f"{test} {verdict}: t {signed(result['t'])}, {signed(result['mean_bps'])} bps a day"
    )
    print(f"the page: {out}")


def parse_settings(test: str, args: list[str], harness) -> dict:
    """COMPONENTS PENALTY [FEATURES] [LEVER=VALUE ...]: N3 and N4 name a feature set, N4 sets
    its levers (those left out take N3's values, and all four are written down)."""
    settings = {"components": int(args[0]), "penalty": float(args[1])}
    if test == "N2":
        if len(args) != 2:
            raise SystemExit("N2 has one feature set; leave FEATURES out")
        return settings
    if len(args) < 3:
        raise SystemExit(f"{test} needs a feature set: one of {harness.N3_FEATURES}")
    settings["features"] = args[2]
    if test == "N3":
        if len(args) != 3:
            raise SystemExit("N3 takes no levers; they are N4's")
        return settings
    levers = dict(harness.N3_LEVERS)
    for arg in args[3:]:
        lever, _, value = arg.partition("=")
        if lever not in harness.N4_LEVERS or value not in harness.N4_LEVERS[lever]:
            raise SystemExit(
                f"a lever is LEVER=VALUE from {harness.N4_LEVERS}; got {arg!r}"
            )
        levers[lever] = value
    return settings | levers


def replace_section(ledger: str, heading: str, section: str) -> str:
    """The ledger with the section under `heading` replaced by `section`, up to the next
    heading, what follows kept; or `section` added at the end."""
    if heading not in ledger:
        return ledger.rstrip("\n") + "\n" + section
    start = ledger.index(heading)
    end = ledger.find("\n## ", start + len(heading))
    rest = "" if end == -1 else ledger[end:]
    return ledger[:start].rstrip("\n") + "\n" + section + rest


# Variants that may run on the holdout years beside the test's own, as exploration.
EXPLORE = {"N4": (6, 10, 8)}  # amendment 6
FAMILY_BAR = {"N4": 2.96}  # four tries sharing N4's 0.625%, one-sided


def explore_section(test: str, explored: dict, official: dict) -> str:
    """The ledger's record of the exploratory runs, beside the test's own."""

    def line(label: str, r: dict) -> str:
        years = " / ".join(signed(y["mean_bps"]) for y in r["years"].values())
        return (
            f"| {label} | {r['days']:,} | {signed(r['mean_bps'])} bps | {signed(r['t'])} | "
            f"{years} |"
        )

    rows = [line(f"Variant {official['variant']['variant']} (the test)", official)]
    rows += [
        line(f"Variant {n}: {settings_text(e['settings'])}", e["result"])
        for n, e in explored.items()
    ]
    return (
        f"\n## {test}'s exploratory runs on the holdout years (amendment 6; not tests)\n\n"
        f"Each once, after the test's own. Four tries sharing {test}'s 0.625% would each "
        f"need t \u2265 {FAMILY_BAR[test]:.2f}; a variant other than the test's counts only "
        "after a forward test in the shadow record.\n\n"
        "| | Days | Mean a day | t | 2022 / 2023 / 2024 / 2025 / 2026, bps a day |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def explore(test: str, numbers: list[int], harness, efe_model) -> None:
    """Development variants on the holdout years, as exploration (amendment 6): only those
    the amendment names, only after the test's own holdout, each once."""
    allowed = EXPLORE.get(test, ())
    if not numbers or any(n not in allowed for n in numbers):
        raise SystemExit(
            f"amendment 6 lets {test} explore variants {allowed}, no others"
        )
    lock = harness.LOCKS / f"holdout_{test}.json"
    if not lock.exists():
        raise SystemExit(
            f"run {test}'s own holdout first: dev_report.py {test} holdout"
        )
    official = json.loads(lock.read_text())
    store = json.loads((DEV / f"{test}.json").read_text())
    path = harness.LOCKS / f"explore_{test}.json"
    explored = json.loads(path.read_text()) if path.exists() else {}
    for n in numbers:
        if str(n) in explored:
            print(f"{test}'s variant {n} has been explored; not run again")
            continue
        settings = next(v["settings"] for v in store["variants"] if v["n"] == n)
        print(f"{test}'s variant {n} on the holdout years, once: ...", flush=True)
        explored[str(n)] = {
            "settings": settings,
            "result": stored(
                harness.run(
                    test,
                    efe_model.make(settings["components"], settings["penalty"]),
                    "holdout",
                    features=settings.get("features"),
                    levers={lever: settings[lever] for lever in LEVERS},
                )
            ),
        }
        path.write_text(json.dumps(explored, indent=2))
    LEDGER.write_text(
        replace_section(
            LEDGER.read_text(),
            f"\n## {test}'s exploratory runs on the holdout years",
            explore_section(test, explored, official),
        )
    )
    bar = official["count_only"]
    label = f"variant {official['variant']['variant']} (the test)"
    rows = [(label, official)] + [
        (f"variant {n}", e["result"]) for n, e in explored.items()
    ]
    head, body = _table(rows, bar)
    out = DEV / f"{test}_explore.html"
    out.write_text(
        PAGE.format(
            title=f"{html.escape(test)} exploration",
            heading=f"{html.escape(test)}: exploration on the holdout years (not tests)",
            updated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            summary=(
                f"<li>The test is {html.escape(label)}: t {signed(official['t'])}. The others "
                f"are exploration (amendment 6): four tries would each need t &ge; "
                f"{FAMILY_BAR[test]:.2f}, and any but the test's counts only after a forward "
                "test.</li>"
            ),
            chart_title="Mean a day after costs, 2022-2026 (bps)",
            chart=_bars(rows, bar),
            table_title="Each holdout year (bps a day after costs)",
            head=head,
            rows=body,
            note="Exploration: none of these passes or fails.",
        )
    )
    for n, e in explored.items():
        r = e["result"]
        print(f"variant {n}: {signed(r['mean_bps'])} bps a day, t {signed(r['t'])}")
    print(f"the page: {out}")


REPORTED = ("N2", "N3", "N4")
PAIRS = (("N3", "N2"), ("N4", "N3"))  # N3 less N2 (N3's file), N4 less N3 (N4's file)


def _mean_t(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean() * 1e4), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))


def sides(days: pl.DataFrame) -> dict:
    """After and before costs, and each side against the mean of the day's members with
    news (the long side's excess, the short side's shortfall), in bps a day with t."""
    long, short, everyone = (days[c].to_numpy() for c in ("long", "short", "everyone"))
    return {
        "after": _mean_t(days["score"].to_numpy()),
        "before": _mean_t(long - short),
        "long": _mean_t(long - everyone),
        "short": _mean_t(everyone - short),
    }


def difference(a: pl.DataFrame, b: pl.DataFrame) -> dict:
    """a less b on the days both scored: bps a day, t, days."""
    both = a.select("day", "score").join(
        b.select("day", pl.col("score").alias("other")), on="day", how="inner"
    )
    mean, t = _mean_t((both["score"] - both["other"]).to_numpy())
    return {"mean_bps": mean, "t": t, "days": both.height}


def concentration(days: pl.DataFrame, best: int = 10) -> dict:
    """How much of the days' sum the best few carry, and the mean a day without them."""
    scores = np.sort(days["score"].to_numpy())[::-1]
    rest = scores[best:]
    return {
        "days": len(scores),
        # a share of a sum that is not positive means nothing
        "share_of_best": (
            float(scores[:best].sum() / scores.sum()) if scores.sum() > 0 else None
        ),
        "without_best_bps": float(rest.mean() * 1e4),
    }


def share(value: float | None) -> str:
    return "\u2014" if value is None else f"{value:.0%}"


def months(days: pl.DataFrame, year: int) -> list[tuple[str, int, float]]:
    """Each month of `year`: (month, days, bps a day after costs)."""
    part = days.filter(pl.col("day").dt.year() == year).with_columns(
        pl.col("day").dt.strftime("%Y-%m").alias("month")
    )
    rows = part.group_by("month").agg(
        pl.len(), (pl.col("score").mean() * 1e4).alias("bps")
    )
    return [tuple(r) for r in rows.sort("month").iter_rows()]


def reports_section(found: dict) -> str:
    """The ledger's record of the reports, after the holdouts."""
    lines = [
        "\n## Reported with the holdouts (not part of the pass)\n",
        (
            "Each choice run again on the holdout years by `dev_report.py reports`, "
            "reproducing its locked mean and days. Bps a day, t in brackets. The long side "
            "is against the mean of the day's members with news, the short side is that "
            "mean less it.\n"
        ),
        (
            "| | After costs | Before costs | Long side | Short side | Best 10 days' share "
            "| Without them |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for test, f in found["tests"].items():
        cells = [
            f"{signed(m)} ({signed(t)})"
            for m, t in (f["sides"][k] for k in ("after", "before", "long", "short"))
        ]
        c = f["concentration"]
        lines.append(
            f"| {test} | "
            + " | ".join(cells)
            + f" | {share(c['share_of_best'])} | {signed(c['without_best_bps'])} |"
        )
    lines += ["", "| Day by day | Bps a day | t | Days |", "|---|---|---|---|"]
    for (a, b), d in found["pairs"].items():
        lines.append(
            f"| {a} less {b} | {signed(d['mean_bps'])} | {signed(d['t'])} | {d['days']:,} |"
        )
    lines += ["", "2026 by month, bps a day after costs (days):", ""]
    for test, f in found["tests"].items():
        cells = ", ".join(
            f"{m[5:]} {signed(bps)} ({n})" for m, n, bps in f["months_2026"]
        )
        lines.append(f"- {test}: {cells}")
    return "\n".join(lines) + "\n"


def reports(harness, efe_model) -> None:
    """The pre-registered reports beside the pass, from each choice run again day by day;
    nothing is written unless every run reproduces its locked result."""
    found = {"tests": {}, "pairs": {}}
    series = {}
    for test in REPORTED:
        lock = harness.LOCKS / f"holdout_{test}.json"
        if not lock.exists():
            raise SystemExit(f"{test}'s holdout has not been run")
        locked = json.loads(lock.read_text())
        chosen = json.loads((harness.VARIANTS / f"{test}_chosen.json").read_text())
        print(
            f"{test}'s choice again, day by day: {settings_text(chosen)} ...",
            flush=True,
        )
        days = harness.daily(
            test,
            efe_model.make(chosen["components"], chosen["penalty"]),
            "holdout",
            features=chosen.get("features"),
            levers={lever: chosen[lever] for lever in LEVERS} if test == "N4" else None,
        )
        mean = float(days["score"].mean() * 1e4)
        if days.height != locked["days"] or abs(mean - locked["mean_bps"]) > 1e-6:
            raise SystemExit(
                f"{test} does not reproduce its holdout: {mean:+.6f} bps over {days.height} "
                f"days, locked {locked['mean_bps']:+.6f} over {locked['days']}; nothing written"
            )
        series[test] = days
        found["tests"][test] = {
            "sides": sides(days),
            "concentration": concentration(days),
            "months_2026": months(days, 2026),
        }
    for a, b in PAIRS:
        found["pairs"][(a, b)] = difference(series[a], series[b])
    DEV.mkdir(parents=True, exist_ok=True)
    for test, days in series.items():
        days.write_parquet(DEV / f"days_{test}.parquet")
    LEDGER.write_text(
        replace_section(
            LEDGER.read_text(),
            "\n## Reported with the holdouts (not part of the pass)",
            reports_section(found),
        )
    )
    out = DEV / "reports.html"
    out.write_text(reports_page(found))
    for test, f in found["tests"].items():
        after, before = f["sides"]["after"], f["sides"]["before"]
        print(
            f"{test}: after costs {signed(after[0])} (t {signed(after[1])}), before {signed(before[0])}"
        )
    for (a, b), d in found["pairs"].items():
        print(f"{a} less {b}: {signed(d['mean_bps'])} bps a day (t {signed(d['t'])})")
    print(f"the page: {out}")


def reports_page(found: dict) -> str:
    """The reports as a page: the tables, months coloured by gain and loss."""

    def row(label: str, cells: list[str]) -> str:
        return f"<tr><td>{html.escape(label)}</td>" + "".join(cells) + "</tr>"

    sides_rows = "".join(
        row(
            test,
            [
                f'<td class="num">{signed(m)} <span class="t">t {signed(t)}</span></td>'
                for m, t in (
                    f["sides"][k] for k in ("after", "before", "long", "short")
                )
            ]
            + [
                f'<td class="num">{share(f["concentration"]["share_of_best"])}</td>',
                f'<td class="num">{signed(f["concentration"]["without_best_bps"])}</td>',
            ],
        )
        for test, f in found["tests"].items()
    )
    pair_rows = "".join(
        row(
            f"{a} less {b}",
            [
                f'<td class="num">{signed(d["mean_bps"])}</td>',
                f'<td class="num">{signed(d["t"])}</td>',
                f'<td class="num">{d["days"]:,}</td>',
            ],
        )
        for (a, b), d in found["pairs"].items()
    )
    all_months = sorted(
        {m for f in found["tests"].values() for m, _, _ in f["months_2026"]}
    )
    scale = max(
        [abs(b) for f in found["tests"].values() for _, _, b in f["months_2026"]]
        + [5.0]
    )
    month_rows = "".join(
        row(
            test,
            [
                _cell({m: b for m, _, b in f["months_2026"]}.get(month, 0.0), scale)
                for month in all_months
            ],
        )
        for test, f in found["tests"].items()
    )
    body = f"""
<h2>Beside the pass (bps a day; t)</h2>
<div class="tables"><table><thead><tr><th>Test</th><th class="num">After costs</th>
<th class="num">Before costs</th><th class="num">Long side</th><th class="num">Short side</th>
<th class="num">Best 10 days' share</th><th class="num">Without them</th></tr></thead>
<tbody>{sides_rows}</tbody></table></div>
<p class="note">The long side is against the mean of the day's members with news; the short
side is that mean less it. Before costs = long side + short side.</p>
<h2>Day by day</h2>
<div class="tables"><table><thead><tr><th></th><th class="num">Bps a day</th>
<th class="num">t</th><th class="num">Days</th></tr></thead><tbody>{pair_rows}</tbody></table></div>
<h2>2026 by month (bps a day after costs)</h2>
<div class="tables"><table><thead><tr><th>Test</th>{"".join(f"<th class='num'>{m[5:]}</th>" for m in all_months)}
</tr></thead><tbody>{month_rows}</tbody></table></div>
"""
    page = PAGE.format(
        title="News reports",
        heading="The news tests: reported with the holdouts",
        updated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        summary="<li>Not part of any pass: what the pre-registrations ask to report, and a "
        "diagnostic of 2026.</li>",
        chart_title="",
        chart="",
        table_title="",
        head="",
        rows="",
        note="",
    )
    start = page.index("<h2></h2>")
    end = page.index("</main>")
    return page[:start] + body + page[end:]


def main(argv: list[str]) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import efe_model
    import harness

    if len(argv) == 2 and argv[1] == "holdout":
        holdout(argv[0], harness, efe_model)
        return
    if argv == ["reports"]:
        reports(harness, efe_model)
        return
    if len(argv) >= 2 and argv[1] == "explore":
        explore(argv[0], [int(n) for n in argv[2:]], harness, efe_model)
        return
    if len(argv) < 3:
        raise SystemExit(
            "usage: dev_report.py TEST COMPONENTS PENALTY [FEATURES] [LEVER=VALUE ...]"
            " | TEST holdout"
        )
    test = argv[0]
    if test not in ("N2", "N3", "N4"):
        raise SystemExit("dev_report.py runs N2, N3 and N4")
    chosen = harness.VARIANTS / f"{test}_chosen.json"
    if chosen.exists():
        raise SystemExit(
            f"{test}'s choice is made ({chosen}): no more variants are run"
        )
    settings = parse_settings(test, argv[1:], harness)
    DEV.mkdir(parents=True, exist_ok=True)
    store_path = DEV / f"{test}.json"
    store = (
        json.loads(store_path.read_text())
        if store_path.exists()
        else {"count_only": None, "variants": []}
    )
    out = DEV / f"{test}.html"
    try:
        n, again = check_new(store, LEDGER.read_text(), test, settings, harness.SCORING)
    except AlreadyTried:
        out.write_text(page(test, store))
        raise
    if (
        store["count_only"] is None
        or store["count_only"].get("scoring") != harness.SCORING
    ):
        print("counting alone, the comparison, under the current rule ...", flush=True)
        store["count_only"] = stored(harness.run(test, None, count_only=True))
    again_text = " again" if again else ""
    print(f"{test} variant {n}{again_text}: {settings_text(settings)} ...", flush=True)
    result = stored(
        harness.run(
            test,
            efe_model.make(settings["components"], settings["penalty"]),
            features=settings.get("features"),
            levers={lever: settings[lever] for lever in LEVERS}
            if test == "N4"
            else None,
        )
    )
    run_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    entry = {"n": n, "settings": settings, "run_utc": run_utc, "result": result}
    if again:
        before = next(v for v in store["variants"] if v["n"] == n)
        row = ledger_row(
            n, test, run_utc, settings, result, store["count_only"], before["result"]
        )
        LEDGER.write_text(replace_row(LEDGER.read_text(), test, n, row))
        store["variants"] = [entry if v["n"] == n else v for v in store["variants"]]
    else:
        row = ledger_row(n, test, run_utc, settings, result, store["count_only"])
        LEDGER.write_text(add_row(LEDGER.read_text(), row))
        store["variants"].append(entry)
    store_path.write_text(json.dumps(store, indent=2))
    out.write_text(page(test, store))
    print(row)
    print(f"the page: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
