"""Efe's development runs of the news tests: fit a variant, count it, show every variant.

Efe runs it, with the settings he chose:

    uv run python research/news_reader/dev_report.py N2 16 10.0
    uv run python research/news_reader/dev_report.py N3 16 10.0 base-last

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
"""

import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

LEDGER = Path("research/news_reader/LEDGER.md")
DEV = Path("data/news_reader/dev")
LIMIT = 10
VARIANTS_HEADER = "| # | Test | Run (UTC) | Settings (`efe_model.make`) |"


def signed(x: float, digits: int = 2) -> str:
    """+1.23 or −1.23, with a true minus sign, as the ledger writes numbers."""
    return f"{x:+.{digits}f}".replace("-", "−")


def settings_text(settings: dict) -> str:
    text = f"components {settings['components']}, penalty {settings['penalty']}"
    if settings.get("features"):
        text += f", features {settings['features']}"
    return text


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


def check_new(
    store: dict, ledger: str, test: str, settings: dict, scoring: str
) -> tuple[int, bool]:
    """The variant's number within its test, and whether it is scored again (settings tried
    under an earlier rule). Refuses a repeat under the same rule, or an eleventh."""
    for v in store["variants"]:
        if v["settings"] == settings:
            if v["result"].get("scoring") == scoring:
                raise SystemExit(
                    f"{test}: {settings_text(settings)} is variant {v['n']} already; "
                    f"its result is on {DEV / (test + '.html')}"
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
    width, left, right, values_x, row_h, top = 1000, 230, 790, 805, 34, 30
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


def page(test: str, store: dict) -> str:
    bar = store["count_only"]
    variants = store["variants"]
    rows = [
        (f"{v['n']} · {settings_text(v['settings'])}", v["result"]) for v in variants
    ]
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
    chart = _bars(rows, bar) if rows else "<p>No variant yet.</p>"
    return PAGE.format(
        test=html.escape(test),
        updated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        summary="".join(f"<li>{s}</li>" for s in summary),
        chart=chart,
        head="".join(f"<th class='num'>{y}</th>" for y in years),
        rows="".join(body_rows),
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{test} development</title>
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
<h1>{test}: the variants on the development years</h1>
<p class="updated">Updated {updated}</p>
<ul class="summary">{summary}</ul>
<h2>Mean a day after costs, 2018-2021 (bps)</h2>
{chart}
<h2>Each development year (bps a day after costs)</h2>
<div class="tables"><table><thead><tr><th>Variant</th>{head}<th class="num">All</th>
<th class="num">t</th><th class="num">Days</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">Blue cells earned, red cells lost; the stronger the colour, the larger the
amount. A variant worth choosing beats counting alone in most years, not in one.</p>
</main>
</body>
</html>
"""


def main(argv: list[str]) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import efe_model
    import harness

    if len(argv) not in (3, 4):
        raise SystemExit("usage: dev_report.py TEST COMPONENTS PENALTY [FEATURES]")
    test = argv[0]
    if test not in ("N2", "N3"):
        raise SystemExit("dev_report.py runs N2 and N3, the tests still open")
    settings = {"components": int(argv[1]), "penalty": float(argv[2])}
    if test == "N3":
        if len(argv) != 4:
            raise SystemExit(f"N3 needs a feature set: one of {harness.N3_FEATURES}")
        settings["features"] = argv[3]
    elif len(argv) == 4:
        raise SystemExit("N2 has one feature set; leave FEATURES out")
    DEV.mkdir(parents=True, exist_ok=True)
    store_path = DEV / f"{test}.json"
    store = (
        json.loads(store_path.read_text())
        if store_path.exists()
        else {"count_only": None, "variants": []}
    )
    n, again = check_new(store, LEDGER.read_text(), test, settings, harness.SCORING)
    if (
        store["count_only"] is None
        or store["count_only"].get("scoring") != harness.SCORING
    ):
        print("counting alone, the comparison, under the current rule ...", flush=True)
        store["count_only"] = harness.run(test, None, count_only=True)
    again_text = " again" if again else ""
    print(f"{test} variant {n}{again_text}: {settings_text(settings)} ...", flush=True)
    result = harness.run(
        test,
        efe_model.make(settings["components"], settings["penalty"]),
        features=settings.get("features"),
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
    out = DEV / f"{test}.html"
    out.write_text(page(test, store))
    print(row)
    print(f"the page: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
