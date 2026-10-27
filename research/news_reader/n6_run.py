"""Efe's development runs of N6: one variant on the development years, beside its placebos.

Efe runs it, with the settings he chose (any of n6_model.Settings, as name=value):

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_run.py v1 dim=32 loss=mse

For each development year (n6_harness.DEVELOPMENT), one model a seed (SEEDS) is trained on
the member-days before it, less the gap, and their forecasts are averaged. The same is done
for the two placebos: the headlines dealt again within each day, and the context alone. Each
is scored with N4's variant 2's portfolio. N4's variant 2 on the same years, from
data/news_reader/dev/N4.json, stands beside them. A variant passes the placebo gate (the
decisions page, 4A) when it makes more a day over the four years than both placebos.

Output: data/news_reader/n6/dev/<name>.json and <name>.html, and a summary printed. As
pre-registration N6 asks, each variant is written in LEDGER.md before it runs, at most ten,
a name and settings once (n6_ledger.py). Then the choice, by the pre-registered rule:

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_run.py choose

Before the first run, a check of speed and memory that shows no score:

    PYTHONPATH=src data/llmenv/bin/python research/news_reader/n6_run.py smoke
"""

import ast
import html
import json
import sys
import time
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import n6_harness as nh
import n6_ledger
import n6_model as nm
from dev_report import PAGE, signed

SEEDS = (0, 1, 2, 3, 4)
OUT = nh.N6 / "dev"
N4_DEV = Path("data/news_reader/dev/N4.json")
LABELS = {
    "model": "The variant",
    "dealt": "Placebo: headlines dealt within each day",
    "silent": "Placebo: the context alone",
}


def parse(args: list[str]) -> tuple[str, nm.Settings]:
    """A run's name and its settings from name=value pairs."""
    if not args or "=" in args[0]:
        raise SystemExit("usage: n6_run.py NAME [setting=value ...]")
    defaults = nm.Settings()
    names = [f.name for f in fields(nm.Settings)]
    given = {}
    for arg in args[1:]:
        key, _, value = arg.partition("=")
        if key not in names:
            raise SystemExit(f"unknown setting {key!r}; settings: {', '.join(names)}")
        given[key] = type(getattr(defaults, key))(value)
    return args[0], nm.Settings(**given)


def gpu_store(path: Path = nh.N6 / "headlines.npy", chunk: int = 32_768):
    """N6's eyes on the GPU, in float16, copied a chunk at a time."""
    source = np.load(path, mmap_mode="r")
    store = torch.empty(source.shape, dtype=torch.float16, device="cuda")
    for start in range(0, len(source), chunk):
        store[start : start + chunk] = torch.from_numpy(
            np.array(source[start : start + chunk])
        )
    return store


def n4_years() -> dict:
    """N4's variant 2 on the development years, as stored by dev_report.py."""
    if not N4_DEV.exists():
        return {}

    def read(value):  # dev_report kept some results as their repr, others as JSON
        return ast.literal_eval(value) if isinstance(value, str) else value

    for variant in json.loads(N4_DEV.read_text())["variants"]:
        if str(variant["n"]) == "2":
            result = read(variant["result"])
            years = read(result["years"])
            out = {int(y): (v["mean_bps"], v["t"], v["days"]) for y, v in years.items()}
            out["all"] = (result["mean_bps"], result["t"], result["days"])
            return out
    return {}


def run(name: str, s: nm.Settings, a=None, store=None, seeds=SEEDS) -> dict:
    """The variant and its placebos on every development year."""
    a = a if a is not None else nh.load()
    store = store if store is not None else gpu_store()
    arrays = {"model": a, "dealt": None, "silent": nh.silent(a)}
    days = {label: [] for label in LABELS}
    years = {}
    for year in nh.DEVELOPMENT:
        trained, validated = nh.fold(a, year)
        arrays["dealt"] = nh.deal(a, seed=year)
        years[year] = {}
        for label in LABELS:
            forecast = nm.ensemble(arrays[label], trained, validated, s, seeds, store)
            scored = nh.score(a, validated, forecast)
            days[label] += scored
            years[year][label] = nh.summary(scored)
        print(
            f"{year}: "
            + ", ".join(f"{k} {v['bps']:+.2f} bps" for k, v in years[year].items()),
            flush=True,
        )
    total = {label: nh.summary(days[label]) for label in LABELS}
    return {
        "name": name,
        "settings": asdict(s),
        "seeds": list(seeds),
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "years": {str(y): v for y, v in years.items()},
        "all": total,
        "gate": total["model"]["bps"]
        > max(total["dealt"]["bps"], total["silent"]["bps"]),
        "n4": {str(k): v for k, v in n4_years().items()},
    }


def page(result: dict) -> str:
    """The run as a page: each year's bps a day and t for the variant, its placebos and N4."""
    head = "".join(f"<th class='num'>{y}</th>" for y in [*result["years"], "All"])

    def cells(per_year: dict, total: dict | None) -> str:
        out = ""
        for value in [*per_year, total]:
            if value is None:
                out += "<td class='num'>—</td>"
            else:
                bps, t = value
                out += f"<td class='num'>{signed(bps)} <span class='t'>t {signed(t)}</span></td>"
        return out

    rows = ""
    for label, text in LABELS.items():
        per_year = [(v[label]["bps"], v[label]["t"]) for v in result["years"].values()]
        total = (result["all"][label]["bps"], result["all"][label]["t"])
        rows += f"<tr><td>{html.escape(text)}</td>{cells(per_year, total)}</tr>"
    n4 = result["n4"]
    if n4:
        per_year = [tuple(n4[y][:2]) if y in n4 else None for y in result["years"]]
        rows += (
            f"<tr><td>N4's variant 2</td>{cells(per_year, tuple(n4['all'][:2]))}</tr>"
        )
    verdict = "passes" if result["gate"] else "does not pass"
    settings = ", ".join(f"{k}={v}" for k, v in result["settings"].items())
    model = result["all"]["model"]
    summary = [
        f"Settings: {settings}; seeds {result['seeds']}.",
        (
            f"Over {model['days']:,} days the variant made {signed(model['bps'])} bps a "
            f"day after costs; it {verdict} the placebo gate (more than both placebos)."
        ),
    ]
    body = f"""
<h2>Bps a day after costs (t), each development year</h2>
<div class="tables"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody>
</table></div>
<p class="note">Scored with N4's variant 2's portfolio: a tenth a side, equal weights, ties in,
at least 20 members, 2 bps a day. Each year's models are trained on the member-days before it,
less {nh.GAP} sessions; each row is the mean forecast of {len(result["seeds"])} seeds.</p>
"""
    out = PAGE.format(
        title=f"N6 {result['name']}",
        heading=f"N6, development run {html.escape(result['name'])}",
        updated=result["run_utc"] + " UTC",
        summary="".join(f"<li>{html.escape(s)}</li>" for s in summary),
        chart_title="",
        chart="",
        table_title="",
        head="",
        rows="",
        note="",
    )
    start, end = out.index("<h2></h2>"), out.index("</main>")
    return out[:start] + body + out[end:]


def smoke() -> None:
    """One seed, one epoch, on 2017 with 2016 behind it: time and memory, no score."""
    began = time.time()
    a = nh.load()
    store = gpu_store()
    loaded = time.time()
    when = a.days["day"].dt.year().to_numpy()
    trained, validated = (when == 2016) & ~np.isnan(a.y), when == 2017
    s = nm.Settings(epochs=1)
    forecast = nm.ensemble(a, trained, validated, s, [0], store)
    print(
        f"loaded in {loaded - began:.0f} s; one epoch on 2016 and 2017's forecasts in "
        f"{time.time() - loaded:.0f} s; {np.isfinite(forecast).mean():.0%} finite; GPU memory "
        f"at most {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    )


def choose() -> None:
    """The pre-registered choice, written to variants/N6_chosen.json once."""
    if n6_ledger.CHOSEN.exists():
        raise SystemExit(f"{n6_ledger.CHOSEN} is written already")
    ledger = n6_ledger.LEDGER.read_text()
    finished = [
        r["name"]
        for r in n6_ledger.rows(ledger)
        if r["result"] not in (n6_ledger.RUNNING, n6_ledger.UNFINISHED)
    ]
    results = {n: json.loads((OUT / f"{n}.json").read_text()) for n in finished}
    chosen = n6_ledger.choose(ledger, results)
    n6_ledger.CHOSEN.write_text(json.dumps(chosen, indent=1) + "\n")
    print(
        f"chosen: variant {chosen['variant']}, {chosen['name']}, "
        f"{chosen['development']['bps']:+.2f} bps a day; {n6_ledger.CHOSEN}"
    )


def main(argv: list[str]) -> None:
    if argv == ["smoke"]:
        return smoke()
    if argv == ["choose"]:
        return choose()
    name, s = parse(argv)
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    ledger, n = n6_ledger.start(n6_ledger.LEDGER.read_text(), name, asdict(s), when)
    n6_ledger.LEDGER.write_text(ledger)
    print(f"N6's variant {n}, {name}, written in {n6_ledger.LEDGER} before it runs")
    try:
        result = run(name, s)
    except BaseException:
        text = n6_ledger.unfinished(n6_ledger.LEDGER.read_text(), n)
        n6_ledger.LEDGER.write_text(text)
        raise
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    (OUT / f"{name}.html").write_text(page(result))
    n6_ledger.LEDGER.write_text(
        n6_ledger.finish(n6_ledger.LEDGER.read_text(), n, result)
    )
    total = result["all"]
    print(
        f"{name}: {total['model']['bps']:+.2f} bps a day (t {total['model']['t']:+.2f}); "
        f"dealt {total['dealt']['bps']:+.2f}, context alone {total['silent']['bps']:+.2f}; "
        f"gate {'passed' if result['gate'] else 'not passed'}; {OUT / name}.html"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
