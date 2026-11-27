"""KUTBIT's third variant (decision 0028): the skill test's ridge given its sector's move,
judged as the decision wrote before it ran.

Only the inputs differ from the skill test's ridge: its 971, and then the 120-minute window of
the channel `sector` (sector_data.py), 1,091 in all. Beside it, the same ridge with each
instrument's least related sector, `wrong`, is shown but is not a gate.

A run, on the skill test's random walks (`placebo`) or once on the real prices (`measure`):
- The skill test's ridge is fitted again as the skill test fitted it (net_harness.py), and it
  must give the skill test's own numbers, or the run stops.
- One pass over 2016-2019 sums every product the new ridges need: of the 971 inputs and the
  windows of `sector` and `wrong`, with `z` and with `z` shuffled between the instruments of
  each minute.
- For each horizon, the new ridge's penalty is chosen on 2019 as the skill test's was: by the
  correlation of the forecasts of its fit on 2016-2018. Then the new ridge, the ridge with the
  wrong sector and the new ridge on shuffled labels are fitted on 2016-2019 with the penalties
  kept, and each one's thresholds are set on its forecasts of 2019.
- Every forecaster gets the skill test's measures on 2020-2021, and its strongest forecasts
  their return after each signal's own cost and after twice that.
- The placebo's rule: no forecaster may reach |t| PLACEBO_T at any horizon in any measure.
- The gates: the new ridge's day-by-day margin over the skill test's ridge, and over itself on
  shuffled labels, in each day's correlation averaged over SCORED, each at a t of PROOF; and,
  at one of SCORED at least, its strongest 1% earning more than their own cost at a t of
  PROOF, clustered by day. Its margin over the wrong sector is shown.

`measure` waits for a placebo that passed on today's code, and runs only once. Each record
keeps the SHA-256 of CODE. The records and the page, sector.html, are in DATA/kutbit/sector.
Eight threads keep the ridge's solves fast; on many more they run a hundred times slower.

    OMP_NUM_THREADS=8 uv run python research/kutbit/sector_harness.py placebo|measure [DATA]
"""

import hashlib
import html
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import costs
import net_data
import net_harness
import numpy as np
import sector_data
import skill
from labels import HORIZONS

from qfin_alphaguard.kutbit.features import SCALARS, WINDOW

DATA = Path("data")
HERE = Path(__file__).parent
SCORED, PROOF, PLACEBO_T = net_harness.SCORED, net_harness.PROOF, skill.PLACEBO_T
LEARN_YEARS = (*skill.FIT_YEARS, skill.CHOOSE_YEAR)  # 2016-2019
BASE = net_data.WINDOW_INPUTS + len(SCALARS)  # the skill test's 971 inputs
WIDE = BASE + 2 * WINDOW  # the 971, then the windows of `sector` and `wrong`
COLUMNS = {  # each new ridge's inputs among the WIDE
    "sector": np.r_[0:BASE, BASE : BASE + WINDOW],
    "wrong": np.r_[0:BASE, BASE + WINDOW : WIDE],
}
TARGETS = {  # z, then z shuffled between the instruments of each minute
    "z": np.arange(len(HORIZONS)),
    "shuffled": np.arange(len(HORIZONS), 2 * len(HORIZONS)),
}
CODE = (
    "sector_harness.py",
    "sector_data.py",
    "net_harness.py",
    "net_data.py",
    "skill.py",
    "costs.py",
)
NAMES = {
    "sector": "The ridge with its sector",
    "ridge": "The skill test's ridge",
    "shuffled": "The ridge with its sector, on shuffled labels",
    "wrong": "The ridge with the wrong sector",
    "habit": "Known pattern: habit",
    "reversal": f"Known pattern: the last {skill.REVERSAL} minutes reversed",
}


# --- the sums -------------------------------------------------------------------------------


def inputs(net: dict, sector: dict, rows: np.ndarray) -> np.ndarray:
    """The WIDE inputs of `rows`: the skill test's 971, then the windows of `sector` and of
    `wrong`, each the WINDOW minutes up to the row, oldest first, zeros before the open."""
    windows = [
        net_data.windows_of(sector[kind][:, None], net["minute"], rows).reshape(
            len(rows), WINDOW
        )
        for kind in sector_data.KINDS
    ]
    return np.concatenate([net_data.inputs_of(net, rows), *windows], axis=1)


def empty() -> skill.Moments:
    targets = len(TARGETS["z"]) + len(TARGETS["shuffled"])
    return skill.Moments(
        0,
        np.zeros(WIDE),
        np.zeros((WIDE, WIDE)),
        np.zeros((WIDE, targets)),
        np.zeros(targets),
    )


def part(m: skill.Moments, columns: np.ndarray, targets: np.ndarray) -> skill.Moments:
    """The sums one ridge needs: those of its inputs, and of their products with its
    targets."""
    return skill.Moments(
        m.n,
        m.s[columns],
        m.q[np.ix_(columns, columns)],
        m.r[np.ix_(columns, targets)],
        m.t[targets],
    )


def year_of(ordinal: int) -> int:
    return date.fromordinal(int(ordinal)).year


def rows_in(table: dict, year: int) -> np.ndarray:
    """The positions of `table`'s rows of `year`."""
    first, last = date(year, 1, 1).toordinal(), date(year, 12, 31).toordinal()
    return np.flatnonzero((table["day"] >= first) & (table["day"] <= last))


def sums(net: dict, sector: dict) -> dict[int, skill.Moments]:
    """Each learning year's sums, day by day: of the WIDE inputs, and of their products with
    `z` and with `z` shuffled between the instruments of each minute over all of `net`."""
    dealt = net_harness.shuffled(net["z"], net["day"], net["minute"])
    target = np.concatenate([net["z"], dealt], axis=1)
    found = {}
    for rows in net_harness.by_day(net):
        year = year_of(net["day"][rows[0]])
        found.setdefault(year, empty()).add(inputs(net, sector, rows), target[rows])
    return found


def fitted(m: skill.Moments, columns, targets, choice: dict) -> skill.Ridge:
    """The ridge on `columns` for `targets`, each horizon at the penalty `choice` keeps."""
    models = [skill.fit(part(m, columns, targets), p) for p in skill.PENALTIES]
    return skill.ridge_of(models, choice)


def forecasts(net: dict, sector: dict, models: dict, rows: np.ndarray) -> dict:
    """Each model's forecasts of `rows`, a day at a time; a model is (its columns, a Ridge or a
    list of them)."""
    out = {name: [] for name in models}
    for day in net_harness.by_day({"day": net["day"][rows]}):
        x = inputs(net, sector, rows[day])
        views = {}  # each set of columns once, in float64 as a Ridge reads it
        for name, (columns, model) in models.items():
            if id(columns) not in views:
                views[id(columns)] = x[:, columns].astype(np.float64)
            view = views[id(columns)]
            if isinstance(model, list):
                out[name].append(
                    np.stack([m.intercept + view @ m.beta for m in model], axis=2)
                )
            else:
                out[name].append(model.intercept + view @ model.beta)
    return {name: np.concatenate(found) for name, found in out.items()}


# --- a run ----------------------------------------------------------------------------------


def run(data: Path, source: str, log=print) -> dict:
    """Decision 0028's run on `source`: the ridges, the choice on 2019, the measures on
    2020-2021, the placebo's rule and the gates."""
    started = time.time()
    ridge = net_harness.ridge_again(data, source, log)
    measured = net_data.load(data, source, skill.MEASURE_YEARS)
    cost = net_harness.signal_costs(data, measured)
    results = {
        "ridge": net_harness.measures_of(
            ridge["forecast"], measured, lambda h: ridge["thresholds"][h], cost
        )
    }
    for i, name in enumerate(skill.PATTERNS):
        p = np.repeat(measured["patterns"][:, i : i + 1], len(HORIZONS), axis=1)
        results[name] = net_harness.measures_of(
            p, measured, lambda h, n=name: ridge["thresholds"][n], cost
        )
    differ = net_harness.as_skill_test(data, source, results, ridge["choice"])
    if differ:
        raise SystemExit(
            "the ridge fitted again does not give the skill test's numbers, so the run stops: "
            + "; ".join(differ[:5])
        )
    log("the ridge and the known patterns give the skill test's numbers")

    net = net_data.load(data, source, LEARN_YEARS)
    sector = sector_data.load(data, source, LEARN_YEARS)
    by_year = sums(net, sector)
    early = sum((by_year[y] for y in skill.FIT_YEARS), empty())
    final = early + by_year[skill.CHOOSE_YEAR]
    log(f"the sums of 2016-2019 ({time.time() - started:.0f} s)")
    columns, z = COLUMNS["sector"], TARGETS["z"]
    tries = {
        "early": (
            columns,
            [skill.fit(part(early, columns, z), p) for p in skill.PENALTIES],
        ),
        "final": (
            columns,
            [skill.fit(part(final, columns, z), p) for p in skill.PENALTIES],
        ),
    }
    chosen = rows_in(net, skill.CHOOSE_YEAR)
    seen = forecasts(net, sector, tries, chosen)
    choice, limits = skill.choose(
        seen["early"], seen["final"], net["patterns"][chosen], net["z"][chosen]
    )
    models = {
        "sector": (columns, skill.ridge_of(tries["final"][1], choice)),
        "wrong": (COLUMNS["wrong"], fitted(final, COLUMNS["wrong"], z, choice)),
        "shuffled": (columns, fitted(final, columns, TARGETS["shuffled"], choice)),
    }
    on_2019 = forecasts(
        net, sector, {k: models[k] for k in ("wrong", "shuffled")}, chosen
    )
    thresholds = {"sector": {h: limits[h] for h in HORIZONS}}
    for name, p in on_2019.items():
        thresholds[name] = net_harness.thresholds_by_horizon(p)
    learned = final.n  # the rows the final fits learned from
    del net, sector, by_year, seen, on_2019
    log(f"penalties chosen on 2019 ({time.time() - started:.0f} s)")

    sector = sector_data.load(data, source, skill.MEASURE_YEARS)
    p = forecasts(measured, sector, models, np.arange(len(measured["day"])))
    p["ridge"] = ridge["forecast"]
    for name in ("sector", "shuffled", "wrong"):
        results[name] = net_harness.measures_of(
            p[name], measured, lambda h, n=name: thresholds[n][h], cost
        )
    order = ("sector", "ridge", "shuffled", "wrong", *skill.PATTERNS)
    results = {name: results[name] for name in order}
    checks = skill.checks(results)
    log(f"measured on 2020-2021 ({time.time() - started:.0f} s)")
    return {
        "source": source,
        "penalties": {str(h): c["penalty"] for h, c in choice.items()},
        "choice_scores": {str(h): c["scores"] for h, c in choice.items()},
        "ridge_penalties": {str(h): c["penalty"] for h, c in ridge["choice"].items()},
        "learned_rows": learned,
        "measured_rows": len(measured["day"]),
        "results": results,
        "gates": gates(p, measured, results),
        "checks": checks,
        "passed": checks["passed"],
        "seconds": round(time.time() - started),
    }


def gates(p: dict, table: dict, results: dict) -> dict:
    """Decision 0028's gates 2 to 4 on the measured rows, and the wrong sector beside them."""
    day = table["day"].astype(np.int64)
    top = {str(h): results["sector"][str(h)]["all"]["top0.01"] for h in SCORED}
    paying = {h: t["own"] > 0 and t["own_t"] >= PROOF for h, t in top.items()}
    found = {
        "over the ridge": net_harness.margin(p["sector"], p["ridge"], table["z"], day),
        "over the shuffled ridge": net_harness.margin(
            p["sector"], p["shuffled"], table["z"], day
        ),
        "its own cost": {"by_horizon": paying, "passed": any(paying.values())},
        "over the wrong sector, shown": net_harness.margin(
            p["sector"], p["wrong"], table["z"], day
        ),
    }
    found["variant passed"] = (
        found["over the ridge"]["passed"] and found["over the shuffled ridge"]["passed"]
    )
    found["bank and simulator next"] = (
        found["variant passed"] and found["its own cost"]["passed"]
    )
    return found


# --- the records ----------------------------------------------------------------------------


def fingerprint(folder: Path = HERE) -> dict:
    """Each file of CODE's SHA-256."""
    return {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in CODE
    }


def refusal(folder: Path, code: dict) -> str | None:
    """Why the measurement may not run yet, or None: it waits for a placebo that passed on
    today's code, and runs only once."""
    path = folder / "placebo.json"
    if not path.exists():
        return "run the placebo first"
    record = json.loads(path.read_text())
    if not record["passed"]:
        return "the placebo failed; nothing runs on real prices until it passes"
    changed = [name for name in code if record.get("code", {}).get(name) != code[name]]
    if changed:
        return (
            f"the placebo ran on other code ({', '.join(changed)} changed since); "
            "run the placebo again"
        )
    if (folder / "measure.json").exists():
        return "the measurement has run; it runs only once"
    return None


# --- the page -------------------------------------------------------------------------------


def results_html(found: dict) -> str:
    head = ["", *(f"{h} min" for h in HORIZONS)]

    def corr(kind):
        return lambda r: (
            "—"
            if r is None
            else f"{net_harness.signed(r[kind]['mean'])} ({net_harness.signed(r[kind]['t'], 1)})"
        )

    def top(key, t=None):
        def cell(r):
            if r is None:
                return "—"
            v = r["top0.01"][key]
            shown = net_harness.signed(v, 2)
            return shown + (f" ({net_harness.signed(r['top0.01'][t], 1)})" if t else "")

        return cell

    parts = [
        ("Each day's correlation with z (t)", corr("daily")),
        ("Within each minute, across instruments (t)", corr("across")),
        ("The strongest 1%, bps a signal before costs (t)", top("gross", "t")),
        ("The same after 2 bps", top("net")),
        ("Their own cost, bps a round trip with the hedge's", top("cost")),
        ("After their own cost (t)", top("own", "own_t")),
        ("After twice their own cost (t)", top("twice", "twice_t")),
    ]

    def rows_of(cell) -> list[list[str]]:
        return [
            [NAMES[name], *(cell(by_h[str(h)].get("all")) for h in HORIZONS)]
            for name, by_h in found["results"].items()
        ]

    return "".join(
        f"<h3>{html.escape(title)}</h3>" + costs.table_html(head, rows_of(cell))
        for title, cell in parts
    )


def gates_html(g: dict) -> str:
    def said(ok: bool) -> str:
        return "passed" if ok else "failed"

    def margin(m: dict) -> str:
        return f"{net_harness.signed(m['mean'], 4)} (t {net_harness.signed(m['t'], 2)})"

    paid = [f"{h} min" for h, ok in g["its own cost"]["by_horizon"].items() if ok]
    rows = [
        [
            "2. Its day-by-day margin over the skill test's ridge, 15-60 minutes",
            margin(g["over the ridge"]),
            said(g["over the ridge"]["passed"]),
        ],
        [
            "3. Its margin over itself on shuffled labels",
            margin(g["over the shuffled ridge"]),
            said(g["over the shuffled ridge"]["passed"]),
        ],
        [
            "4. Its strongest 1% after their own cost, at a t of 2, at",
            ", ".join(paid) or "no horizon",
            said(g["its own cost"]["passed"]),
        ],
        [
            "Shown, not a gate: its margin over the wrong sector",
            margin(g["over the wrong sector, shown"]),
            "",
        ],
    ]
    return costs.table_html(["gate", "found", ""], rows, numeric_from=3)


def section(name: str, found: dict) -> list[str]:
    penalties = ", ".join(f"{h} min {p:g}" for h, p in found["penalties"].items())
    head = [f"<h2>{html.escape(name)}, {html.escape(found['run_utc'])} UTC</h2>"]
    if found["source"] == "random":
        verdict = (
            "The placebo passed: nothing on the random walks reaches |t| 3."
            if found["passed"]
            else "The placebo failed, so the harness is wrong and nothing runs on real "
            "prices: " + "; ".join(found["checks"]["failed"])
        )
    else:
        verdict = (
            "Variant 3 passed."
            if found["gates"]["variant passed"]
            else "Variant 3 failed, and is written down as it is."
        )
    return [
        *head,
        f"<p class=verdict>{html.escape(verdict)}</p>",
        f"<p class=note>Penalties chosen on 2019: {html.escape(penalties)}.</p>",
        gates_html(found["gates"]),
        results_html(found),
    ]


def page(folder: Path) -> str:
    """The page of every run made so far."""
    sections = ["<h1>KUTBIT: variant 3, the sector's move</h1>"]
    sections.append(
        "<p class=note>Decision 0028. The skill test's rows, target and ridge, given one more "
        "channel: the move apart from QQQ, a minute old, of the instrument's sector ETF, "
        "matched on 2016-2018 and scaled by that match. Penalties chosen on 2019, fitted "
        "again on 2016-2019, measured once on 2020-2021. A t is in brackets.</p>"
    )
    for name, title in (
        ("measure", "Measured once on 2020-2021"),
        ("placebo", "The placebo, on random walks"),
    ):
        path = folder / f"{name}.json"
        if path.exists():
            sections += section(title, json.loads(path.read_text()))
    return (
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<title>KUTBIT sector</title><style>{costs.STYLE}</style></head><body><main>"
        + "".join(sections)
        + "</main></body></html>\n"
    )


def write(folder: Path, name: str, record: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(
        json.dumps(net_harness.plain(record), indent=1)
    )
    path = folder / "sector.html"
    path.write_text(page(folder))
    return path


# --- the commands ---------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("placebo", "measure"):
        raise SystemExit("usage: sector_harness.py placebo|measure [DATA]")
    data = Path(argv[1]) if len(argv) > 1 else DATA
    folder = data / "kutbit" / "sector"
    code = fingerprint()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    started = time.time()
    if argv[0] == "placebo":
        found = run(data, "random")
        path = write(folder, "placebo", found | {"run_utc": stamp, "code": code})
        verdict = "passes" if found["passed"] else "FAILS"
        print(f"placebo: {verdict} ({time.time() - started:.0f} s); {path}")
        return
    why = refusal(folder, code)
    if why:
        raise SystemExit(why)
    found = run(data, "real")
    path = write(folder, "measure", found | {"run_utc": stamp, "code": code})
    verdict = "passed" if found["gates"]["variant passed"] else "failed"
    print(f"measure: variant 3 {verdict} ({time.time() - started:.0f} s); {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
