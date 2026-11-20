"""The harness of KUTBIT's first agent (decision 0026).

It runs Efe's agent over the event gym in time order, as live. At each month's start the agent
learns afresh on every event before the month. At each minute's close it is given every event
whose outcome has just become known, and then asked for its action at that minute's extreme
minutes. The harness never gives an outcome before it is known, and it refuses an action that
Efe's limits forbid. Then, as decision 0026 orders:

- `placebo`: the whole procedure on random walks, where nothing can be learned. The agent may
  not beat doing nothing at a t of PLACEBO_T. It runs first.
- `choose`: each of the agent's SETTINGS runs over 2019 from every event since 2016, and the
  one of the highest mean utility is kept, before 2020-2021 are read.
- `measure`: once, on 2020-2021, with the setting chosen, beside the bars: doing nothing, the
  fixed bars, the best simple policy re-chosen each month, and the agent itself trained on
  outcomes dealt between stocks within each month, so that it never learns from beyond it.
  The agent passes if it beats every bar, its margin over the best one has a t of PASS_T
  clustered by day, and its certainty equivalent after 5 bps is above zero.

Efe's agent lives in agent.py: `make(settings)` returns an agent and SETTINGS lists the
settings chosen among. An agent has
- fit(history): every event before the month, with every action's utility;
- update(known): the events whose outcomes have just become known, the same way;
- act(events): the events of one minute, without their outcomes, returning one action index
  for each, among those `allowed`;
- and it may have forecast(events): its forecast utility of every action, for the page.

    PYTHONPATH=src uv run python research/kutbit/harness.py placebo [DATA]
    PYTHONPATH=src uv run python research/kutbit/harness.py choose [DATA]
    PYTHONPATH=src uv run python research/kutbit/harness.py measure [DATA]

Output in DATA/kutbit/agent/: placebo.json, choice.json and measure.json, and the pages
choice.html (2019) and agent.html (2020-2021).
"""

import html
import importlib
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import gym
import numpy as np
from skill import clustered_mean

DATA = Path("data")
CHOOSE_YEAR = 2019
MEASURE_YEARS = (2020, 2021)
PASS_T = 2.0  # the margin over the best bar must reach this t
PLACEBO_T = 2.0  # on random walks the agent may not beat doing nothing at this t
SEED = 26
SAMPLES = 4  # sample events of each sort on the page: the best, the worst, at random
VIEW = ("state", "ticker", "kind", "day", "minute", "beta")  # all known at the decision
OUTCOMES = ("own", "qqq")  # given only once known


# --- the run --------------------------------------------------------------------------------


def month_key(days: np.ndarray) -> np.ndarray:
    """Each day's month, as year * 12 + month - 1."""
    found = [date.fromordinal(int(d)) for d in days]
    return np.array([d.year * 12 + d.month - 1 for d in found], dtype=int)


def months_of(years) -> list[int]:
    return [year * 12 + m for year in years for m in range(12)]


def view(
    data: dict, idx: np.ndarray, permitted: np.ndarray, values=None, outcomes=None
) -> dict:
    """What the agent is given about the events `idx`: what it sees and what it may do, and,
    when `values` are given, what followed: every action's utility, the stock's and QQQ's
    moves over each hold (`own` and `qqq`, signed by the move's direction, taken from
    `outcomes` when given) and when they became known."""
    out = {k: data[k][idx] for k in VIEW}
    out["allowed"] = permitted[idx]
    if values is not None:
        source = data if outcomes is None else outcomes
        out["known"] = data["known"][idx]
        out["utility"] = values[idx]
        for k in OUTCOMES:
            out[k] = source[k][idx]
    return out


def run(
    agent,
    data: dict,
    values: np.ndarray,
    permitted: np.ndarray,
    months: list[int],
    forecasts: bool = False,
    outcomes: dict | None = None,
) -> dict:
    """`agent` over the events of `months`, in time order: at each month's start, fit on
    every event before it; at each minute's close, update with the outcomes known by then,
    then act on that minute's events. `values` and `outcomes` are what it learns from, the
    events' own unless given. Returns each event's action (-1 outside the months)
    and, if asked and the agent can, its forecast utility of the action it took."""
    n = len(data["day"])
    key = month_key(data["day"])
    choice = np.full(n, -1)
    expected = np.full(n, np.nan)
    decide_at = data["day"].astype(np.int64) * 1000 + data["minute"]
    known_at = data["day"].astype(np.int64) * 1000 + data["known"]
    forecast = getattr(agent, "forecast", None) if forecasts else None
    for month in months:
        before = np.flatnonzero(key < month)
        agent.fit(view(data, before, permitted, values, outcomes))
        mine = np.flatnonzero(key == month)
        to_decide = mine[np.lexsort((data["ticker"][mine], decide_at[mine]))]
        to_learn = mine[np.lexsort((data["ticker"][mine], known_at[mine]))]
        times, starts = np.unique(decide_at[to_decide], return_index=True)
        ends = [*starts[1:], len(to_decide)]
        learned = 0
        for when, first, last in zip(times, starts, ends, strict=True):
            ready = learned
            while ready < len(to_learn) and known_at[to_learn[ready]] <= when:
                ready += 1
            if ready > learned:
                batch = to_learn[learned:ready]
                agent.update(view(data, batch, permitted, values, outcomes))
                learned = ready
            group = to_decide[first:last]
            events = view(data, group, permitted)
            actions = np.asarray(agent.act(events), dtype=int)
            if actions.shape != (len(group),):
                raise ValueError("act must return one action for each event")
            if (actions < 0).any() or (actions >= permitted.shape[1]).any():
                raise ValueError("act returned an action that does not exist")
            if not permitted[group, actions].all():
                raise ValueError("the agent chose an action Efe's limits forbid")
            choice[group] = actions
            if forecast is not None:
                found = np.asarray(forecast(events))
                expected[group] = found[np.arange(len(group)), actions]
        if learned < len(to_learn):
            agent.update(view(data, to_learn[learned:], permitted, values, outcomes))
    return {"choice": choice, "expected": expected}


def within_months(data: dict, rng: np.random.Generator) -> dict:
    """The events with their outcomes dealt to another stock's event in the same part of the
    day and the same month, so that a bar trained on them never learns from beyond it. In a
    month where one part of the day has too few stocks for that, the month's parts are
    dealt together."""
    key = month_key(data["day"])
    out = {k: v.copy() for k, v in data.items()}
    for month in np.unique(key):
        mine = np.flatnonzero(key == month)
        part = gym.subset(data, mine)
        try:
            dealt = gym.shuffled(part, rng)
        except ValueError:
            dealt = gym.shuffled({**part, "minute": np.zeros_like(part["minute"])}, rng)
        for k in ("own", "qqq"):
            out[k][mine] = dealt[k]
    return out


# --- the bars and the test ------------------------------------------------------------------


def simple_policy(
    data: dict, values: np.ndarray, permitted: np.ndarray, months
) -> np.ndarray:
    """The best simple policy, chosen afresh at each month's start on every event before it."""
    key = month_key(data["day"])
    choice = np.full(len(key), -1)
    for month in months:
        before = key < month
        mine = np.flatnonzero(key == month)
        policy = gym.best_simple(
            values[before], permitted[before], data["kind"][before]
        )
        choice[mine] = gym.follow(policy, data["kind"][mine], permitted[mine])
    return choice


def bars(data: dict, values: np.ndarray, permitted: np.ndarray, months) -> dict:
    """Every bar's action at each event of the months, -1 outside them."""
    inside = np.isin(month_key(data["day"]), months)
    out = {k: np.where(inside, v, -1) for k, v in gym.bars(data, permitted).items()}
    out["best simple policy"] = simple_policy(data, values, permitted, months)
    return out


def gate(agent: np.ndarray, others: dict, day: np.ndarray) -> dict:
    """Decision 0026's test on per-event utilities: the agent beats every bar, its margin over
    the best has a t of PASS_T clustered by day, and its certainty equivalent is above zero."""
    means = {k: float(v.mean()) for k, v in others.items()}
    best = max(means, key=means.get)
    margin = clustered_mean(agent - others[best], day)
    beats = {k: bool(agent.mean() > m) for k, m in means.items()}
    ce = gym.certainty(float(agent.mean()))
    passed = all(beats.values()) and np.isfinite(margin["t"]) and margin["t"] >= PASS_T
    return {
        "best bar": best,
        "margin": margin,
        "beats": beats,
        "certainty": ce,
        "passed": bool(passed and ce > 0),
    }


def scored(values: dict, choice: np.ndarray, idx: np.ndarray) -> dict:
    """At each cost, a choice's certainty equivalent an event and its count of trades."""
    out = {}
    for cost, v in values.items():
        u = gym.taken(v[idx], choice[idx])
        out[f"{cost * 1e4:g} bps"] = {
            "certainty": gym.certainty(float(u.mean())),
            "trades": int((choice[idx] > 0).sum()),
        }
    return out


# --- the three steps --------------------------------------------------------------------------


def choose(make, settings: list, data: dict, values, permitted, log=print) -> dict:
    """Each setting run over CHOOSE_YEAR from every event before it; the best mean utility
    is kept."""
    months = months_of([CHOOSE_YEAR])
    idx = np.flatnonzero(np.isin(month_key(data["day"]), months))
    found, runs = [], []
    for i, setting in enumerate(settings):
        result = run(make(setting), data, values, permitted, months, forecasts=True)
        found.append(float(gym.taken(values[idx], result["choice"][idx]).mean()))
        runs.append(result)
        log(f"setting {i + 1} of {len(settings)}: mean utility {found[-1]:+.6f}")
    best = int(np.argmax(found))
    return {
        "settings": settings[best],
        "index": best,
        "scores": found,
        "candidates": settings,
        "run": runs[best],
    }


def placebo(make, settings: list, data: dict, log=print) -> dict:
    """The procedure on random walks: the choice on CHOOSE_YEAR, then MEASURE_YEARS. It
    passes unless the agent beats doing nothing there at a t of PLACEBO_T."""
    values = gym.utility(gym.returns(data))
    permitted = gym.allowed(data)
    chosen = choose(make, settings, data, values, permitted, log)
    months = months_of(MEASURE_YEARS)
    result = run(make(chosen["settings"]), data, values, permitted, months)
    idx = np.flatnonzero(np.isin(month_key(data["day"]), months))
    u = gym.taken(values[idx], result["choice"][idx])
    found = clustered_mean(u, data["day"][idx])
    beaten = np.isfinite(found["t"]) and found["t"] >= PLACEBO_T
    return {
        "settings": chosen["settings"],
        "agent": found,
        "trades": int((result["choice"][idx] > 0).sum()),
        "passed": not beaten,
    }


def measure(make, setting, data: dict, log=print) -> dict:
    """Once, on MEASURE_YEARS: the agent, the bars and the agent on dealt outcomes."""
    values = {c: gym.utility(gym.returns(data, cost=c)) for c in gym.COSTS}
    main = values[gym.COST]
    permitted = gym.allowed(data)
    months = months_of(MEASURE_YEARS)
    idx = np.flatnonzero(np.isin(month_key(data["day"]), months))
    result = run(make(setting), data, main, permitted, months, forecasts=True)
    log("the agent: done")
    dealt = within_months(data, np.random.default_rng(SEED))
    teach = gym.utility(gym.returns(dealt))
    shuffled_run = run(make(setting), data, teach, permitted, months, outcomes=dealt)
    log("the agent on dealt outcomes: done")
    others = bars(data, main, permitted, months)
    others["itself on dealt outcomes"] = shuffled_run["choice"]
    day = data["day"][idx]
    agent_u = gym.taken(main[idx], result["choice"][idx])
    found = gate(
        agent_u, {k: gym.taken(main[idx], v[idx]) for k, v in others.items()}, day
    )
    table = {"the agent": scored(values, result["choice"], idx)}
    table |= {k: scored(values, v, idx) for k, v in others.items()}
    return {
        "settings": setting,
        "gate": found,
        "table": table,
        "choice": result["choice"],
        "expected": result["expected"],
        "bars": others,
        "values": values,
        "idx": idx,
    }


# --- the page ---------------------------------------------------------------------------------


STYLE = """:root { color-scheme: light; --surface: #fcfcfb; --grid: #e4e3df; --text: #0b0b0b;
  --muted: #77766f; --secondary: #52514e; --line: #2a78d6; --hold: rgba(42,120,214,.12); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface: #1a1a19; --grid: #33332f; --text: #f0efec; --muted: #9a998f; --secondary: #c3c2b7;
  --line: #3987e5; --hold: rgba(57,135,229,.18); } }
:root[data-theme="dark"] { color-scheme: dark; --surface: #1a1a19; --grid: #33332f;
  --text: #f0efec; --muted: #9a998f; --secondary: #c3c2b7; --line: #3987e5;
  --hold: rgba(57,135,229,.18); }
body { margin: 0; background: var(--surface); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; }
.tables { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); }
th { color: var(--secondary); font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.note { color: var(--muted); font-size: 12px; }
.verdict { font-size: 16px; margin: 12px 0; }
.events { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 12px; }
.event { border: 1px solid var(--grid); border-radius: 8px; padding: 8px; font-size: 12px; }
.event svg { width: 100%; height: 80px; display: block; }"""


def table_html(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    cells = "".join(
        f"<th{' class=num' if i >= numeric_from else ''}>{html.escape(h)}</th>"
        for i, h in enumerate(head)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{' class=num' if i >= numeric_from else ''}>{html.escape(c)}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<div class=tables><table><tr>{cells}</tr>{body}</table></div>"


def sparkline(chart: np.ndarray, hold: int) -> str:
    """The chart around an event: the line in log moves from the extreme minute's close, the
    hold shaded from the next bar."""
    keep = np.isfinite(chart)
    x = np.arange(len(chart))[keep]
    y = chart[keep]
    low, high = float(y.min()), float(y.max())
    span = high - low or 1.0
    width, height = 240, 80

    def px(i):
        return i / (len(chart) - 1) * width

    def py(v):
        return height - 4 - (v - low) / span * (height - 8)

    points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in zip(x, y, strict=True))
    centre = (len(chart) - 1) // 2
    shade = ""
    if hold:
        left, right = px(centre + 1), px(min(centre + hold, len(chart) - 1))
        shade = (
            f'<rect x="{left:.1f}" y="0" width="{right - left:.1f}" height="{height}" '
            'fill="var(--hold)"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        'aria-label="the minutes around the event">'
        f'{shade}<line x1="{px(centre):.1f}" x2="{px(centre):.1f}" y1="0" y2="{height}" '
        'stroke="var(--grid)"/>'
        f'<polyline points="{points}" fill="none" stroke="var(--line)" stroke-width="1.5"/>'
        "</svg>"
    )


def page(
    title: str,
    data: dict,
    tickers: list[str],
    idx: np.ndarray,
    choice: np.ndarray,
    expected: np.ndarray,
    values: dict,
    others: dict,
    found: dict | None = None,
) -> str:
    """What the agent did on the events `idx`: its worth against the bars, its actions by
    kind of event, its hedges, its forecasts against what followed, where it left the best
    simple policy, and sample events."""
    main = values[gym.COST]
    sections = [f"<h1>{html.escape(title)}</h1>"]
    sections.append(
        "<p class=note>Decision 0026. Certainty equivalents are in bps of the account an "
        "event, at γ 2; an event the agent lets be counts as zero.</p>"
    )
    if found is not None:
        verdict = "passed" if found["passed"] else "failed"
        margin = found["margin"]
        sections.append(
            f"<p class=verdict>The test: <b>{verdict}</b>. Best bar: "
            f"{html.escape(found['best bar'])}; margin over it {margin['mean'] * 1e4:+.2f} bps "
            f"of utility an event, t {margin['t']:+.2f}; certainty equivalent "
            f"{found['certainty'] * 1e4:+.2f} bps.</p>"
        )
    rows = []
    for name, choices in {"the agent": choice, **others}.items():
        row = [name]
        for v in values.values():
            u = gym.taken(v[idx], choices[idx])
            row.append(f"{gym.certainty(float(u.mean())) * 1e4:+.2f}")
        row.append(f"{int((choices[idx] > 0).sum()):,}")
        rows.append(row)
    head = ["", *(f"{c * 1e4:g} bps" for c in values), "trades"]
    sections += ["<h2>Its worth against the bars</h2>", table_html(head, rows)]

    acts = [gym.ACTIONS[a] for a in choice[idx]]
    kind = data["kind"][idx].astype(int)
    rows = []
    for k, name in enumerate(gym.KINDS):
        mine = [a for a, kk in zip(acts, kind, strict=True) if kk == k]
        traded = [a for a in mine if a.side != 0]
        if not mine:
            continue
        rows.append(
            [
                name,
                f"{len(mine):,}",
                f"{len(traded) / len(mine):.0%}",
                f"{sum(a.side < 0 for a in traded) / max(len(traded), 1):.0%}",
                f"{np.mean([a.size for a in traded]):.2f}x" if traded else "-",
                f"{np.mean([a.hold for a in traded]):.0f} min" if traded else "-",
                f"{np.mean([a.hedge for a in traded]):.2f}" if traded else "-",
            ]
        )
    head = ["kind of event", "events", "traded", "against", "mean size", "mean hold",
            "mean hedge"]  # fmt: skip
    sections += ["<h2>What it did, by kind of event</h2>", table_html(head, rows)]

    moved = (kind & 2) == 2
    rows = []
    for label, where in (("QQQ quiet", ~moved), ("QQQ moved", moved)):
        hedges = [
            a.hedge for a, w in zip(acts, where, strict=True) if w and a.side != 0
        ]
        if hedges:
            rows.append(
                [label, f"{len(hedges):,}"]
                + [f"{np.mean(np.isclose(hedges, h)):.0%}" for h in gym.HEDGES]
            )
    head = ["trades", "count", *(f"hedge {h:g}" for h in gym.HEDGES)]
    sections += ["<h2>Where it hedged</h2>", table_html(head, rows)]

    traded = idx[choice[idx] > 0]
    guesses = expected[traded]
    if len(traded) and np.isfinite(guesses).any():
        keep = traded[np.isfinite(guesses)]
        guesses = expected[keep]
        realised = gym.taken(main[keep], choice[keep])
        edges = np.quantile(guesses, np.linspace(0, 1, 11))
        rows = []
        for i in range(10):
            inside = (guesses >= edges[i]) & (
                guesses <= edges[i + 1] if i == 9 else guesses < edges[i + 1]
            )
            if inside.any():
                rows.append(
                    [
                        f"{i + 1}",
                        f"{inside.sum():,}",
                        f"{guesses[inside].mean() * 1e4:+.1f}",
                        f"{realised[inside].mean() * 1e4:+.1f}",
                    ]
                )
        head = ["tenth", "trades", "forecast", "what followed"]
        sections += [
            "<h2>Its forecasts against what followed, utility in bps</h2>",
            table_html(head, rows),
        ]

    simple = others.get("best simple policy")
    if simple is not None:
        apart = idx[choice[idx] != simple[idx]]
        if len(apart):
            gain = gym.taken(main[apart], choice[apart]) - gym.taken(
                main[apart], simple[apart]
            )
            found_apart = clustered_mean(gain, data["day"][apart])
            sections += [
                "<h2>Where it left the best simple policy</h2>",
                (
                    f"<p>{len(apart):,} events; there it earned "
                    f"{found_apart['mean'] * 1e4:+.2f} bps of utility an event more than "
                    f"the policy would have, t {found_apart['t']:+.2f}.</p>"
                ),
            ]

    if len(traded):
        u = gym.taken(main[traded], choice[traded])
        order = np.argsort(u)
        rng = np.random.default_rng(SEED)
        picks = []
        for label, chosen in (
            ("best", order[::-1][:SAMPLES]),
            ("worst", order[:SAMPLES]),
            (
                "at random",
                rng.choice(len(traded), min(SAMPLES, len(traded)), replace=False),
            ),
        ):
            picks += [(label, traded[i]) for i in chosen]
        cards = []
        for label, i in picks:
            action = gym.ACTIONS[choice[i]]
            when = date.fromordinal(int(data["day"][i]))
            minute = int(data["minute"][i])
            clock = f"{9 + (30 + minute) // 60:02d}:{(30 + minute) % 60:02d}"
            r = gym.returns(gym.subset(data, np.array([i])))[0, choice[i]]
            cards.append(
                "<div class=event>"
                + sparkline(data["chart"][i], action.hold)
                + f"<div><b>{html.escape(tickers[int(data['ticker'][i])])}</b> "
                f"{when} {clock} ET, {label}</div>"
                f"<div>{data['state'][i, 0]:.1f} usual moves "
                f"{'up' if data['state'][i, 1] > 0 else 'down'}; "
                f"{html.escape(gym.KINDS[int(data['kind'][i])])}</div>"
                f"<div>{html.escape(gym.describe(action))}: {r * 1e4:+.1f} bps of the account"
                "</div></div>"
            )
        sections += [
            "<h2>Sample events</h2>",
            (
                "<p class=note>The line is the stock's price around the extreme minute, "
                "the vertical line that minute, the shade the hold.</p>"
            ),
            f"<div class=events>{''.join(cards)}</div>",
        ]
    return (
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<title>KUTBIT agent</title><style>{STYLE}</style></head><body><main>"
        + "".join(sections)
        + "</main></body></html>\n"
    )


# --- the commands -------------------------------------------------------------------------------


def plain(found: dict) -> dict:
    """A result without its arrays, for JSON."""
    return json.loads(
        json.dumps(
            {k: v for k, v in found.items() if not isinstance(v, np.ndarray)},
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )


def efe_agent():
    try:
        return importlib.import_module("agent")
    except ModuleNotFoundError as missing:
        raise SystemExit(
            "write research/kutbit/agent.py first, with make(settings) and SETTINGS"
        ) from missing


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("placebo", "choose", "measure"):
        raise SystemExit("usage: harness.py placebo|choose|measure [DATA]")
    data_dir = Path(argv[1]) if len(argv) > 1 else DATA
    folder = data_dir / "kutbit" / "agent"
    agent = efe_agent()
    started = time.time()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    if argv[0] == "placebo":
        data = gym.load_events(folder / "random", gym.YEARS)
        found = placebo(agent.make, agent.SETTINGS, data)
        found["run_utc"] = stamp
        (folder / "placebo.json").write_text(json.dumps(plain(found), indent=1))
        verdict = "passes" if found["passed"] else "FAILS"
        print(
            f"placebo: {verdict}, t {found['agent']['t']:+.2f} ({time.time() - started:.0f} s)"
        )
        return
    tickers = json.loads((folder / "real" / "tickers.json").read_text())
    if argv[0] == "choose":
        years = (*gym.LEARN, CHOOSE_YEAR)
        data = gym.load_events(folder / "real", years)
        values = gym.utility(gym.returns(data))
        permitted = gym.allowed(data)
        found = choose(agent.make, agent.SETTINGS, data, values, permitted)
        months = months_of([CHOOSE_YEAR])
        idx = np.flatnonzero(np.isin(month_key(data["day"]), months))
        costs = {c: gym.utility(gym.returns(data, cost=c)) for c in gym.COSTS}
        text = page(
            f"KUTBIT agent on {CHOOSE_YEAR}, where its setting was chosen",
            data,
            tickers,
            idx,
            found["run"]["choice"],
            found["run"]["expected"],
            costs,
            bars(data, values, permitted, months),
        )
        (folder / "choice.html").write_text(text)
        record = {k: v for k, v in found.items() if k != "run"} | {"run_utc": stamp}
        (folder / "choice.json").write_text(json.dumps(plain(record), indent=1))
        print(f"choose: setting {found['index'] + 1} kept; {folder / 'choice.html'}")
        return
    placebo_file, choice_file = folder / "placebo.json", folder / "choice.json"
    if not placebo_file.exists() or not json.loads(placebo_file.read_text())["passed"]:
        raise SystemExit("run the placebo first; measure runs only once it passes")
    if not choice_file.exists():
        raise SystemExit("choose the setting on 2019 first")
    setting = json.loads(choice_file.read_text())["settings"]
    data = gym.load_events(folder / "real", gym.YEARS)
    found = measure(agent.make, setting, data)
    text = page(
        "KUTBIT agent on 2020-2021, measured once",
        data,
        tickers,
        found["idx"],
        found["choice"],
        found["expected"],
        found["values"],
        found["bars"],
        found["gate"],
    )
    (folder / "agent.html").write_text(text)
    record = {k: found[k] for k in ("settings", "gate", "table")} | {"run_utc": stamp}
    (folder / "measure.json").write_text(json.dumps(plain(record), indent=1))
    verdict = "passed" if found["gate"]["passed"] else "failed"
    print(f"measure: the agent {verdict}; {folder / 'agent.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
