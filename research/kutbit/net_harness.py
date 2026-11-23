"""Efe's network, fed and judged as decision 0027 wrote before it ran.

The network, Efe's `net.py`, learns from and is measured on the skill test's rows
(`net_data.py`), so that only the model differs from the ridge. It gives two things, which
`NETWORK_GUIDE.md` builds step by step:

    learn(rows, width, decay, seed, epochs, after_epoch) -> params
        Learns on `rows` from `seed` for `epochs` epochs. `rows.count` is their number and
        `rows.batch(index)` gives, for a tensor of row positions, their standardised inputs
        and their `z`, on `rows.device`. It calls after_epoch(epoch, params) after each
        epoch, counted from 1.
    forecast(params, x) -> a tensor [rows, horizons]

The runs, in decision 0027's order:

- `smoke`: random-walk rows of 2016 through Efe's network at each width, for one epoch, to
  time it with its own Adam and to check its forecasts are finite. It shows no score.
- `placebo`: the choice and the measurement below, on the skill test's random walks. No
  forecaster may reach |t| PLACEBO_T at any horizon in any measure, or the harness is wrong
  and nothing runs on real prices.
- `choose`, on 2019: each of SETTINGS learns from SEEDS on 2016-2018 for EPOCHS epochs, and
  after each epoch the seeds' mean forecasts 2019. The setting and the epoch whose forecasts
  score best are kept: each day's correlation with `z`, about the means of all the rows,
  averaged over the days and over SCORED. A tie keeps the earlier.
- `measure`, once, on 2020-2021:
  - the ridge is fitted again as the skill test fitted it, from the same rows in the same
    order, and must give the skill test's own numbers, or the run stops;
  - with the setting kept, the seeds learn again on 2016-2019 for the epochs kept, and so
    does the same network on `z` shuffled between the instruments of each minute;
  - every forecaster gets the skill test's measures, and its strongest forecasts their
    return after each signal's own cost, and after twice that;
  - the gates: the network's day-by-day margin over the ridge, and over the shuffled
    network, in each day's correlation averaged over SCORED, each at a t of PROOF; and, at
    one of SCORED at least, its strongest 1% earning more than their own cost at a t of
    PROOF, clustered by day.

Each run's record keeps the SHA-256 of CODE. `choose` waits for a placebo that passed on
today's code; `measure` waits for that placebo and a choice made on today's code, and runs
only once. The records and the page, net.html, are in DATA/kutbit/net. Eight threads keep the
ridge's small solves fast; on many more they run a hundred times slower.

    OMP_NUM_THREADS=8 PYTHONPATH=src data/llmenv/bin/python research/kutbit/net_harness.py \
        smoke|placebo|choose|measure [DATA]
"""

import hashlib
import html
import importlib.util
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import costs
import net_data
import numpy as np
import polars as pl
import skill
from labels import HORIZONS

from qfin_alphaguard.kutbit.features import SCALARS

DATA = Path("data")
HERE = Path(__file__).parent
SETTINGS = ((128, 0.1), (128, 0.01), (512, 0.1), (512, 0.01))  # width, weight decay
SEEDS = (1, 2, 3, 4, 5)
EPOCHS = 20
# minutes: the horizons whose strongest forecasts could pay a cost (0024's result)
SCORED = (15, 30, 60)
SCORED_AT = [HORIZONS.index(h) for h in SCORED]
PROOF = 2.0
PLACEBO_T = skill.PLACEBO_T
SHUFFLE_SEED = 27
CHUNK = 32_768  # rows forecast or summed at once
CODE = ("net.py", "net_harness.py", "net_data.py", "skill.py", "costs.py")
NAMES = {
    "network": "Efe's network",
    "ridge": "The ridge on the 971 inputs",
    "shuffled": "The network on shuffled labels",
    "habit": "Known pattern: habit",
    "reversal": f"Known pattern: the last {skill.REVERSAL} minutes reversed",
}
AGAIN = {"placebo": "run the placebo again", "choice": "choose again"}


# --- the rows on the device ---------------------------------------------------------------


class Rows:
    """Stored rows on a device. `batch(index)` gives, for a tensor of row positions, their 971
    inputs standardised by `mean` and `scale`, and their `z` (or `z` given in its place), as
    float32 tensors."""

    def __init__(self, table: dict, mean, scale, device, z: np.ndarray | None = None):
        import torch

        self.count = len(table["day"])
        self.device = device
        self.inputs = net_data.WINDOW_INPUTS + len(SCALARS)
        self.horizons = table["z"].shape[1]
        self.channels = torch.from_numpy(table["channels"]).to(device)
        self.scalars = torch.from_numpy(table["scalars"]).to(device)
        self.minute = torch.from_numpy(table["minute"].astype(np.int64)).to(device)
        target = table["z"] if z is None else z
        self.z = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device)
        self.mean = torch.as_tensor(np.asarray(mean, dtype=np.float32), device=device)
        self.scale = torch.as_tensor(np.asarray(scale, dtype=np.float32), device=device)

    def raw(self, index):
        """The inputs of `index` before standardising, laid out as the skill test's."""
        import torch

        windows = net_data.gathered(self.channels, self.minute, index)
        return torch.cat([windows.reshape(len(index), -1), self.scalars[index]], dim=1)

    def batch(self, index):
        return (self.raw(index) - self.mean) / self.scale, self.z[index]


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chunks(count: int, on):
    """Consecutive row positions, CHUNK at a time, as tensors on the device `on`."""
    import torch

    for start in range(0, count, CHUNK):
        yield start, torch.arange(start, min(start + CHUNK, count), device=on)


def standardisation(table: dict, on) -> tuple[np.ndarray, np.ndarray]:
    """Each input's mean and standard deviation over the rows a network learns from, in two
    passes: the means, then the squares about them, so that an input that never varies has
    no spread at all and is scaled by 1, staying at zero."""
    import torch

    rows = Rows(table, 0.0, 1.0, on)
    total = torch.zeros(rows.inputs, dtype=torch.float64, device=on)
    for _, index in chunks(rows.count, on):
        total += rows.raw(index).double().sum(dim=0)
    mean = total / rows.count
    squares = torch.zeros_like(total)
    for _, index in chunks(rows.count, on):
        squares += ((rows.raw(index).double() - mean) ** 2).sum(dim=0)
    sd = (squares / rows.count).sqrt()
    scale = torch.where(sd > 1e-9, sd, torch.ones_like(sd))
    return mean.cpu().numpy(), scale.cpu().numpy()


def forecasts(net, params, rows: Rows) -> np.ndarray:
    """The network's forecast of every row of `rows`, [rows, horizons], on the CPU."""
    import torch

    out = np.empty((rows.count, rows.horizons))
    with torch.no_grad():
        for start, index in chunks(rows.count, rows.device):
            x, _ = rows.batch(index)
            found = net.forecast(params, x)
            out[start : start + len(index)] = found.double().cpu().numpy()
    return out


def shuffled(
    z: np.ndarray, day: np.ndarray, minute: np.ndarray, seed: int = SHUFFLE_SEED
):
    """`z` dealt again at random between the instruments of each minute of each day: each row
    gets the target of a row of the same minute. A network learning from it can learn only
    what all the instruments of a minute share."""
    group = day.astype(np.int64) * 1000 + minute
    order = np.lexsort((np.arange(len(group)), group))
    mixed = np.lexsort((np.random.default_rng(seed).random(len(group)), group))
    out = np.empty_like(z)
    out[order] = z[mixed]
    return out


# --- learning and the choice --------------------------------------------------------------


def learned(net, learn: Rows, width: int, decay: float, epochs: int, after) -> dict:
    """One seed after another learn on `learn`; after each epoch of each, `after(seed, epoch,
    params)`. Efe's network must call back once after each epoch."""
    for seed in SEEDS:
        called = []

        def each(epoch, params, seed=seed, called=called):
            called.append(epoch)
            after(seed, epoch, params)

        net.learn(learn, width, decay, seed, epochs, each)
        if called != list(range(1, epochs + 1)):
            raise ValueError(
                f"learn must call after_epoch once after each of its {epochs} epochs, "
                f"counted from 1; it called {called[:5]}"
            )


def score(p: np.ndarray, z: np.ndarray, day: np.ndarray) -> float:
    """Each day's correlation of forecast and `z`, about the means of all the rows, averaged
    over the days and over the horizons given, one a column."""
    return float(
        np.mean(
            [
                skill.daily_correlation(p[:, k].astype(np.float64), z[:, k], day)[
                    "mean"
                ]
                for k in range(p.shape[1])
            ]
        )
    )


def kept(scores: np.ndarray) -> tuple[int, int]:
    """The setting, counted from 0, and the epoch, counted from 1, that score best. A tie, or
    a score that is not a number, keeps the earlier."""
    best, at = -math.inf, (0, 1)
    for i in range(scores.shape[0]):
        for e in range(scores.shape[1]):
            if scores[i, e] > best:
                best, at = scores[i, e], (i, e + 1)
    return at


def choice_on(net, learn_table: dict, choose_table: dict, log=print) -> dict:
    """Decision 0027's choice: each setting's seeds learn on `learn_table` for EPOCHS epochs,
    and after each epoch their mean forecasts `choose_table` and is scored."""
    on = device()
    mean, scale = standardisation(learn_table, on)
    learn = Rows(learn_table, mean, scale, on)
    target = Rows(choose_table, mean, scale, on)
    z = choose_table["z"][:, SCORED_AT]
    day = choose_table["day"].astype(np.int64)
    scores = np.full((len(SETTINGS), EPOCHS), np.nan)
    for i, (width, decay) in enumerate(SETTINGS):
        started = time.time()
        total = np.zeros((EPOCHS, target.count, len(SCORED_AT)), dtype=np.float32)

        def after(seed, epoch, params, total=total):
            total[epoch - 1] += forecasts(net, params, target)[:, SCORED_AT]

        learned(net, learn, width, decay, EPOCHS, after)
        for e in range(EPOCHS):
            scores[i, e] = score(total[e] / len(SEEDS), z, day)
        best = int(np.nanargmax(scores[i])) if np.isfinite(scores[i]).any() else 0
        log(
            f"setting {i + 1} (width {width}, decay {decay:g}): best at epoch {best + 1} "
            f"({time.time() - started:.0f} s)"
        )
    i, epochs = kept(scores)
    width, decay = SETTINGS[i]
    return {
        "setting": i + 1,
        "width": width,
        "decay": decay,
        "epochs": epochs,
        "scores": scores.tolist(),
        "learned_rows": learn.count,
        "chosen_rows": target.count,
    }


# --- the ridge, fitted again --------------------------------------------------------------


def by_day(table: dict) -> list[np.ndarray]:
    """The positions of each day's rows, in order; a day's rows are stored together."""
    day = table["day"]
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    return [np.arange(s, e) for s, e in zip(starts, ends)]


def ridge_again(data: Path, source: str, log=print) -> dict:
    """The skill test's ridge, fitted again as the skill test fitted it: each year's moments
    from the same rows, day by day in the same order; its penalties and thresholds chosen on
    2019; and its forecasts of 2020-2021, with the known patterns' thresholds."""
    started = time.time()
    moments = {}
    for year in (*skill.FIT_YEARS, skill.CHOOSE_YEAR):
        table = net_data.load(data, source, [year])
        moments[year] = skill.Moments()
        for rows in by_day(table):
            moments[year].add(net_data.inputs_of(table, rows), table["z"][rows])
    early = sum((moments[y] for y in skill.FIT_YEARS), skill.Moments())
    final = early + moments[skill.CHOOSE_YEAR]
    early_models = [skill.fit(early, penalty) for penalty in skill.PENALTIES]
    final_models = [skill.fit(final, penalty) for penalty in skill.PENALTIES]
    table = net_data.load(data, source, [skill.CHOOSE_YEAR])
    seen = {"early": [], "final": []}
    for rows in by_day(table):
        x = net_data.inputs_of(table, rows)
        seen["early"].append(np.stack([m(x) for m in early_models], axis=2))
        seen["final"].append(np.stack([m(x) for m in final_models], axis=2))
    choice, thresholds = skill.choose(
        np.concatenate(seen["early"]),
        np.concatenate(seen["final"]),
        table["patterns"],
        table["z"],
    )
    del seen
    model = skill.ridge_of(final_models, choice)
    measured = net_data.load(data, source, skill.MEASURE_YEARS)
    p = np.concatenate(
        [model(net_data.inputs_of(measured, r)) for r in by_day(measured)]
    )
    log(f"the ridge fitted again ({time.time() - started:.0f} s)")
    return {"forecast": p, "choice": choice, "thresholds": thresholds}


def differences(theirs, ours, where: str = "") -> list[str]:
    """Where `ours` differs from `theirs`, number by number, beyond rounding: another build
    of the same linear algebra may move a forecast in its last digits, and a signal at its
    threshold in or out."""
    if isinstance(theirs, dict):
        return [
            d
            for k, v in theirs.items()
            for d in differences(
                v, ours.get(k) if isinstance(ours, dict) else None, f"{where}/{k}"
            )
        ]
    if not isinstance(theirs, (int, float)) or not isinstance(ours, (int, float)):
        return [] if theirs == ours else [f"{where}: {theirs} against {ours}"]
    if math.isnan(theirs) and math.isnan(ours):
        return []
    if math.isclose(theirs, ours, rel_tol=1e-4, abs_tol=1e-6):
        return []
    return [f"{where}: {theirs} against {ours}"]


def as_skill_test(data: Path, source: str, results: dict, choice: dict) -> list[str]:
    """Where the ridge and the known patterns, measured again, differ from the skill test's
    own run on `source`."""
    path = data / "kutbit" / "skill" / f"{source}.json"
    if not path.exists():
        return [f"the skill test's {source} run is missing: {path}"]
    theirs = json.loads(path.read_text())
    found = differences(
        {str(h): c["penalty"] for h, c in theirs["choice"].items()},
        {str(h): c["penalty"] for h, c in choice.items()},
        "penalty",
    )
    for name in ("ridge", "habit", "reversal"):
        found += differences(theirs["results"][name], results[name], name)
    return found


# --- the measures and the gates -----------------------------------------------------------


def signal_costs(data: Path, table: dict) -> np.ndarray:
    """Each row's round trip in bps (decision 0027, section 7): its instrument's mean spread in
    its part of the day over the cost study's moments, or the median of its kind's there,
    plus |beta| times QQQ's in that part, for the hedge."""
    tickers = json.loads((data / "kutbit" / "net" / "tickers.json").read_text())
    days = pl.read_parquet(data / "kutbit" / "days.parquet", columns=["ticker", "etf"])
    etfs = set(days.filter(pl.col("etf"))["ticker"].unique())
    folder = data / "kutbit" / "costs" / "moments"
    moments = costs.valid_moments(costs.load(folder, costs.MOMENT_SCHEMA), etfs)
    parts = [name for name, _, _ in costs.PARTS]
    means = moments.group_by("ticker", "part", "etf").agg(pl.col("spread").mean())
    own = {(r["ticker"], r["part"]): r["spread"] for r in means.iter_rows(named=True)}
    kinds = means.group_by("etf", "part").agg(pl.col("spread").median())
    usual = {(r["etf"], r["part"]): r["spread"] for r in kinds.iter_rows(named=True)}
    cost_of = np.array(
        [[own.get((t, p), usual[(t in etfs, p)]) for p in parts] for t in tickers]
    )
    qqq = np.array([own[(skill.MARKET, p)] for p in parts])
    part = np.digitize(table["minute"], [end for _, _, end in costs.PARTS[:-1]])
    beta = np.abs(table["scalars"][:, SCALARS.index("beta")].astype(np.float64))
    return cost_of[table["instrument"], part] + beta * qqq[part]


def own_cost(p, apart_bps, cost_bps, day, threshold: float) -> dict:
    """Trading in the forecast's direction where its size reaches `threshold`: each signal's
    own round trip, and its hedged return in bps after that cost and after twice it, each with
    a t clustered by day."""
    chosen = np.abs(p) >= threshold
    gross = np.sign(p[chosen]) * apart_bps[chosen]
    paid = cost_bps[chosen]
    once = skill.clustered_mean(gross - paid, day[chosen])
    twice = skill.clustered_mean(gross - 2 * paid, day[chosen])
    return {
        "cost": float(paid.mean()) if len(paid) else float("nan"),
        "own": once["mean"],
        "own_t": once["t"],
        "twice": twice["mean"],
        "twice_t": twice["t"],
    }


def measures_of(p: np.ndarray, table: dict, limits_of, cost: np.ndarray) -> dict:
    """Every measure of one forecaster at each horizon, on all instruments, the stocks and
    the ETFs: the skill test's, and its strongest after their own cost."""
    day = table["day"].astype(np.int64)
    minute = table["minute"].astype(np.int64)
    groups = {"all": slice(None), "stocks": ~table["etf"], "etfs": table["etf"]}
    out = {}
    for j, h in enumerate(HORIZONS):
        limits, out[str(h)] = limits_of(h), {}
        for g, s in groups.items():
            if not len(day[s]):
                continue
            found = skill.measures(
                p[s, j],
                table["z"][s, j],
                table["apart"][s, j],
                day[s],
                minute[s],
                limits,
            )
            for share in skill.STRONGEST:
                found[f"top{share:g}"] |= own_cost(
                    p[s, j], table["apart"][s, j] * 1e4, cost[s], day[s], limits[share]
                )
            out[str(h)][g] = found
    return out


def margin(p: np.ndarray, q: np.ndarray, z: np.ndarray, day: np.ndarray) -> dict:
    """p's day-by-day margin over q in each day's correlation with `z`, about the means of all
    the rows, averaged over SCORED: the mean of the margins over the days, and its t."""
    by_horizon = []
    for j in SCORED_AT:
        keys, mine = skill.grouped_correlations(p[:, j], z[:, j], day, about_all=True)
        others, theirs = skill.grouped_correlations(
            q[:, j], z[:, j], day, about_all=True
        )
        common, a, b = np.intersect1d(keys, others, return_indices=True)
        by_horizon.append(dict(zip(common.tolist(), (mine[a] - theirs[b]).tolist())))
    days = sorted(set.intersection(*(set(d) for d in by_horizon)))
    found = skill.mean_and_t(
        np.array([np.mean([d[k] for d in by_horizon]) for k in days], dtype=float)
    )
    return found | {"passed": bool(found["t"] >= PROOF)}


def gates(p: dict, table: dict, results: dict) -> dict:
    """Decision 0027's gates 2 to 4 on the measured rows."""
    day = table["day"].astype(np.int64)
    paying = {
        str(h): results["network"][str(h)]["all"]["top0.01"]["own"] > 0
        and results["network"][str(h)]["all"]["top0.01"]["own_t"] >= PROOF
        for h in SCORED
    }
    found = {
        "over the ridge": margin(p["network"], p["ridge"], table["z"], day),
        "over the shuffled network": margin(
            p["network"], p["shuffled"], table["z"], day
        ),
        "its own cost": {"by_horizon": paying, "passed": any(paying.values())},
    }
    found["variant passed"] = (
        found["over the ridge"]["passed"]
        and found["over the shuffled network"]["passed"]
    )
    found["bank and simulator next"] = (
        found["variant passed"] and found["its own cost"]["passed"]
    )
    return found


def thresholds_by_horizon(p: np.ndarray) -> dict:
    return {h: skill.thresholds_of(p[:, j]) for j, h in enumerate(HORIZONS)}


def measurement(net, data: Path, source: str, choice: dict, log=print) -> dict:
    """Decision 0027's measurement on `source`'s 2020-2021: the ridge again, checked against
    the skill test's; the network and its shuffled twin learning on 2016-2019; every
    forecaster's measures; and the gates."""
    import torch

    started = time.time()
    ridge = ridge_again(data, source, log)
    measured = net_data.load(data, source, skill.MEASURE_YEARS)
    cost = signal_costs(data, measured)
    patterns = {
        name: np.repeat(measured["patterns"][:, i : i + 1], len(HORIZONS), axis=1)
        for i, name in enumerate(skill.PATTERNS)
    }
    results = {
        "ridge": measures_of(
            ridge["forecast"], measured, lambda h: ridge["thresholds"][h], cost
        )
    }
    for name, p in patterns.items():
        results[name] = measures_of(
            p, measured, lambda h, n=name: ridge["thresholds"][n], cost
        )
    differ = as_skill_test(data, source, results, ridge["choice"])
    if differ:
        raise SystemExit(
            "the ridge fitted again does not give the skill test's numbers, so the run stops: "
            + "; ".join(differ[:5])
        )
    log("the ridge and the known patterns give the skill test's numbers")
    learn_years = (*skill.FIT_YEARS, skill.CHOOSE_YEAR)
    learn_table = net_data.load(data, source, learn_years)
    chosen = net_data.load(data, source, [skill.CHOOSE_YEAR])
    on = device()
    mean, scale = standardisation(learn_table, on)
    targets = [Rows(chosen, mean, scale, on), Rows(measured, mean, scale, on)]
    p, limits = {"ridge": ridge["forecast"]}, {}
    for name, z in (
        ("network", None),
        (
            "shuffled",
            shuffled(learn_table["z"], learn_table["day"], learn_table["minute"]),
        ),
    ):
        learn = Rows(learn_table, mean, scale, on, z=z)
        sums = [np.zeros((t.count, t.horizons)) for t in targets]

        def after(seed, epoch, params, sums=sums):
            if epoch == choice["epochs"]:
                for s, t in zip(sums, targets):
                    s += forecasts(net, params, t)

        learned(net, learn, choice["width"], choice["decay"], choice["epochs"], after)
        limits[name] = thresholds_by_horizon(sums[0] / len(SEEDS))
        p[name] = sums[1] / len(SEEDS)
        del learn
        if on.type == "cuda":
            torch.cuda.empty_cache()
        log(
            f"{name}: learned on {learn_years[0]}-{learn_years[-1]} ({time.time() - started:.0f} s)"
        )
    for name in ("network", "shuffled"):
        results[name] = measures_of(
            p[name], measured, lambda h, n=name: limits[n][h], cost
        )
    order = ("network", "ridge", "shuffled", *skill.PATTERNS)
    return {
        "source": source,
        "choice": choice,
        "ridge_penalties": {str(h): c["penalty"] for h, c in ridge["choice"].items()},
        "learned_rows": len(learn_table["day"]),
        "measured_rows": len(measured["day"]),
        "results": {name: results[name] for name in order},
        "gates": gates(p, measured, results),
        "seconds": round(time.time() - started),
    }


def placebo(net, data: Path, log=print) -> dict:
    """The choice and the measurement on the skill test's random walks, and the placebo rule:
    no forecaster may reach |t| PLACEBO_T at any horizon in any measure."""
    learn = net_data.load(data, "random", skill.FIT_YEARS)
    chosen = net_data.load(data, "random", [skill.CHOOSE_YEAR])
    choice = choice_on(net, learn, chosen, log)
    del learn, chosen
    found = measurement(net, data, "random", choice, log)
    found["checks"] = skill.checks(found["results"])
    found["passed"] = found["checks"]["passed"]
    return found


def smoke(net, data: Path, log=print, rows: int = 400_000) -> dict:
    """Random-walk rows of 2016 through Efe's network at each width for one epoch, timed with
    its own Adam, and their forecasts checked to be finite. It shows no score."""
    import torch

    table = net_data.load(data, "random", [skill.FIT_YEARS[0]])
    table = {k: v[:rows] for k, v in table.items()}  # a window looks only back
    on = device()
    mean, scale = standardisation(table, on)
    learn = Rows(table, mean, scale, on)
    batches = math.ceil(learn.count / net_data.BATCH)
    found = {"batch": net_data.BATCH, "rows": learn.count}
    for width in sorted({w for w, _ in SETTINGS}):
        decay = next(d for w, d in SETTINGS if w == width)
        called = []

        def after(epoch, params, called=called):
            called.append(epoch)

        if on.type == "cuda":
            torch.cuda.synchronize(on)
        started = time.perf_counter()
        params = net.learn(learn, width, decay, SEEDS[0], 1, after)
        if on.type == "cuda":
            torch.cuda.synchronize(on)
        step = (time.perf_counter() - started) / batches
        started = time.perf_counter()
        p = forecasts(net, params, learn)
        forecast = (time.perf_counter() - started) / batches
        if called != [1]:
            raise ValueError(
                "learn must call after_epoch(1, params) after its one epoch"
            )
        if p.shape != (learn.count, learn.horizons) or not np.isfinite(p).all():
            raise ValueError(
                "forecast must give a finite number for every row and horizon"
            )
        found[width] = {"step": step, "forecast": forecast}
    found |= net_data.runs(found, net_data.manifest(data, "real"))
    lines = [
        f"width {w}: a batch of {net_data.BATCH:,} learned in {found[w]['step'] * 1e3:.1f} ms, "
        "Adam included"
        for w in sorted({w for w, _ in SETTINGS})
    ]
    lines.append(
        f"decision 0027 at the most: the choice {found['choice hours']:.1f} h, the "
        f"measurement {found['measurement hours']:.1f} h; the placebo as much again"
    )
    log("\n".join(lines))
    return found


# --- the order of the runs ----------------------------------------------------------------


def fingerprint(folder: Path = HERE) -> dict:
    """Each file of CODE's SHA-256, or None for one not written yet."""
    return {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if (folder / name).exists()
        else None
        for name in CODE
    }


def stale(step: str, record: dict, code: dict) -> str | None:
    """Why a step's record does not hold for today's code, or None."""
    changed = [name for name in code if record.get("code", {}).get(name) != code[name]]
    if changed:
        return f"the {step} ran on other code ({', '.join(changed)} changed since); {AGAIN[step]}"
    return None


def refusal(folder: Path, command: str, code: dict) -> str | None:
    """Why `command` may not run yet, or None. The choice waits for a placebo that passed on
    today's code; the measurement waits for that placebo and a choice made on today's code,
    and runs only once."""
    placebo_file, choice_file = folder / "placebo.json", folder / "choice.json"
    if not placebo_file.exists():
        return "run the placebo first"
    record = json.loads(placebo_file.read_text())
    if not record["passed"]:
        return "the placebo failed; nothing runs on real prices until it passes"
    why = stale("placebo", record, code)
    if why or command == "choose":
        return why
    if not choice_file.exists():
        return "choose the setting on 2019 first"
    why = stale("choice", json.loads(choice_file.read_text()), code)
    if why:
        return why
    if (folder / "measure.json").exists():
        return "the measurement has run; it runs only once"
    return None


def efe_net():
    """Efe's network, from research/kutbit/net.py."""
    path = HERE / "net.py"
    if not path.exists():
        raise SystemExit("write research/kutbit/net.py first, from NETWORK_GUIDE.md")
    spec = importlib.util.spec_from_file_location("efe_net", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the page -------------------------------------------------------------------------------


def signed(x, digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:+.{digits}f}".replace("-", "−")


def by_forecaster(results: dict, cell, group: str = "all") -> list[list[str]]:
    return [
        [NAMES[name], *(cell(by_horizon[str(h)].get(group)) for h in HORIZONS)]
        for name, by_horizon in results.items()
    ]


def results_html(found: dict) -> str:
    head = ["", *(f"{h} min" for h in HORIZONS)]

    def corr(kind):
        return lambda r: (
            "—"
            if r is None
            else f"{signed(r[kind]['mean'])} ({signed(r[kind]['t'], 1)})"
        )

    def top(key, t=None):
        def cell(r):
            if r is None:
                return "—"
            v = r["top0.01"][key]
            return signed(v, 2) + (f" ({signed(r['top0.01'][t], 1)})" if t else "")

        return cell

    results = found["results"]
    parts = [
        ("Each day's correlation with z (t)", corr("daily")),
        ("Within each minute, across instruments (t)", corr("across")),
        ("The strongest 1%, bps a signal before costs", top("gross", "t")),
        ("The same after 2 bps", top("net")),
        ("Their own cost, bps a round trip with the hedge's", top("cost")),
        ("After their own cost (t)", top("own", "own_t")),
        ("After twice their own cost (t)", top("twice", "twice_t")),
    ]
    return "".join(
        f"<h3>{html.escape(title)}</h3>"
        + costs.table_html(head, by_forecaster(results, cell))
        for title, cell in parts
    )


def gates_html(g: dict) -> str:
    def said(ok: bool) -> str:
        return "passed" if ok else "failed"

    rows = [
        [
            "2. Its day-by-day margin over the ridge, 15-60 minutes",
            f"{signed(g['over the ridge']['mean'], 4)} (t {signed(g['over the ridge']['t'], 2)})",
            said(g["over the ridge"]["passed"]),
        ],
        [
            "3. Its margin over the network on shuffled labels",
            (
                f"{signed(g['over the shuffled network']['mean'], 4)} "
                f"(t {signed(g['over the shuffled network']['t'], 2)})"
            ),
            said(g["over the shuffled network"]["passed"]),
        ],
        [
            "4. Its strongest 1% after their own cost, at a t of 2, at",
            ", ".join(
                f"{h} min" for h, ok in g["its own cost"]["by_horizon"].items() if ok
            )
            or "no horizon",
            said(g["its own cost"]["passed"]),
        ],
    ]
    return costs.table_html(["gate", "found", ""], rows, numeric_from=3)


def page(folder: Path) -> str:
    """The page of every run made so far."""
    sections = ["<h1>Efe's network: variant 2</h1>"]
    sections.append(
        "<p class=note>Decision 0027. The skill test's rows and inputs; learned on 2016-2018, "
        "its setting and epochs chosen on 2019, learned again on 2016-2019 and measured once "
        "on 2020-2021. A t is in brackets.</p>"
    )
    runs = {n: folder / f"{n}.json" for n in ("measure", "choice", "placebo")}
    found = {n: json.loads(p.read_text()) for n, p in runs.items() if p.exists()}
    if "measure" in found:
        m = found["measure"]
        g = m["gates"]
        verdict = (
            "Variant 2 passed."
            if g["variant passed"]
            else "Variant 2 failed, and is written down as it is."
        )
        sections += [
            f"<h2>Measured once on 2020-2021, {html.escape(m['run_utc'])} UTC</h2>",
            f"<p class=verdict>{verdict}</p>",
            gates_html(g),
            results_html(m),
        ]
    if "choice" in found:
        c = found["choice"]
        head = ["setting", *(str(e) for e in range(1, EPOCHS + 1))]
        rows = [
            [f"{w}, {d:g}", *(signed(s, 4) for s in c["scores"][i])]
            for i, (w, d) in enumerate(SETTINGS)
        ]
        sections += [
            f"<h2>The choice on 2019, {html.escape(c['run_utc'])} UTC</h2>",
            (
                f"<p class=verdict>Kept: setting {c['setting']} (width {c['width']}, decay "
                f"{c['decay']:g}) after {c['epochs']} epochs.</p>"
            ),
            (
                "<p class=note>Each day's correlation with z on 2019, averaged over the days "
                "and over 15, 30 and 60 minutes, of the five seeds' mean, by epoch.</p>"
            ),
            costs.table_html(head, rows),
        ]
    if "placebo" in found:
        pl_ = found["placebo"]
        verdict = (
            "The placebo passed: nothing on the random walks reaches |t| 3."
            if pl_["passed"]
            else "The placebo failed, so the harness is wrong and nothing runs on real "
            "prices: " + "; ".join(pl_["checks"]["failed"])
        )
        sections += [
            f"<h2>The placebo, on random walks, {html.escape(pl_['run_utc'])} UTC</h2>",
            f"<p class=verdict>{html.escape(verdict)}</p>",
            results_html(pl_),
        ]
    return (
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<title>KUTBIT network</title><style>{costs.STYLE}</style></head><body><main>"
        + "".join(sections)
        + "</main></body></html>\n"
    )


def plain(x):
    """`x` with its keys as text and its numbers as plain floats, for JSON."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def write(folder: Path, name: str, record: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(plain(record), indent=1))
    path = folder / "net.html"
    path.write_text(page(folder))
    return path


# --- the commands ---------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    commands = ("smoke", "placebo", "choose", "measure")
    if not argv or argv[0] not in commands:
        raise SystemExit("usage: net_harness.py smoke|placebo|choose|measure [DATA]")
    data = Path(argv[1]) if len(argv) > 1 else DATA
    folder = data / "kutbit" / "net"
    net = efe_net()
    code = fingerprint()  # of the code just loaded
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    started = time.time()
    if argv[0] == "smoke":
        smoke(net, data)
        return
    if argv[0] == "placebo":
        found = placebo(net, data)
        path = write(folder, "placebo", found | {"run_utc": stamp, "code": code})
        verdict = "passes" if found["passed"] else "FAILS"
        print(f"placebo: {verdict} ({time.time() - started:.0f} s); {path}")
        return
    why = refusal(folder, argv[0], code)
    if why:
        raise SystemExit(why)
    if argv[0] == "choose":
        learn = net_data.load(data, "real", skill.FIT_YEARS)
        chosen = net_data.load(data, "real", [skill.CHOOSE_YEAR])
        found = choice_on(net, learn, chosen)
        path = write(folder, "choice", found | {"run_utc": stamp, "code": code})
        print(
            f"choose: setting {found['setting']} after {found['epochs']} epochs kept "
            f"({time.time() - started:.0f} s); {path}"
        )
        return
    choice = json.loads((folder / "choice.json").read_text())
    found = measurement(net, data, "real", choice)
    path = write(folder, "measure", found | {"run_utc": stamp, "code": code})
    verdict = "passed" if found["gates"]["variant passed"] else "failed"
    print(f"measure: variant 2 {verdict} ({time.time() - started:.0f} s); {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
