"""KUTBIT's skill test: how much of `z` a linear model of the inputs forecasts (decision 0024, 9).

It is a ruler, not a variant: it is never traded and does not count among the ten. Efe's
network has to beat it on the same years. Its rules were written in decision 0024 first:

- **Model:** for each horizon, a ridge regression of `z` (labels.targets) on everything the
  network sees at a minute: the 120-minute window's channels and the scalars, 971 inputs.
  Inputs are clipped at CLIP_INPUT and `z` at CLIP_TARGET, standardised on the fitted rows.
- **Rows:** every minute but the last of every usable session of every instrument but QQQ,
  the hedge, on days when QQQ's session is usable too.
- **Years:** fitted on FIT_YEARS for each of PENALTIES; for each horizon the penalty whose
  forecasts correlate best with `z` on CHOOSE_YEAR is kept, and the model is fitted again on
  FIT_YEARS and CHOOSE_YEAR together; that model is measured once on MEASURE_YEARS. The penalty
  and the thresholds are fixed before any minute of MEASURE_YEARS is read. 2022-2026 are not
  read.
- **Measures on MEASURE_YEARS,** for each horizon, beside the two known patterns (`habit`, and
  the last 30 minutes apart from the market, reversed):
  - each day's correlation of forecast and clipped `z`, about the means of all the measured
    rows, averaged over the days, and its t; not about each day's own means, which made a
    random walk look like it reverses (decision 0024, amendment of 17 November 2026);
  - the same within each minute across instruments (at least MIN_ACROSS), which takes out what
    is common to them all, averaged over each day's minutes and then over the days;
  - the hedged return of trading in the forecast's direction on its STRONGEST forecasts, by
    thresholds of its size set on CHOOSE_YEAR, in bps a trade before and after COST, with a t
    clustered by day. Consecutive minutes of one signal are one trade in practice, so the
    count is of signals, not trades.
- **The placebo, which must show nothing, or the harness is wrong:** `random`, every
  session's prices replaced by a random walk with each minute's usual move, run through the
  same code. No forecaster may reach |t| >= PLACEBO_T at any horizon in any measure. It is
  run first; `real` refuses to run until it passes.

Efe runs it, first a check of speed that shows no score:

    OMP_NUM_THREADS=8 uv run python research/kutbit/skill.py smoke [DATA]
    OMP_NUM_THREADS=8 uv run python research/kutbit/skill.py random [DATA]
    OMP_NUM_THREADS=8 uv run python research/kutbit/skill.py real [DATA]

Output: DATA/kutbit/skill/<random|real>.json and DATA/kutbit/skill/skill.html.
"""

import html
import json
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from dataset import MARKET, day_from
from labels import HORIZONS, targets

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, WINDOW, Day, inputs

DATA = Path("data")
FIT_YEARS = (2016, 2017, 2018)
CHOOSE_YEAR = 2019
MEASURE_YEARS = (2020, 2021)
PENALTIES = (0.001, 0.01, 0.1, 1.0, 10.0)  # on the inputs' correlation matrix
CLIP_INPUT, CLIP_TARGET = 10.0, 5.0
COST = (
    2.0  # bps a round trip, decision 0023's figure; the hedge's own cost is not in it
)
STRONGEST = (0.10, 0.01)
MIN_ACROSS = 10
PLACEBO_T = 3.0
REVERSAL = 30  # minutes of the window summed for the reversal pattern
SEED = 24
INPUTS = WINDOW * len(CHANNELS) + len(SCALARS)
PATTERNS = ("habit", "reversal")


# --- the rows -----------------------------------------------------------------------------


@dataclass
class Rows:
    """One day's rows, every eligible instrument's minutes but the last, in one block."""

    x: np.ndarray  # [rows, INPUTS] float32, clipped
    z: np.ndarray  # [rows, horizons], clipped
    apart: np.ndarray  # [rows, horizons], the hedged return, unclipped
    patterns: (
        np.ndarray
    )  # [rows, 2]: habit, and the last REVERSAL minutes apart, reversed
    minute: np.ndarray  # [rows] int
    instrument: np.ndarray  # [rows] int, an index into the run's list of tickers
    etf: np.ndarray  # [rows] bool


def instrument_rows(day: Day, market: Day) -> dict[str, np.ndarray]:
    """An instrument's rows for one session: what it sees at the close of each bar but the
    last, and what followed."""
    windows, scalars = inputs(day, market)
    n = len(day.open)
    x = np.concatenate([windows.reshape(n, -1), scalars], axis=1)[:-1]
    found = targets(day, market)
    z = np.stack([found[h]["z"][:-1] for h in HORIZONS], axis=1)
    apart = np.stack([found[h]["apart"][:-1] for h in HORIZONS], axis=1)
    if not (np.isfinite(x).all() and np.isfinite(z).all()):
        raise ValueError("a session's inputs and targets must be finite")
    apart_channel = windows[:-1, -REVERSAL:, CHANNELS.index("apart")]
    patterns = np.stack(
        [scalars[:-1, SCALARS.index("habit")], -apart_channel.sum(axis=1)], axis=1
    )
    return {
        "x": np.clip(x, -CLIP_INPUT, CLIP_INPUT).astype(np.float32),
        "z": np.clip(z, -CLIP_TARGET, CLIP_TARGET),
        "apart": apart,
        "patterns": patterns,
    }


def random_walk(day: Day, rng: np.random.Generator) -> Day:
    """The session with its prices replaced by a random walk from its open: each minute's
    step has the usual move at that minute as its size and a Student t's fat tails. Nothing
    in it can be forecast; everything known before the open, and the volumes, are kept."""
    n = len(day.open)
    swing = np.asarray(day.swing)[:n]
    steps = (
        rng.standard_t(4, n) / math.sqrt(2) * swing
    )  # a t with 4 degrees has variance 2
    close = day.open[0] * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[day.open[0]], close[:-1]])
    wick = np.abs(rng.standard_t(4, (2, n))) / math.sqrt(2) * swing / 2
    return replace(
        day,
        open=open_,
        close=close,
        high=np.maximum(open_, close) * np.exp(wick[0]),
        low=np.minimum(open_, close) * np.exp(-wick[1]),
    )


def eligible(days: pl.DataFrame, years) -> pl.DataFrame:
    """The usable sessions of every instrument but the market, in `years`, on days when the
    market's session is usable too."""
    market_days = days.filter((pl.col("ticker") == MARKET) & pl.col("usable"))["day"]
    return days.filter(
        pl.col("year").is_in(list(years))
        & pl.col("usable")
        & (pl.col("ticker") != MARKET)
        & pl.col("day").is_in(market_days.to_list())
    ).sort("day", "ticker")


def day_rows(
    rows: list[dict],
    market_row: dict,
    arrays: dict,
    tickers: dict[str, int],
    source: str,
) -> Rows | None:
    """One day's rows of every eligible instrument; `source` is "real" or "random"."""
    market = day_from(market_row, arrays)
    if source == "random":
        seed = [SEED, market_row["day"].toordinal()]
        market = random_walk(market, np.random.default_rng(seed))
    parts = []
    for row in rows:
        day = day_from(row, arrays)
        if source == "random":
            seed = [SEED, row["day"].toordinal(), tickers[row["ticker"]] + 1]
            day = random_walk(day, np.random.default_rng(seed))
        part = instrument_rows(day, market)
        n = len(part["z"])
        part["minute"] = np.arange(n)
        part["instrument"] = np.full(n, tickers[row["ticker"]])
        part["etf"] = np.full(n, row["etf"])
        parts.append(part)
    if not parts:
        return None
    return Rows(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def each_day(
    days: pl.DataFrame,
    arrays: dict,
    years,
    tickers: dict[str, int],
    source: str,
    limit: int | None = None,
) -> Iterator[tuple[object, Rows]]:
    """Every day's rows in `years`, oldest first; `limit` keeps each year's first days."""
    market = {
        r["day"]: r
        for r in days.filter(
            (pl.col("ticker") == MARKET) & pl.col("year").is_in(list(years))
        ).iter_rows(named=True)
    }
    taken: dict[int, int] = {}
    for (day,), part in eligible(days, years).group_by("day", maintain_order=True):
        taken[day.year] = taken.get(day.year, 0) + 1
        if limit is not None and taken[day.year] > limit:
            continue
        found = day_rows(
            list(part.iter_rows(named=True)), market[day], arrays, tickers, source
        )
        if found is not None:
            yield day, found


# --- the ridge ----------------------------------------------------------------------------


@dataclass
class Moments:
    """What a least-squares fit needs of its rows: their count, sums and cross products."""

    n: int = 0
    s: np.ndarray = field(default_factory=lambda: np.zeros(INPUTS))
    q: np.ndarray = field(default_factory=lambda: np.zeros((INPUTS, INPUTS)))
    r: np.ndarray = field(default_factory=lambda: np.zeros((INPUTS, len(HORIZONS))))
    t: np.ndarray = field(default_factory=lambda: np.zeros(len(HORIZONS)))

    def add(self, x: np.ndarray, z: np.ndarray) -> None:
        self.n += len(x)
        self.s += x.sum(axis=0, dtype=np.float64)
        self.q += (x.T @ x).astype(np.float64)
        # in float64: the forecastable part of z is small, and this is where it shows
        self.r += x.T.astype(np.float64) @ z
        self.t += z.sum(axis=0)

    def __add__(self, other: "Moments") -> "Moments":
        return Moments(
            self.n + other.n,
            self.s + other.s,
            self.q + other.q,
            self.r + other.r,
            self.t + other.t,
        )


@dataclass(frozen=True)
class Ridge:
    """A forecast for each horizon: intercept + x @ beta."""

    intercept: np.ndarray  # [horizons]
    beta: np.ndarray  # [INPUTS, horizons]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + x.astype(np.float64) @ self.beta


def fit(m: Moments, penalty: float) -> Ridge:
    """The ridge regression of each horizon's target on the inputs, standardised: on their
    correlation matrix R, (R + penalty I) b = the inputs' covariance with the target over
    their standard deviations. An input that never varies gets no weight."""
    mean = m.s / m.n
    cov = m.q / m.n - np.outer(mean, mean)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    live = sd > 1e-9
    k = int(live.sum())
    corr = cov[np.ix_(live, live)] / np.outer(sd[live], sd[live])
    target_mean = m.t / m.n
    cross = (m.r[live] / m.n - np.outer(mean[live], target_mean)) / sd[live, None]
    b = np.linalg.solve(corr + penalty * np.eye(k), cross)
    beta = np.zeros((len(mean), m.r.shape[1]))
    beta[live] = b / sd[live, None]
    return Ridge(target_mean - mean @ beta, beta)


# --- the measures -------------------------------------------------------------------------


def correlation(p: np.ndarray, z: np.ndarray) -> float:
    p, z = p - p.mean(), z - z.mean()
    denominator = math.sqrt(float((p * p).sum() * (z * z).sum()))
    return float((p * z).sum()) / denominator if denominator > 0 else float("nan")


def mean_and_t(values: np.ndarray) -> dict:
    """The mean of independent values, its t and their count; NaNs are left out."""
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2 or values.std(ddof=1) == 0:
        return {
            "mean": float(values.mean()) if n else float("nan"),
            "t": float("nan"),
            "n": n,
        }
    t = values.mean() / values.std(ddof=1) * math.sqrt(n)
    return {"mean": float(values.mean()), "t": float(t), "n": n}


def grouped_correlations(
    p, z, groups, least: int = 2, about_all: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Each group's correlation of p and z, for groups of at least `least` rows: (the groups'
    keys, their correlations). About each group's own means, or with `about_all` about the
    means of all the rows."""
    keys, index, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if about_all:
        dp, dz = p - p.mean(), z - z.mean()
    else:
        size = counts.astype(float)
        dp = p - (np.bincount(index, p) / size)[index]
        dz = z - (np.bincount(index, z) / size)[index]
    spz = np.bincount(index, dp * dz)
    spp = np.bincount(index, dp * dp)
    szz = np.bincount(index, dz * dz)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = spz / np.sqrt(spp * szz)
    keep = (counts >= least) & (spp > 0) & (szz > 0)
    return keys[keep], corr[keep]


def daily_correlation(p, z, day) -> dict:
    """Each day's correlation of forecast and target about the means of all the rows, then
    their mean over the days and t.

    Not about each day's own means: every instrument of a day shares QQQ's path, and once a
    day's mean is taken out, what went before a minute and what followed it move against each
    other, even in a random walk. Over a day the path's ups and downs add up to the day's
    move, so a rise above the day's mean before a minute leaves less than the mean after it.
    The last 30 minutes reversed then 'forecast' a random walk at t +5.8 to +8.7 (decision
    0024, amendment of 17 November 2026)."""
    _, corr = grouped_correlations(p, z, day, about_all=True)
    return mean_and_t(corr)


def across_correlation(p, z, day, minute) -> dict:
    """At each minute, the correlation across instruments; each day's mean of them; then the
    mean over the days and its t."""
    keys, corr = grouped_correlations(p, z, day * 1000 + minute, least=MIN_ACROSS)
    _, index = np.unique(keys // 1000, return_inverse=True)
    per_day = np.bincount(index, corr) / np.bincount(index)
    return mean_and_t(per_day)


def clustered_mean(values: np.ndarray, groups: np.ndarray) -> dict:
    """The mean of values, with a t whose error allows values of one group to move together."""
    n = len(values)
    if n < 2:
        return {"mean": float("nan"), "t": float("nan"), "n": n}
    mean = values.mean()
    _, index = np.unique(groups, return_inverse=True)
    sums = np.bincount(index, values - mean)
    se = math.sqrt(float((sums**2).sum())) / n
    return {
        "mean": float(mean),
        "t": float(mean / se) if se > 0 else float("nan"),
        "n": n,
    }


def strongest(p, apart_bps, day, threshold: float) -> dict:
    """Trading in the forecast's direction where its size reaches `threshold`: the hedged
    return in bps a signal, before and after COST, with a t clustered by day."""
    chosen = np.abs(p) >= threshold
    gross = np.sign(p[chosen]) * apart_bps[chosen]
    found = clustered_mean(gross, day[chosen])
    days = len(np.unique(day))
    return {
        "gross": found["mean"],
        "net": found["mean"] - COST,
        "t": found["t"],
        "signals_a_day": float(chosen.sum()) / days if days else float("nan"),
    }


def measures(p, z, apart, day, minute, thresholds: dict) -> dict:
    """Every measure of one forecaster at one horizon on the measured rows."""
    out = {
        "daily": daily_correlation(p, z, day),
        "across": across_correlation(p, z, day, minute),
        "slope": float((p - p.mean()) @ z / ((p - p.mean()) @ (p - p.mean())))
        if p.std() > 0
        else float("nan"),
    }
    for share in STRONGEST:
        out[f"top{share:g}"] = strongest(p, apart * 1e4, day, thresholds[share])
    return out


def thresholds_of(p: np.ndarray) -> dict:
    """The size a forecast must reach to be among its STRONGEST share, from these forecasts."""
    return {share: float(np.quantile(np.abs(p), 1 - share)) for share in STRONGEST}


# --- the run ------------------------------------------------------------------------------


def load(data: Path, years) -> tuple[pl.DataFrame, dict]:
    days = pl.read_parquet(data / "kutbit" / "days.parquet")
    arrays = {}
    for year in years:  # read once: a field of an .npz file is unpacked at every access
        with np.load(data / "kutbit" / f"{year}.npz") as stored:
            arrays[year] = {name: stored[name] for name in stored.files}
    return days, arrays


def ridge_of(models: list[Ridge], choice: dict) -> Ridge:
    """One model whose forecast for each horizon is that of the penalty chosen for it."""
    picked = [models[PENALTIES.index(choice[h]["penalty"])] for h in HORIZONS]
    return Ridge(
        np.array([m.intercept[j] for j, m in enumerate(picked)]),
        np.stack([m.beta[:, j] for j, m in enumerate(picked)], axis=1),
    )


def choose(early, final, patterns, z) -> tuple[dict, dict]:
    """On CHOOSE_YEAR: for each horizon the penalty whose early model's forecasts correlate
    best with z, and the thresholds of the strongest forecasts, from the final model at that
    penalty and from each pattern. `early` and `final` are [rows, horizons, penalties]."""
    choice, thresholds = {}, {}
    for j, h in enumerate(HORIZONS):
        scores = [correlation(early[:, j, i], z[:, j]) for i in range(len(PENALTIES))]
        best = int(np.nanargmax(scores))
        choice[h] = {"penalty": PENALTIES[best], "scores": scores}
        thresholds[h] = thresholds_of(final[:, j, best])
    for i, name in enumerate(PATTERNS):
        thresholds[name] = thresholds_of(patterns[:, i])
    return choice, thresholds


def run(data: Path, source: str, limit: int | None = None, log=print) -> dict:
    """The test on `source` ("real" or "random"): the fits, the choice on CHOOSE_YEAR, and
    the measures on MEASURE_YEARS. With `limit`, each year's first days only."""
    started = time.time()
    fitted_years = (*FIT_YEARS, CHOOSE_YEAR)
    days, arrays = load(data, (*fitted_years, *MEASURE_YEARS))
    tickers = {t: i for i, t in enumerate(sorted(days["ticker"].unique()))}

    # 1. the moments of each year to be fitted
    fitted = {y: Moments() for y in fitted_years}
    rows = 0
    for day, r in each_day(days, arrays, fitted_years, tickers, source, limit):
        fitted[day.year].add(r.x, r.z)
        rows += len(r.x)
    log(f"fitted rows: {rows:,} ({time.time() - started:.0f} s)")
    early = sum((fitted[y] for y in FIT_YEARS), Moments())
    early_models = [fit(early, penalty) for penalty in PENALTIES]
    final_models = [fit(early + fitted[CHOOSE_YEAR], penalty) for penalty in PENALTIES]

    # 2. the penalty and the thresholds, on CHOOSE_YEAR, before MEASURE_YEARS are read
    seen = {k: [] for k in ("early", "final", "patterns", "z")}
    for _, r in each_day(days, arrays, [CHOOSE_YEAR], tickers, source, limit):
        seen["early"].append(np.stack([m(r.x) for m in early_models], axis=2))
        seen["final"].append(np.stack([m(r.x) for m in final_models], axis=2))
        seen["patterns"].append(r.patterns)
        seen["z"].append(r.z)
    seen = {k: np.concatenate(v) for k, v in seen.items()}
    choice, thresholds = choose(
        seen["early"], seen["final"], seen["patterns"], seen["z"]
    )
    model = ridge_of(final_models, choice)
    del seen
    log(f"penalties chosen on {CHOOSE_YEAR} ({time.time() - started:.0f} s)")

    # 3. the measures, on MEASURE_YEARS, read once
    kept = {k: [] for k in ("ridge", "patterns", "z", "apart", "day", "minute", "etf")}
    for day, r in each_day(days, arrays, MEASURE_YEARS, tickers, source, limit):
        kept["ridge"].append(model(r.x))
        kept["patterns"].append(r.patterns)
        kept["z"].append(r.z)
        kept["apart"].append(r.apart)
        kept["day"].append(np.full(len(r.x), day.toordinal()))
        kept["minute"].append(r.minute)
        kept["etf"].append(r.etf)
    m = {k: np.concatenate(v) for k, v in kept.items()}
    log(f"measured rows: {len(m['day']):,} ({time.time() - started:.0f} s)")

    forecasts = {"ridge": m["ridge"]}
    for i, name in enumerate(PATTERNS):
        forecasts[name] = np.repeat(m["patterns"][:, i : i + 1], len(HORIZONS), axis=1)
    groups = {"all": slice(None), "stocks": ~m["etf"], "etfs": m["etf"]}
    results = {}
    for name, p in forecasts.items():
        results[name] = {}
        for j, h in enumerate(HORIZONS):
            limits = thresholds[name] if name in PATTERNS else thresholds[h]
            results[name][h] = {
                g: measures(
                    p[s, j],
                    m["z"][s, j],
                    m["apart"][s, j],
                    m["day"][s],
                    m["minute"][s],
                    limits,
                )
                for g, s in groups.items()
                if len(m["day"][s])
            }
    return {
        "source": source,
        "limit": limit,
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - started),
        "fitted_rows": rows,
        "measured_rows": len(m["day"]),
        "choice": {str(h): c for h, c in choice.items()},
        "results": {n: {str(h): v for h, v in r.items()} for n, r in results.items()},
        "checks": checks(results) if source == "random" else None,
    }


def checks(results: dict) -> dict:
    """The placebo rule, on random walks: no forecaster may reach |t| >= PLACEBO_T at any
    horizon, in any measure."""
    failed = []
    for name, by_horizon in results.items():
        for h, groups in by_horizon.items():
            found = groups["all"]
            ts = {"daily": found["daily"]["t"], "across": found["across"]["t"]}
            ts |= {f"top{s:g}": found[f"top{s:g}"]["t"] for s in STRONGEST}
            failed += [
                f"{name}, {h} min, {k}: t {t:+.1f}"
                for k, t in ts.items()
                if math.isfinite(t) and abs(t) >= PLACEBO_T
            ]
    return {"passed": not failed, "failed": failed}


# --- the page -----------------------------------------------------------------------------

NAMES = {
    "habit": "Known pattern: habit",
    "reversal": f"Known pattern: the last {REVERSAL} minutes reversed",
    "ridge": "Ridge on the 971 inputs",
}


def signed(x: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(x) else f"{x:+.{digits}f}".replace("-", "−")


def table(result: dict, title: str, cell, group: str = "all") -> str:
    head = "".join(f"<th class='num'>{h} min</th>" for h in HORIZONS)
    rows = ""
    for name, by_horizon in result["results"].items():
        cells = "".join(
            f"<td class='num'>{cell(by_horizon[str(h)][group])}</td>" for h in HORIZONS
        )
        rows += f"<tr><td>{html.escape(NAMES[name])}</td>{cells}</tr>"
    return (
        f"<h2>{html.escape(title)}</h2><div class='tables'><table><thead><tr><th></th>{head}"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def section(result: dict) -> str:
    years = f"{MEASURE_YEARS[0]}-{MEASURE_YEARS[-1]}"
    label = (
        "Real prices" if result["source"] == "real" else "Placebo: random-walk prices"
    )
    if result["checks"] is None:
        verdict = "Run after the random-walk placebo passed."
    elif result["checks"]["passed"]:
        verdict = "The placebo rule holds: nothing here reaches |t| 3."
    else:
        verdict = (
            "The placebo rule fails, so the harness is wrong and no result may be read: "
            + "; ".join(result["checks"]["failed"])
        )
    penalties = ", ".join(
        f"{h} min {result['choice'][str(h)]['penalty']:g}" for h in HORIZONS
    )
    out = (
        f"<h2 class='run'>{label}</h2><ul class='summary'>"
        f"<li>{html.escape(verdict)}</li>"
        f"<li>Fitted on {result['fitted_rows']:,} rows of {FIT_YEARS[0]}-{CHOOSE_YEAR}; "
        f"measured on {result['measured_rows']:,} rows of {years}; run {result['run_utc']} UTC "
        f"in {result['seconds'] / 60:.0f} minutes.</li>"
        f"<li>Penalties chosen on {CHOOSE_YEAR}: {penalties}.</li></ul>"
    )

    def corr(found):
        return (
            f"{signed(found['mean'])} <span class='t'>t {signed(found['t'], 1)}</span>"
        )

    out += table(
        result,
        f"Each day's correlation with z, about all rows' means, {years}",
        lambda g: corr(g["daily"]),
    )
    out += table(
        result,
        "The same within each minute, across instruments",
        lambda g: corr(g["across"]),
    )
    for share in STRONGEST:
        key = f"top{share:g}"
        out += table(
            result,
            f"The strongest {share:.0%} of forecasts: hedged bps a signal, after {COST:g} bps",
            lambda g, key=key: (
                f"{signed(g[key]['net'], 1)} <span class='t'>gross {signed(g[key]['gross'], 1)}, "
                f"t {signed(g[key]['t'], 1)}, {g[key]['signals_a_day']:.0f} a day</span>"
            ),
        )
    for group, name in (("stocks", "stocks"), ("etfs", "ETFs")):
        out += table(
            result,
            f"Each day's correlation with z, {name} alone",
            lambda g: corr(g["daily"]),
            group,
        )
    return out


def page(results: list[dict]) -> str:
    body = "".join(section(r) for r in results)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KUTBIT skill test</title>
<style>
:root {{ color-scheme: light; --surface: #fcfcfb; --grid: #e4e3df; --text: #0b0b0b;
  --muted: #77766f; --secondary: #52514e; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
  --surface: #1a1a19; --grid: #33332f; --text: #f0efec; --muted: #9a998f; --secondary: #c3c2b7; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface: #1a1a19; --grid: #33332f;
  --text: #f0efec; --muted: #9a998f; --secondary: #c3c2b7; }}
body {{ margin: 0; background: var(--surface); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 24px 0 8px; }}
h2.run {{ font-size: 18px; margin-top: 36px; border-top: 1px solid var(--grid); padding-top: 16px; }}
ul.summary {{ padding-left: 18px; color: var(--secondary); }}
.tables {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--secondary); font-weight: 600; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.t {{ color: var(--muted); font-size: 12px; }}
.note {{ color: var(--muted); font-size: 12px; }}
</style></head><body><main>
<h1>KUTBIT skill test</h1>
<p class="note">Decision 0024, section 9. A ruler, not a variant: Efe's network has to beat the
ridge's correlation on the same years. The correlations use z clipped at ±{CLIP_TARGET:g}; the bps
are the hedged return, unclipped. A signal is one minute's forecast, so one trade may span several
signals. The hedge's own cost is not in the {COST:g} bps.</p>
{body}
</main></body></html>
"""


def write(data: Path, result: dict) -> Path:
    out = data / "kutbit" / "skill"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['source']}.json").write_text(json.dumps(result, indent=1))
    found = [
        json.loads((out / f"{s}.json").read_text())
        for s in ("real", "random")
        if (out / f"{s}.json").exists()
    ]
    (out / "skill.html").write_text(page(found))
    return out / "skill.html"


def smoke(data: Path) -> None:
    """The whole test on each year's first 3 days, to check speed and memory; no score."""
    result = run(data, "random", limit=3, log=lambda s: print("smoke:", s))
    per_day = result["seconds"] / (3 * 7)
    print(
        f"smoke: {result['fitted_rows']:,} rows fitted, {result['measured_rows']:,} measured, "
        f"{result['seconds']} s; a whole run takes about {per_day * 252 * 7 / 60:.0f} minutes"
    )


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("smoke", "random", "real"):
        raise SystemExit("usage: skill.py smoke|random|real [DATA]")
    data = Path(argv[1]) if len(argv) > 1 else DATA
    if argv[0] == "smoke":
        smoke(data)
        return
    if argv[0] == "real":
        placebo = data / "kutbit" / "skill" / "random.json"
        if (
            not placebo.exists()
            or not json.loads(placebo.read_text())["checks"]["passed"]
        ):
            raise SystemExit(
                "run the random-walk placebo first; real runs only once it passes"
            )
    result = run(data, argv[0])
    path = write(data, result)
    if argv[0] == "random":
        verdict = "passes" if result["checks"]["passed"] else "FAILS"
        print(f"random: the placebo rule {verdict}; {path}")
    else:
        print(f"real: done; {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
