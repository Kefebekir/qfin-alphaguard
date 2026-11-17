"""How much of a book's market part KUTBIT hedges with QQQ: the evidence behind decision 0025.

KUTBIT forecasts each instrument's move apart from QQQ (`z`, labels.targets), never QQQ's
direction. Hedging the market part takes out risk that earns nothing, but the QQQ leg takes
room under the gross cap that the stocks could use. This study weighs the two on the stored
minutes of 2016-2021, the development years; 2022-2026 are not read.

- **Windows:** each stock's 30-minute windows that start at minutes 0, 30, ..., 330 and end
  before the close, entered at the next bar's open and left at the close HORIZON bars later,
  as the labels are; on days when QQQ's session is usable too. ETFs are left out.
- **An edge is planted** in the part apart from QQQ: `z`, less its mean in that part of the
  day, plus `edge`, times the usual move over the bars held. QQQ's move, less its mean in that
  part of the day, is given none, as KUTBIT never forecasts it.
- **Books:** one stock; ten stocks, all long; ten stocks, five long and five short. The ten are
  drawn at random from the window's stocks, DRAWS times, with equal exposure each.
- **The choice:** the share `u` of the gross cap the book uses, and the share `k` of its
  market part hedged. The stocks' exposure is u * cap / (1 + k * |b|), b being the book's beta
  per unit of exposure; the QQQ leg is k * |b| of it; both legs count towards the cap and pay
  their costs.
- **Its worth:** the account's certainty equivalent over the windows at risk appetite gamma:
  the sure return the account would take instead. gamma 1 is the Kelly criterion, the
  expected log growth; 2 is about half Kelly; 4 and 8 are more careful still. A choice that
  would have wiped the account out in any window is worth -100%, so it is never taken.
- **A day:** for the books of ten, the windows of each day and draw compounded, as if the
  account held such a book all day: how often a day would have ended below a loss.

    PYTHONPATH=src uv run python research/kutbit/hedge_study.py [DATA]

Output: research/kutbit/results_hedge.txt.
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from dataset import MARKET, day_from
from labels import labels, targets
from skill import load

from qfin_alphaguard.kutbit.features import Day

DATA = Path("data")
RESULTS = Path(__file__).parent / "results_hedge.txt"
YEARS = tuple(range(2016, 2022))
LAST_YEAR = 2021  # the last development year; the study never reads a later one
HORIZON = 30  # minutes held
STARTS = tuple(range(0, 331, HORIZON))
PARTS = {"open": (0, 0), "morning": (30, 90), "midday": (120, 270), "late": (300, 330)}
REGIMES = (
    "calm",
    "usual",
    "jumpy",
)  # QQQ's volatility before the open, by thirds of days
EDGES = (0.05, 0.07, 0.1, 0.2, 0.3)  # the forecast's mean z
CAPS = (4.0, 10.0)
GAMMAS = (1.0, 2.0, 4.0, 8.0)
SHARES = tuple(i / 20 for i in range(21))  # u
HEDGES = (1.0, 0.75, 0.5, 0.25, 0.0)  # k, from the full hedge down
BOOKS = ("one stock", "ten long", "five long, five short")
BOOK = 10
DRAWS = 4
SEED = 25
COST_STOCK = 2.0e-4  # a round trip, as in decision 0024's skill test
COST_QQQ = (
    0.5e-4  # a round trip; QQQ's spread is about a cent on several hundred dollars
)
CONDITIONS = (  # book, cap, gamma, edge: the best choice by part of the day and regime
    ("ten long", 10.0, 2.0, 0.1),
    ("ten long", 10.0, 4.0, 0.1),
    ("ten long", 10.0, 4.0, 0.2),
)
DAILY_BOOKS = ("ten long", "five long, five short")
DAILY_EDGE = 0.1
DAILY_HEDGES = (1.0, 0.5, 0.0)
LOSSES = (0.05, 0.10)  # a day's losses whose frequency is counted


# --- the windows ------------------------------------------------------------------------


def windows(day: Day, market: Day) -> dict[str, np.ndarray]:
    """A stock's windows in one session: for each start whose HORIZON bars all come before
    the close, the start, z, the usual move over the bars held, and QQQ's return."""
    found = targets(day, market, (HORIZON,))[HORIZON]
    of_market = labels(market.open, market.high, market.low, market.close, (HORIZON,))[
        HORIZON
    ]
    starts = np.array(
        [t for t in STARTS if t + HORIZON <= len(day.open) - 1], dtype=int
    )
    return {
        "start": starts,
        "z": found["z"][starts],
        "scale": found["scale"][starts],
        "market": of_market["ret"][starts],
    }


def part_of(start: np.ndarray) -> np.ndarray:
    """The part of the day each start falls in, as a name from PARTS."""
    out = np.full(len(start), "", dtype=object)
    for name, (first, last) in PARTS.items():
        out[(start >= first) & (start <= last)] = name
    return out


def collect(days: pl.DataFrame, arrays: dict, years=YEARS) -> pl.DataFrame:
    """Every stock's windows in `years`, one row each, on days when QQQ's session is usable
    too: day, ticker, start, part, beta, z, scale, QQQ's return and QQQ's volatility."""
    if max(years) > LAST_YEAR:
        raise ValueError("the hedge study reads the development years only")
    usable = days.filter(pl.col("usable") & pl.col("year").is_in(list(years)))
    markets = {
        r["day"]: r
        for r in usable.filter(pl.col("ticker") == MARKET).iter_rows(named=True)
    }
    stocks = usable.filter(~pl.col("etf") & pl.col("day").is_in(list(markets)))
    columns = {
        k: []
        for k in ("day", "ticker", "start", "beta", "z", "scale", "market", "qqq_vol")
    }
    for (d,), group in stocks.sort("day", "ticker").group_by(
        "day", maintain_order=True
    ):
        market = day_from(markets[d], arrays)
        for row in group.iter_rows(named=True):
            day = day_from(row, arrays)
            if len(day.open) != len(market.open):
                continue
            found = windows(day, market)
            n = len(found["start"])
            columns["day"].append(np.full(n, d.toordinal()))
            columns["ticker"].append(np.full(n, row["ticker"]).astype(str))
            columns["beta"].append(np.full(n, row["beta"]))
            columns["qqq_vol"].append(np.full(n, markets[d]["vol"]))
            for k in ("start", "z", "scale", "market"):
                columns[k].append(found[k])
    frame = pl.DataFrame({k: np.concatenate(v) for k, v in columns.items()})
    return frame.with_columns(
        pl.Series("part", part_of(frame["start"].to_numpy()), dtype=pl.String)
    )


def centred(frame: pl.DataFrame) -> pl.DataFrame:
    """The windows with the planted edge's ground laid: `own`, the part apart from QQQ with
    the mean of z in its part of the day taken out; `market`, QQQ's return less its mean
    over that part's windows, each window counted once; `regime`, QQQ's volatility by thirds
    of days."""
    z_mean = frame.group_by("part").agg(pl.col("z").mean().alias("z_mean"))
    market_mean = (
        frame.unique(["day", "start"])
        .group_by("part")
        .agg(pl.col("market").mean().alias("market_mean"))
    )
    vols = frame.unique("day")["qqq_vol"].to_numpy()
    low, high = np.quantile(vols, [1 / 3, 2 / 3])
    return (
        frame.join(z_mean, on="part")
        .join(market_mean, on="part")
        .with_columns(
            ((pl.col("z") - pl.col("z_mean")) * pl.col("scale")).alias("own"),
            (pl.col("market") - pl.col("market_mean")).alias("market"),
            pl.when(pl.col("qqq_vol") <= low)
            .then(pl.lit(REGIMES[0]))
            .when(pl.col("qqq_vol") <= high)
            .then(pl.lit(REGIMES[1]))
            .otherwise(pl.lit(REGIMES[2]))
            .alias("regime"),
        )
        .drop("z_mean", "market_mean")
        .sort("day", "start", "ticker")
    )


# --- the books --------------------------------------------------------------------------


def single(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Each stock window as a book of one long position."""
    return {
        "own": frame["own"].to_numpy(),
        "unit": frame["scale"].to_numpy(),
        "beta": frame["beta"].to_numpy(),
        "market": frame["market"].to_numpy(),
        "part": frame["part"].to_numpy(),
        "regime": frame["regime"].to_numpy(),
        "day": frame["day"].to_numpy(),
        "draw": np.zeros(len(frame), dtype=int),
    }


def books(
    frame: pl.DataFrame, size: int, longs: int, draws: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Books of `size` stocks drawn from each window's stocks, `draws` times, `longs` of them
    long and the rest short, with equal exposure: per unit of exposure, the part apart from
    QQQ that each position earns (`own`, before the edge), the usual move the edge is planted
    in (`unit`), and the book's beta (`beta`); with the window's day and the draw."""
    out = {
        k: []
        for k in ("own", "unit", "beta", "market", "part", "regime", "day", "draw")
    }
    keys = frame.select("day", "start").to_numpy()
    own, scale, beta = (frame[c].to_numpy() for c in ("own", "scale", "beta"))
    market, part, regime = (frame[c].to_numpy() for c in ("market", "part", "regime"))
    edges = np.flatnonzero((np.diff(keys, axis=0) != 0).any(axis=1)) + 1
    for first, last in zip(
        np.concatenate([[0], edges]), np.concatenate([edges, [len(keys)]]), strict=True
    ):
        if last - first < size:
            continue
        for draw in range(draws):
            chosen = first + rng.choice(last - first, size, replace=False)
            side = np.where(np.arange(size) < longs, 1.0, -1.0)
            out["own"].append(np.mean(side * own[chosen]))
            out["unit"].append(np.mean(scale[chosen]))
            out["beta"].append(np.mean(side * beta[chosen]))
            out["market"].append(market[first])
            out["part"].append(part[first])
            out["regime"].append(regime[first])
            out["day"].append(keys[first, 0])
            out["draw"].append(draw)
    return {k: np.array(v) for k, v in out.items()}


# --- the choice -------------------------------------------------------------------------


def account(
    book: dict, edge: float, cap: float, share: float, hedge: float
) -> np.ndarray:
    """The account's return in each window: the stocks at u * cap / (1 + k * |b|), their part
    apart from QQQ with the edge planted, the market part left unhedged, and both legs'
    costs."""
    b = book["beta"]
    exposure = share * cap / (1 + hedge * np.abs(b))
    qqq = hedge * np.abs(b) * exposure
    earned = book["own"] + edge * book["unit"] + (1 - hedge) * b * book["market"]
    return exposure * earned - exposure * COST_STOCK - qqq * COST_QQQ


def certainty(returns: np.ndarray, gamma: float) -> float:
    """The sure return worth as much as these returns at risk appetite gamma; -1 if any
    of them wipes the account out."""
    if (returns <= -1).any():
        return -1.0
    if gamma == 1:
        return float(np.expm1(np.log1p(returns).mean()))
    return float(np.mean((1 + returns) ** (1 - gamma)) ** (1 / (1 - gamma)) - 1)


def risk(returns: np.ndarray) -> dict:
    return {
        "std": float(returns.std()),
        "one_in_1000": float(np.quantile(returns, 0.001)),
        "worst": float(returns.min()),
    }


def choose(book: dict, edge: float, cap: float, gamma: float) -> dict:
    """The best share of the cap and share hedged; ties go to the larger hedge and the
    smaller share. With them, the best share at a full hedge and with none."""
    best = {"share": 0.0, "hedge": None, "ce": 0.0}
    by_hedge = {}
    for hedge in HEDGES:
        top = {"share": 0.0, "ce": 0.0}
        for share in SHARES[1:] if len(book["own"]) else ():
            ce = certainty(account(book, edge, cap, share, hedge), gamma)
            if ce > top["ce"]:
                top = {"share": share, "ce": ce}
        by_hedge[hedge] = top
        if top["ce"] > best["ce"]:
            best = {"share": top["share"], "hedge": hedge, "ce": top["ce"]}
    out = {**best, "full": by_hedge[1.0], "none": by_hedge[0.0]}
    if best["hedge"] is not None:
        out["risk"] = risk(account(book, edge, cap, best["share"], best["hedge"]))
    return out


def subset(book: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in book.items()}


# --- a day ------------------------------------------------------------------------------


def daily(
    book: dict, edge: float, cap: float, share: float, hedge: float
) -> np.ndarray:
    """Each day's return of an account that holds such a book in every window of the day,
    one book a draw, its windows compounded; a window that wipes it out ends its day at
    -100%."""
    returns = np.maximum(account(book, edge, cap, share, hedge), -1.0)
    _, index = np.unique(book["day"] * 100 + book["draw"], return_inverse=True)
    with np.errstate(divide="ignore"):
        return np.expm1(np.bincount(index, np.log1p(returns)))


def day_risk(returns: np.ndarray) -> dict:
    out = {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "worst": float(returns.min()),
    }
    for loss in LOSSES:
        out[f"below {loss:g}"] = float((returns <= -loss).mean())
    return out


# --- the run ----------------------------------------------------------------------------


def study(data: Path = DATA, years=YEARS, log=print) -> dict:
    started = time.time()
    days, arrays = load(data, years)
    frame = centred(collect(days, arrays, years))
    log(f"windows: {len(frame):,} ({time.time() - started:.0f} s)")
    rng = np.random.default_rng(SEED)
    all_books = {
        "one stock": single(frame),
        "ten long": books(frame, BOOK, BOOK, DRAWS, rng),
        "five long, five short": books(frame, BOOK, BOOK // 2, DRAWS, rng),
    }
    results = {}
    for name, book in all_books.items():
        for cap in CAPS:
            for gamma in GAMMAS:
                for edge in EDGES:
                    results[(name, cap, gamma, edge)] = choose(book, edge, cap, gamma)
        log(f"{name}: chosen ({time.time() - started:.0f} s)")
    conditions = {}
    for name, cap, gamma, edge in CONDITIONS:
        book = all_books[name]
        for part in PARTS:
            for regime in REGIMES:
                keep = (book["part"] == part) & (book["regime"] == regime)
                conditions[(name, cap, gamma, edge, part, regime)] = choose(
                    subset(book, keep), edge, cap, gamma
                )
    log(f"conditions: chosen ({time.time() - started:.0f} s)")
    days_held = {
        (name, cap, hedge): day_risk(
            daily(all_books[name], DAILY_EDGE, cap, 1.0, hedge)
        )
        for name in DAILY_BOOKS
        for cap in CAPS
        for hedge in DAILY_HEDGES
    }
    return {
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - started),
        "years": (min(years), max(years)),
        "windows": len(frame),
        "books": {k: len(v["own"]) for k, v in all_books.items()},
        "betas": {k: float(np.mean(np.abs(v["beta"]))) for k, v in all_books.items()},
        "usual_bps": float(frame["scale"].mean() * 1e4),
        "results": results,
        "conditions": conditions,
        "daily": days_held,
    }


# --- the report -------------------------------------------------------------------------


def bps(x: float) -> str:
    return f"{x * 1e4:+.1f}"


def cell(found: dict) -> str:
    if found["hedge"] is None:
        return "no trade"
    return f"u {found['share']:.2f} k {found['hedge']:.2f} {bps(found['ce'])}"


def header(first: str, columns) -> str:
    return f"  {first:<10}" + "".join(f"{c:>22}" for c in columns)


def report(found: dict) -> str:
    lines = [
        "KUTBIT's hedge study (decision 0025)",
        (
            f"Stored minutes of {found['years'][0]}-{found['years'][1]}; 2022-2026 not "
            f"read. Run {found['run_utc']} UTC in {found['seconds']} s."
        ),
        (
            f"{found['windows']:,} stock windows of {HORIZON} minutes, starting at "
            f"minutes {STARTS[0]}, {STARTS[1]}, ..., {STARTS[-1]}; books of {BOOK}: "
        )
        + ", ".join(f"{k} {v:,}" for k, v in found["books"].items() if k != "one stock")
        + f" ({DRAWS} draws a window).",
        "Mean |beta| per unit of exposure: "
        + ", ".join(f"{k} {v:.2f}" for k, v in found["betas"].items())
        + ".",
        (
            f"Costs a round trip: stocks {COST_STOCK * 1e4:.1f} bps, QQQ "
            f"{COST_QQQ * 1e4:.1f} bps; both legs count towards the cap. QQQ is given "
            "no edge."
        ),
        (
            f"The usual move apart from QQQ over {HORIZON} minutes is "
            f"{found['usual_bps']:.1f} bps on average, so an edge of 0.1 is about "
            f"{0.1 * found['usual_bps']:.1f} bps."
        ),
        "",
        "u: the share of the cap used; k: the share of the market part hedged; then the",
        "account's certainty equivalent in bps per window. gamma is the risk appetite: 1 is",
        "Kelly, 2 about half Kelly, 4 and 8 more careful still.",
    ]
    gammas = [f"gamma {g:g}" for g in GAMMAS]
    for name in BOOKS:
        for cap in CAPS:
            lines += [
                "",
                f"{name}, cap {cap:g}x: the best choice",
                header("edge", gammas),
            ]
            for edge in EDGES:
                row = [found["results"][(name, cap, g, edge)] for g in GAMMAS]
                lines.append(f"  {edge:<10g}" + "".join(f"{cell(r):>22}" for r in row))
            lines += [f"{name}, cap {cap:g}x: hedged in full | not at all"]
            for edge in EDGES:
                row = [found["results"][(name, cap, g, edge)] for g in GAMMAS]
                lines.append(
                    f"  {edge:<10g}"
                    + "".join(
                        f"{bps(r['full']['ce']) + ' | ' + bps(r['none']['ce']):>22}"
                        for r in row
                    )
                )
    for name, cap, gamma, edge in CONDITIONS:
        lines += [
            "",
            (
                f"When to hedge: {name}, cap {cap:g}x, gamma {gamma:g}, edge {edge:g}; by "
                "part of the day and QQQ's volatility before the open"
            ),
            header("", REGIMES),
        ]
        for part in PARTS:
            row = [
                found["conditions"][(name, cap, gamma, edge, part, r)] for r in REGIMES
            ]
            lines.append(f"  {part:<10}" + "".join(f"{cell(c):>22}" for c in row))
    lines += [
        "",
        (
            f"A day of such books in every window, edge {DAILY_EDGE:g}, the whole cap "
            "used: the day's return in %"
        ),
        f"  {'book':<22}{'cap':>5}{'k':>6}{'mean':>8}{'std':>8}{'worst':>9}"
        + "".join(
            f"{'days <= -' + format(loss * 100, 'g') + '%':>16}" for loss in LOSSES
        ),
    ]
    for name in DAILY_BOOKS:
        for cap in CAPS:
            for hedge in DAILY_HEDGES:
                r = found["daily"][(name, cap, hedge)]
                lines.append(
                    f"  {name:<22}{cap:>4g}x{hedge:>6.2f}{r['mean'] * 100:>8.2f}"
                    f"{r['std'] * 100:>8.2f}{r['worst'] * 100:>9.2f}"
                    + "".join(
                        f"{r[f'below {loss:g}'] * 100:>15.2f}%" for loss in LOSSES
                    )
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    data = Path(argv[0]) if argv else DATA
    found = study(data)
    text = report(found)
    RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
