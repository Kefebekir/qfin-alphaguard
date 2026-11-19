"""The event gym of KUTBIT's first agent (decision 0026).

At each fresh extreme minute of a stock (events.py), the agent sees STATE, 16 numbers known at
that minute's close, each turned to the move's direction where it has one, and chooses one of
ACTIONS: let the move be, or a side, a size, a hold and a hedge. Every action's outcome is in
the stored minutes, so the gym gives them all at once:

- `own` and `qqq`: for each hold, the stock's return and QQQ's, from the next bar's open to the
  close of the hold's last bar or of the session's last bar, each signed by the move's
  direction;
- `returns`: the account's return for every action, after a stock's round trip of COST and
  QQQ's of COST_QQQ, the hedge leg included;
- `allowed`: the actions within the position cap and Efe's limit M on the market part left
  unhedged;
- `utility`: the account's utility at Efe's risk appetite GAMMA, -inf where the account would
  be wiped out.

Beside the real sessions it builds the same events on random walks (the placebo) and deals
outcomes between events (the shuffled bar), and it gives decision 0026's fixed bars and best
simple policy. It reads at most 2016-2021; 2022-2026 are never read.

    PYTHONPATH=src uv run python research/kutbit/gym.py build [DATA]   # Efe: 2016-2021
    PYTHONPATH=src uv run python research/kutbit/gym.py learning [DATA]  # 2016-2018 only

`build` writes DATA/kutbit/agent/<real|random>/<year>.npz. `learning` builds 2016-2018 in
memory and writes results_gym.txt: the events and the bars on the years an agent learns on.
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from dataset import MARKET, day_from
from events import COOLDOWN, MARKET_SHOCK, NEWS_MINUTES, PARTS, fresh
from events import SIZES as EXTREME
from labels import labels
from skill import SEED, load, random_walk

from qfin_alphaguard.kutbit.features import CHANNELS, SCALARS, Day, channels, scalars

DATA = Path("data")
RESULTS = Path(__file__).parent / "results_gym.txt"
LEARN = (2016, 2017, 2018)
CHOOSE = (2019,)
MEASURE = (2020, 2021)
YEARS = (*LEARN, *CHOOSE, *MEASURE)
SOURCES = ("real", "random")
SIZES = (0.25, 0.5, 1.0, 2.0, 4.0)  # times the account's equity
HOLDS = (5, 15, 30)  # minutes
HEDGES = (0.0, 0.5, 1.0)  # share of the position's market part hedged with QQQ
CAP = 4.0  # the position cap, times equity
M = 2.0  # the most market exposure a position may leave unhedged, times equity
GAMMA = 2.0  # Efe's risk appetite: about half Kelly
COST = 5e-4  # a stock's round trip
COSTS = (2e-4, 5e-4, 10e-4)  # reported beside it; 10 bps is costs doubled
COST_QQQ = 0.5e-4  # QQQ's round trip
CHART = 30  # minutes kept on each side of an extreme minute, for the page
MIN_KIND = 100  # events of a kind before the best simple policy acts on it
STATE = (
    "size",  # the move apart from QQQ, in usual moves
    "direction",  # +1 up, -1 down
    "range",  # the minute's high over low, in usual moves
    "volume",  # log of the minute's volume over its usual
    "qqq",  # QQQ's move that minute in its usual moves, + the stock's way
    "headlines",  # log(1 + headlines naming the stock in the 30 minutes up to it)
    "vwap",  # the close's distance from the session's VWAP, + the move's way
    "time",  # minutes since the open, over a full session
    "gap",  # the gap at the open, + the move's way
    "since_open",  # the move since the open, + the move's way
    "place",  # where the close stands in the day's range, 1 at the move's end
    "day_vol",  # the day's typical move so far
    "vix",  # log of the VIX's last close
    "beta",  # the stock's beta to QQQ, known before the open
    "news_before",  # log(1 + headlines since the previous close)
    "habit",  # its usual move apart from QQQ over the next 30 minutes, + the move's way
)
KINDS = tuple(
    f"{'at the open' if o else 'later'}, {'QQQ moved' if q else 'QQQ quiet'}, "
    f"{'a headline' if h else 'no headline'}"
    for o in (0, 1)
    for q in (0, 1)
    for h in (0, 1)
)
FIELDS = ("state", "own", "qqq", "beta", "kind", "day", "minute", "ticker", "chart")


# --- the actions ------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """What the agent may do at an extreme minute: let it be (side 0), or take a side, +1
    with the move or -1 against it, at a size, for a hold, with a share of its market part
    hedged."""

    side: float
    size: float
    hold: int
    hedge: float

    def __post_init__(self) -> None:
        if self.side == 0:
            if (self.size, self.hold, self.hedge) != (0, 0, 0):
                raise ValueError("letting the move be has no size, hold or hedge")
            return
        if self.side not in (1, -1):
            raise ValueError("a side is +1, with the move, or -1, against it")
        if self.size not in SIZES or self.hold not in HOLDS or self.hedge not in HEDGES:
            raise ValueError(
                "a size, hold and hedge must come from SIZES, HOLDS, HEDGES"
            )


def actions(cap: float = CAP) -> tuple[Action, ...]:
    """Every action, letting the move be first, then each side, size within the cap, hold
    and hedge."""
    out = [Action(0.0, 0.0, 0, 0.0)]
    for side in (1.0, -1.0):
        for size in SIZES:
            if size > cap:
                continue
            for hold in HOLDS:
                for hedge in HEDGES:
                    out.append(Action(side, size, hold, hedge))
    return tuple(out)


ACTIONS = actions()


def index_of(action: Action, table: tuple[Action, ...] = ACTIONS) -> int:
    return table.index(action)


# --- the events -------------------------------------------------------------------------


def kind_of(minute: np.ndarray, qqq: np.ndarray, headlines: np.ndarray) -> np.ndarray:
    """Each event's kind, an index into KINDS: at the open or later, QQQ moved or quiet, a
    headline or none."""
    at_open = (minute >= PARTS["open"][0]) & (minute <= PARTS["open"][1])
    moved = np.abs(qqq) >= MARKET_SHOCK
    return (4 * at_open + 2 * moved + (headlines > 0)).astype(np.uint8)


def session(day: Day, market: Day) -> dict[str, np.ndarray]:
    """The fresh extreme minutes of one session: what the agent sees at each, every hold's
    outcome, its kind and the chart around it."""
    rows = channels(day, market)
    at = fresh(
        rows[:, CHANNELS.index("apart")],
        rows[:, CHANNELS.index("seen")],
        EXTREME[0],
        COOLDOWN,
    )
    n = len(at)
    if not n:
        return empty()
    values = scalars(day)
    sign = np.sign(rows[at, CHANNELS.index("apart")])
    headlines = np.convolve(day.news, np.ones(NEWS_MINUTES))[: len(day.news)][at]

    def channel(name):
        return rows[at, CHANNELS.index(name)]

    def scalar(name):
        return values[at, SCALARS.index(name)]

    place = scalar("place")
    state = np.stack(
        [
            np.abs(channel("apart")),
            sign,
            channel("range"),
            channel("volume"),
            sign * channel("market"),
            np.log1p(headlines),
            sign * channel("vwap"),
            scalar("time"),
            sign * scalar("gap"),
            sign * scalar("since_open"),
            np.where(sign > 0, place, 1 - place),
            scalar("day_vol"),
            scalar("vix"),
            scalar("beta"),
            scalar("news_before"),
            sign * scalar("habit"),
        ],
        axis=1,
    )
    of_stock = labels(day.open, day.high, day.low, day.close, HOLDS)
    of_market = labels(market.open, market.high, market.low, market.close, HOLDS)
    log_close = np.log(day.close)
    offsets = np.arange(-CHART, CHART + 1)
    where = at[:, None] + offsets[None, :]
    inside = (where >= 0) & (where < len(day.close))
    chart = np.where(
        inside,
        log_close[np.clip(where, 0, len(day.close) - 1)] - log_close[at][:, None],
        np.nan,
    )
    return {
        "state": state.astype(np.float32),
        "own": np.stack([sign * of_stock[h]["ret"][at] for h in HOLDS], axis=1),
        "qqq": np.stack([sign * of_market[h]["ret"][at] for h in HOLDS], axis=1),
        "beta": np.full(n, day.beta),
        "kind": kind_of(at, channel("market"), headlines),
        "minute": at.astype(np.int16),
        "chart": chart.astype(np.float32),
    }


def empty() -> dict[str, np.ndarray]:
    k = len(HOLDS)
    return {
        "state": np.zeros((0, len(STATE)), np.float32),
        "own": np.zeros((0, k)),
        "qqq": np.zeros((0, k)),
        "beta": np.zeros(0),
        "kind": np.zeros(0, np.uint8),
        "minute": np.zeros(0, np.int16),
        "chart": np.zeros((0, 2 * CHART + 1), np.float32),
    }


def build(days: pl.DataFrame, arrays: dict, years, source: str = "real") -> dict:
    """Every stock's events in `years`, on days when QQQ's session is usable too, oldest
    first; `source` "random" replaces every session's prices by the skill test's random
    walk, seeded the same way."""
    if max(years) > YEARS[-1]:
        raise ValueError("the gym reads 2016-2021 at most; 2022-2026 stay unread")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    tickers = {t: i for i, t in enumerate(sorted(days["ticker"].unique()))}
    usable = days.filter(pl.col("usable") & pl.col("year").is_in(list(years)))
    markets = {
        r["day"]: r
        for r in usable.filter(pl.col("ticker") == MARKET).iter_rows(named=True)
    }
    stocks = usable.filter(~pl.col("etf") & pl.col("day").is_in(list(markets)))
    parts = {k: [] for k in FIELDS}
    for (d,), group in stocks.sort("day", "ticker").group_by(
        "day", maintain_order=True
    ):
        market = day_from(markets[d], arrays)
        if source == "random":
            market = random_walk(market, np.random.default_rng([SEED, d.toordinal()]))
        for row in group.iter_rows(named=True):
            day = day_from(row, arrays)
            if source == "random":
                seed = [SEED, d.toordinal(), tickers[row["ticker"]] + 1]
                day = random_walk(day, np.random.default_rng(seed))
            if len(day.open) != len(market.open):
                continue
            found = session(day, market)
            n = len(found["minute"])
            found["day"] = np.full(n, d.toordinal(), dtype=np.int32)
            found["ticker"] = np.full(n, tickers[row["ticker"]], dtype=np.int16)
            for k in FIELDS:
                parts[k].append(found[k])
    return {k: np.concatenate(v) if v else empty_field(k) for k, v in parts.items()}


def empty_field(name: str) -> np.ndarray:
    blank = empty()
    if name in blank:
        return blank[name]
    return np.zeros(0, np.int32 if name == "day" else np.int16)


def subset(data: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in data.items()}


def year_of(data: dict) -> np.ndarray:
    return np.array([date.fromordinal(int(d)).year for d in data["day"]], dtype=int)


def save(folder: Path, data: dict, tickers: list[str]) -> None:
    """One file a year, and the list of tickers the events' indices point to."""
    folder.mkdir(parents=True, exist_ok=True)
    years = year_of(data)
    for year in sorted(set(years)):
        part = subset(data, years == year)
        np.savez_compressed(folder / f"{year}.npz", **part)
    (folder / "tickers.json").write_text(json.dumps(tickers))


def load_events(folder: Path, years) -> dict:
    """The saved events of `years`, oldest first."""
    if max(years) > YEARS[-1]:
        raise ValueError("the gym reads 2016-2021 at most; 2022-2026 stay unread")
    parts = []
    for year in years:
        with np.load(folder / f"{year}.npz") as stored:
            parts.append({k: stored[k] for k in stored.files})
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


# --- outcomes, choices and their worth ----------------------------------------------------


def returns(
    data: dict, table: tuple[Action, ...] = ACTIONS, cost: float = COST
) -> np.ndarray:
    """The account's return of every action at every event, [events, actions]: the side
    times the stock's move less the hedged share of QQQ's, at the size, less both legs'
    round trips. Letting the move be returns nothing."""
    out = np.zeros((len(data["beta"]), len(table)))
    beta = data["beta"]
    for j, a in enumerate(table):
        if a.side == 0:
            continue
        h = HOLDS.index(a.hold)
        move = data["own"][:, h] - a.hedge * beta * data["qqq"][:, h]
        legs = a.size * cost + a.hedge * np.abs(beta) * a.size * COST_QQQ
        out[:, j] = a.size * a.side * move - legs
    return out


def allowed(
    data: dict, table: tuple[Action, ...] = ACTIONS, cap: float = CAP, limit: float = M
) -> np.ndarray:
    """Which actions each event allows, [events, actions]: a size within the cap, and a
    market part left unhedged of at most `limit` times equity."""
    beta = np.abs(data["beta"])
    out = np.ones((len(beta), len(table)), dtype=bool)
    for j, a in enumerate(table):
        if a.side == 0:
            continue
        left = (1 - a.hedge) * beta * a.size
        out[:, j] = (a.size <= cap) & (left <= limit + 1e-12)
    return out


def utility(r: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """The account's utility of returns r at constant relative risk aversion gamma; -inf
    where r wipes the account out."""
    r = np.asarray(r, dtype=float)
    alive = r > -1
    grown = np.where(alive, 1 + r, 1.0)
    if gamma == 1:
        value = np.log(grown)
    else:
        value = (grown ** (1 - gamma) - 1) / (1 - gamma)
    return np.where(alive, value, -np.inf)


def certainty(mean_utility: float, gamma: float = GAMMA) -> float:
    """The sure return worth a mean utility."""
    if not np.isfinite(mean_utility):
        return -1.0
    if gamma == 1:
        return float(np.expm1(mean_utility))
    base = 1 + (1 - gamma) * mean_utility
    return float(base ** (1 / (1 - gamma)) - 1) if base > 0 else float("inf")


def taken(values: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """Each event's value of the action chosen for it."""
    return values[np.arange(len(choice)), choice]


# --- the bars -----------------------------------------------------------------------------

AGAINST = Action(-1.0, 1.0, 15, 1.0)  # the fixed bars: 1x, 15 minutes, hedged in full


def bars(
    data: dict, permitted: np.ndarray, table: tuple[Action, ...] = ACTIONS
) -> dict:
    """Decision 0026's fixed bars, each an action an event; one an event cannot take is let
    be."""
    kind = data["kind"].astype(int)
    headline = (kind & 1) == 1
    at_open = (kind & 4) == 4
    j = index_of(AGAINST, table)
    out = {}
    for name, where in (
        ("nothing", np.zeros(len(kind), dtype=bool)),
        ("against every event", np.ones(len(kind), dtype=bool)),
        ("against, no headline", ~headline),
        ("against, no headline, at the open", ~headline & at_open),
    ):
        out[name] = np.where(where & permitted[:, j], j, 0)
    return out


def best_simple(
    values: np.ndarray, permitted: np.ndarray, kind: np.ndarray, least: int = MIN_KIND
) -> np.ndarray:
    """For each kind of event, the action of the highest mean utility over these events,
    an action an event cannot take counting as letting the move be: one action a kind. A
    kind with fewer than `least` events lets the move be, since the best of 91 actions on a
    few events is chance."""
    out = np.zeros(len(KINDS), dtype=int)
    scored = np.where(permitted, values, 0.0)
    for k in range(len(KINDS)):
        mine = kind == k
        if mine.sum() >= least:
            out[k] = int(np.argmax(scored[mine].mean(axis=0)))
    return out


def follow(policy: np.ndarray, kind: np.ndarray, permitted: np.ndarray) -> np.ndarray:
    """A policy of one action a kind, as an action an event; one it cannot take is let be."""
    chosen = policy[kind.astype(int)]
    return np.where(taken(permitted, chosen), chosen, 0)


# --- the shuffled bar -----------------------------------------------------------------------


def shuffled(data: dict, rng: np.random.Generator, tries: int = 100) -> dict:
    """The events with their outcomes dealt to another stock's event in the same part of
    the day; everything the agent sees stays. An agent trained on it learns only what is
    common to all events."""
    out = dict(data)
    source = np.arange(len(data["beta"]))
    minute = data["minute"].astype(int)
    for first, last in PARTS.values():
        mine = np.flatnonzero((minute >= first) & (minute <= last))
        if len(mine) < 2:
            continue
        order = rng.permutation(mine)
        for _ in range(tries):
            same = data["ticker"][mine] == data["ticker"][order]
            if not same.any():
                break
            for i in np.flatnonzero(same):
                j = rng.integers(len(mine))
                order[i], order[j] = order[j], order[i]
        if (data["ticker"][mine] == data["ticker"][order]).any():
            raise ValueError("cannot deal outcomes away from their own stock")
        source[mine] = order
    for k in ("own", "qqq"):
        out[k] = data[k][source]
    return out


# --- the run --------------------------------------------------------------------------------


def summary(data: dict, cost: float = COST) -> dict:
    """The fixed bars and the best simple policy on these events: each one's certainty
    equivalent in bps an event and its count of trades."""
    values = utility(returns(data, cost=cost))
    permitted = allowed(data)
    found = {}
    for name, choice in bars(data, permitted).items():
        found[name] = (certainty(taken(values, choice).mean()), int((choice > 0).sum()))
    policy = best_simple(values, permitted, data["kind"])
    choice = follow(policy, data["kind"], permitted)
    found["best simple policy, in sample"] = (
        certainty(taken(values, choice).mean()),
        int((choice > 0).sum()),
    )
    return {"bars": found, "policy": policy}


def describe(action: Action) -> str:
    if action.side == 0:
        return "let it be"
    side = "with" if action.side > 0 else "against"
    return f"{side}, {action.size:g}x, {action.hold} min, hedge {action.hedge:g}"


def report(data: dict, seconds: float) -> str:
    years = year_of(data)
    lines = [
        "KUTBIT's event gym (decision 0026), on the years an agent learns on",
        (
            f"Stored minutes of {min(years)}-{max(years)}; 2019-2026 not read. Run "
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC in {seconds:.0f} s."
        ),
        f"{len(years):,} events: "
        + ", ".join(f"{y} {int((years == y).sum()):,}" for y in sorted(set(years)))
        + f". {len(ACTIONS)} actions; position cap {CAP:g}x, M {M:g}x, gamma {GAMMA:g}.",
        "Events by kind: "
        + "; ".join(
            f"{name} {int((data['kind'] == k).sum()):,}" for k, name in enumerate(KINDS)
        ),
    ]
    for cost in COSTS:
        found = summary(data, cost)
        lines += [
            "",
            (
                f"At {cost * 1e4:g} bps a round trip: certainty equivalent in bps an "
                "event (trades)"
            ),
        ]
        for name, (ce, trades) in found["bars"].items():
            lines.append(f"  {name:<36}{ce * 1e4:>+8.2f}  ({trades:,})")
        lines.append(
            "  the best simple policy's action by kind, chosen on these events:"
        )
        for k, name in enumerate(KINDS):
            count = int((data["kind"] == k).sum())
            lines.append(
                f"    {name:<40}{count:>7,}  {describe(ACTIONS[found['policy'][k]])}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("build", "learning"):
        raise SystemExit("usage: gym.py build|learning [DATA]")
    data_dir = Path(argv[1]) if len(argv) > 1 else DATA
    started = time.time()
    if argv[0] == "learning":
        days, arrays = load(data_dir, LEARN)
        data = build(days, arrays, LEARN)
        text = report(data, time.time() - started)
        RESULTS.write_text(text)
        print(text)
        return
    days, arrays = load(data_dir, YEARS)
    tickers = sorted(days["ticker"].unique())
    for source in SOURCES:
        data = build(days, arrays, YEARS, source)
        save(data_dir / "kutbit" / "agent" / source, data, tickers)
        years = year_of(data)
        counts = ", ".join(f"{y} {int((years == y).sum()):,}" for y in YEARS)
        print(
            f"{source}: {len(years):,} events ({counts}), {time.time() - started:.0f} s"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
