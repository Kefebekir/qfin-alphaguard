"""The daily plan: the portfolio to hold, decided the night before.

For trading day D, using only the daily bars before D:

1. the year's trading universe (decision 0005)
2. each stock's daily log returns over the last LOOKBACK_DAYS sessions; a
   stock without a full window waits until it has one
3. their covariance, shrunk with Ledoit-Wolf, and the long-only portfolio
   with the least variance, no stock above Guard's max_weight
4. at most as many stocks as the capital can hold: each position must be big
   enough to build in max_children orders of min_trade_usd. The optimiser's
   largest weights are kept and the portfolio solved again on them.

The plan also carries each stock's daily volatility, for the band that
decides at the open whether a gap is worth trading (Phase 1, step 7b).
"""

import json
import math
from collections.abc import Iterable
from datetime import date

import numpy as np
import polars as pl

from qfin_alphaguard.data.universe import TRADING_UNIVERSE_SIZE, trading_universe
from qfin_alphaguard.estimation.covariance import TRADING_DAYS, ledoit_wolf_covariance
from qfin_alphaguard.events import DailyPlan, ExecutionSettings
from qfin_alphaguard.optimize.min_variance import min_variance_weights

LOOKBACK_DAYS = 252  # one year of daily returns
MODEL_VERSION = "min-variance-lw-v1"


def positions_for(capital: float, settings: ExecutionSettings) -> int:
    """How many stocks `capital` can hold, each big enough for max_children orders."""
    smallest_position = settings.max_children * settings.min_trade_usd
    return max(1, math.floor(capital / smallest_position))


def build_plan(
    prices: pl.DataFrame,
    day: date,
    capital: float,
    max_weight: float,
    settings: ExecutionSettings | None = None,
    universe: tuple[str, ...] | None = None,
) -> DailyPlan:
    """The plan for trading day `day`, from the daily bars before it.

    `universe` is the year's trading universe, if the caller already has it.
    """
    settings = settings or ExecutionSettings()
    if universe is None:
        universe = trading_universe(prices, day.year, TRADING_UNIVERSE_SIZE)
    window = (
        prices.filter((pl.col("date") < day) & pl.col("ticker").is_in(universe))
        .sort("date")
        .with_columns(pl.col("close").log().diff().over("ticker").alias("return"))
        .pivot(on="ticker", index="date", values="return")
        .sort("date")
        .tail(LOOKBACK_DAYS)
    )
    full = [
        ticker
        for ticker in universe
        if ticker in window.columns and window[ticker].null_count() == 0
    ]
    if len(window) < LOOKBACK_DAYS or not full:
        raise ValueError(f"not a year of returns before {day} for the universe")
    cov, _ = ledoit_wolf_covariance(window.select(full).to_numpy())
    volatility = np.sqrt(np.diag(cov) / TRADING_DAYS)

    held = _min_variance(cov, max_weight)
    count = positions_for(capital, settings)
    if np.count_nonzero(held > 1e-6) > count:
        keep = np.argsort(-held, kind="stable")[:count]
        held = np.zeros(len(full))
        held[keep] = _min_variance(cov[np.ix_(keep, keep)], max_weight)
    weights = {
        ticker: float(w) for ticker, w in zip(full, held, strict=True) if w > 1e-6
    }
    return DailyPlan(
        trading_date=day,
        model_version=MODEL_VERSION,
        target_weights=weights,
        execution=settings,
        volatility=dict(zip(full, map(float, volatility), strict=True)),
    )


def _min_variance(cov: np.ndarray, max_weight: float) -> np.ndarray:
    """Least-variance weights; with too few stocks for max_weight, max_weight each.

    Below 1 / max_weight stocks (4 at 25%) the portfolio cannot be fully
    invested without breaking the limit, so each gets max_weight and the rest
    stays in cash.
    """
    if len(cov) * max_weight < 1:
        return np.full(len(cov), max_weight)
    return min_variance_weights(cov, max_weight)


def plan_to_json(plan: DailyPlan) -> str:
    execution = plan.execution
    return json.dumps(
        {
            "date": plan.trading_date.isoformat(),
            "model_version": plan.model_version,
            "target_weights": dict(plan.target_weights),
            "volatility": dict(plan.volatility),
            "execution": {
                name: getattr(execution, name)
                for name in execution.__dataclass_fields__
            },
        },
        indent=2,
        sort_keys=True,
    )


def plan_from_json(text: str) -> DailyPlan:
    data = json.loads(text)
    return DailyPlan(
        trading_date=date.fromisoformat(data["date"]),
        model_version=data["model_version"],
        target_weights=data["target_weights"],
        execution=ExecutionSettings(**data["execution"]),
        volatility=data["volatility"],
    )


def build_plans(
    prices: pl.DataFrame,
    days: Iterable[date],
    capital: float,
    max_weight: float,
    settings: ExecutionSettings | None = None,
) -> dict[date, DailyPlan]:
    """A plan for each of `days`, each from the bars before its own day.

    The trading universe is chosen once a year, so it is worked out once per year.
    """
    universes: dict[int, tuple[str, ...]] = {}
    plans = {}
    for day in days:
        if day.year not in universes:
            universes[day.year] = trading_universe(
                prices, day.year, TRADING_UNIVERSE_SIZE
            )
        plans[day] = build_plan(
            prices, day, capital, max_weight, settings, universes[day.year]
        )
    return plans
