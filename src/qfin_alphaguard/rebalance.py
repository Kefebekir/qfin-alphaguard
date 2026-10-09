"""Which stocks to trade today, and how many shares.

At the open the engine compares the day's plan with the shares it holds. A
stock is traded only when its gap is larger than its own band (decision
0006): band_k days of its daily volatility, times the larger of its target
and current weight. The strategy then splits the day's trades into orders and
times them (Phase 1, step 8).

trades_at_open checks its inputs and values the account. The two rules it
applies are Efe's (Phase 1, step 7b): _change for each stock and
_pay_for_buys for the cash.
"""

import math
from collections.abc import Mapping

from qfin_alphaguard.events import DailyPlan, ExecutionSettings


def trades_at_open(
    plan: DailyPlan,
    shares: Mapping[str, int],
    prices: Mapping[str, float],
    cash: float,
) -> dict[str, int]:
    """Shares to buy (positive) or sell (negative) today, by ticker; only trades.

    `shares` are the shares held now, `prices` the latest price of every stock
    in the plan or held, and `cash` the cash in the account. Equity, what the
    account is worth, is the cash plus the value of the shares.
    """
    held = {ticker: n for ticker, n in shares.items() if n > 0}
    tickers = sorted(set(plan.target_weights) | set(held))
    missing = [ticker for ticker in tickers if ticker not in prices]
    if missing:
        raise ValueError(f"no price for {', '.join(missing)}")
    if plan.target_weights and not plan.volatility:
        raise ValueError("the plan has no volatility for its stocks")
    equity = cash + sum(n * prices[ticker] for ticker, n in held.items())
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")

    trades = {}
    for ticker in tickers:
        change = _change(
            target=plan.target_weights.get(ticker, 0.0),
            held=held.get(ticker, 0),
            price=prices[ticker],
            volatility=plan.volatility.get(ticker, 0.0),
            equity=equity,
            settings=plan.execution,
        )
        if change != 0:
            trades[ticker] = change
    return _pay_for_buys(trades, prices, cash, plan.execution.min_trade_usd)


def _change(
    target: float,
    held: int,
    price: float,
    volatility: float,
    equity: float,
    settings: ExecutionSettings,
) -> int:
    """Shares to trade in one stock today: positive to buy, negative to sell.

    `target` is the plan's weight for the stock, 0 if the plan dropped it, and
    `held` the shares held now. The current weight is held * price / equity.

    1. If the target is 0, sell every share held, whatever the band or the
       minimum: a stock the plan no longer wants is not kept.
    2. If |target - current| <= settings.band_k * volatility *
       max(target, current), return 0: the gap is within the stock's own
       noise.
    3. Otherwise trade to the nearest whole number of shares at the target,
       round(target * equity / price), unless the trade is worth less than
       settings.min_trade_usd; then return 0.
    """
    if target == 0:
        return -held
    current = held * price / equity
    band = settings.band_k * volatility * max(target, current)
    if abs(target - current) <= band:
        return 0
    change = round(target * equity / price) - held
    if abs(change) * price < settings.min_trade_usd:
        return 0
    return change


def _pay_for_buys(
    trades: dict[str, int],
    prices: Mapping[str, float],
    cash: float,
    min_trade_usd: float,
) -> dict[str, int]:
    """The day's trades, with the buys cut to what the account can pay.

    Sales pay first: the money available is the cash plus the value of every
    sale. If the buys cost more than that, every buy is multiplied by the same
    factor, available / cost of the buys (never below 0), and rounded down to
    whole shares. A buy worth less than min_trade_usd after the cut waits for
    another day. Sales are never changed.
    """
    sales = sum(-n * prices[ticker] for ticker, n in trades.items() if n < 0)
    cost = sum(n * prices[ticker] for ticker, n in trades.items() if n > 0)
    available = cash + sales
    if cost <= available:
        return trades
    factor = max(available, 0.0) / cost
    paid = {}
    for ticker, n in trades.items():
        if n > 0:
            n = math.floor(n * factor)
            if n * prices[ticker] < min_trade_usd:
                continue
        paid[ticker] = n
    return paid
