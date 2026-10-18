"""The live engine's morning: the strategy relearns the sessions before today.

Decision 0021. The QQQ strategy trades on what the last sessions taught it: their moves
away from the open (the noise band) and their closes (the volatility). The backtest's
strategy learns them as it goes. The live engine starts afresh each morning, so it plays
the last sessions' minutes through the strategy first, as the backtest's engine calls it
(each session's start, its bars one by one, its end) and sends nothing. The minutes are
IEX's from Alpaca's history, the feed the strategy then trades on.

A session with no bar on the feed is skipped, as the backtest's engine never starts one.
"""

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from itertools import groupby
from types import MappingProxyType

from qfin_alphaguard.data.alpaca import AlpacaClient
from qfin_alphaguard.data.intraday import regular_session
from qfin_alphaguard.events import Bar
from qfin_alphaguard.sessions import Session, sessions
from qfin_alphaguard.strategy import Strategy, View

# Sessions played each morning: the strategy needs 15 closes (14 returns) and 14 sessions'
# moves; a few more cover sessions the feed has no bar for.
WARM_UP_SESSIONS = 20


def sessions_before(day: date, count: int = WARM_UP_SESSIONS) -> tuple[Session, ...]:
    """The `count` sessions before `day`, oldest first."""
    found = sessions(day - timedelta(days=2 * count + 10), day - timedelta(days=1))
    if len(found) < count:
        raise ValueError(f"fewer than {count} sessions known before {day}")
    return found[-count:]


def history(
    client: AlpacaClient, ticker: str, day: date, count: int = WARM_UP_SESSIONS
) -> list[tuple[Session, list[Bar]]]:
    """IEX's minute bars of `ticker` in the `count` sessions before `day`, by session."""
    past = sessions_before(day, count)
    minutes = regular_session(
        client.minute_bars(ticker, past[0].day, past[-1].day, feed="iex")
    )
    bars = [
        Bar(
            ticker,
            r["start"],
            60,
            r["open"],
            r["high"],
            r["low"],
            r["close"],
            r["volume"],
        )
        for r in minutes.sort("start").iter_rows(named=True)
    ]
    by_day = {s.day: s for s in past}
    return [
        (by_day[when], list(day_bars))
        for when, day_bars in groupby(bars, key=lambda b: b.start.date())
        if when in by_day
    ]


def warm_up(
    strategy: Strategy,
    past: Iterable[tuple[Session, Sequence[Bar]]],
    cash: float,
) -> None:
    """Play `past`, oldest first, through `strategy` as the backtest's engine would.

    Nothing is held and nothing is sent: what the strategy decides is dropped. A session
    with no bars is skipped.
    """
    last_prices: dict[str, float] = {}
    price_times: dict[str, datetime] = {}

    def view(session: Session, now: datetime) -> View:
        return View(
            now=now,
            session=session,
            cash=cash,
            positions=MappingProxyType({}),
            last_prices=MappingProxyType(last_prices),
            open_orders=(),
            price_times=MappingProxyType(price_times),
            last_close_value=cash,
        )

    for session, bars in past:
        if not bars:
            continue
        strategy.on_session_start(view(session, session.open))
        for bar in bars:
            if not (session.open <= bar.start and bar.end <= session.close):
                raise ValueError(f"{bar} lies outside its session {session.day}")
            last_prices[bar.ticker] = bar.close
            price_times[bar.ticker] = bar.end
            strategy.on_bars(
                MappingProxyType({bar.ticker: bar}), view(session, bar.end)
            )
        strategy.on_session_end(view(session, session.close))
