"""Phase 3: the evening report, tried on a session played from Alpaca's history.

Before the first paper session, what `qfin live-report` will say about one. The session of
DAY is played through the live engine's core as the runner drives it: the strategy warmed
on the 20 sessions before, as each morning; IEX's minutes from Alpaca's history as the
stream, each seen 0.2 s after its minute ends; each order accepted at once and filled in
full at the next minute's open. The journal and the warm-up history are written as the
runner writes them, under ROOT/<day>/, where `qfin live-report --root ROOT` reads them.

The prices stay in the git-ignored data/.

Usage: report_demo.py DAY ROOT GUARD_QQQ_YAML   (with Alpaca's keys set)
"""

import sys
from datetime import date, timedelta
from pathlib import Path

from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.intraday import regular_session
from qfin_alphaguard.events import Order
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.live.engine import Cancel, SessionStart, Submit
from qfin_alphaguard.live.journal import Journal
from qfin_alphaguard.live.orders import Accepted, Cancelled, FillReport
from qfin_alphaguard.live.report import bars_of
from qfin_alphaguard.live.runner import TICKER, DayFiles, make_engine_for, past_sessions
from qfin_alphaguard.sessions import sessions

DAY = date.fromisoformat(sys.argv[1])
ROOT = Path(sys.argv[2])
LIMITS = load_guard_config(sys.argv[3])
SEEN = timedelta(milliseconds=200)

files = DayFiles(ROOT, DAY)
if files.journal.exists():
    sys.exit(f"{files.journal} exists: remove it to play the session again")
client = AlpacaClient(*api_keys())
past = past_sessions(client, DAY, files.warm_up)
session = sessions(DAY, DAY)[0]
bars = bars_of(
    regular_session(client.minute_bars(TICKER, DAY, DAY, feed="iex")), TICKER
)
start = SessionStart(DAY, session.open - timedelta(minutes=15), 30_000.0, 30_000.0)

journal = Journal(files.journal)
journal.write(start, at=start.time)
core = make_engine_for(LIMITS, past)(start)
waiting: list[Order] = []


def event(e, at):
    journal.write(e, at=at)
    take(core.on_order_event(e, at), at)


def take(step, at):
    for record in step.records:
        journal.write(record, at=at)
    for action in step.actions:
        if isinstance(action, Submit):
            waiting.append(action.order)
            event(Accepted(action.order.client_order_id, at), at)
        elif isinstance(action, Cancel):
            event(Cancelled(action.client_order_id, at), at)


for bar in bars:
    filled_at = bar.end + SEEN / 2
    due, waiting[:] = list(waiting), []
    for order in due:
        cid = order.client_order_id
        if core.orders[cid].is_open:
            event(
                FillReport(cid, filled_at, order.intent.quantity, bar.open), filled_at
            )
    at = bar.end + SEEN
    journal.write(bar, at=at)
    take(core.on_bars([bar], at), at)
journal.close()
print(
    f"{DAY}: {len(bars)} IEX minutes played; {len(core.orders)} orders; "
    f"journal {files.journal} ({len(journal.entries)} records)"
)
