"""The evening report: what the live engine did in a session, and whether it is the backtest's.

Phase 3's gate (M3) asks of each paper session:
1. Does the session replay from its journal to the same decisions (engine.replay_day)?
2. Does a backtest of the same session make the same decisions? On IEX's minutes as Alpaca
   keeps them, the feed the engine traded on, and on the consolidated tape (SIP). The gate
   wants at least 95% of decisions the same.
3. Did the fills cost what the backtest assumes? Each fill is measured against the last
   price the strategy saw when it decided: positive basis points cost the account.

day_report reads it all from the journal and the day's minutes; render_html makes it a page,
with the session's chart. The page holds licensed prices: it stays in the git-ignored data/.
"""

import html
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from qfin_alphaguard.backtest.broker import CostModel, NextBarBroker
from qfin_alphaguard.backtest.engine import SessionClose, run_backtest
from qfin_alphaguard.events import Bar, DecisionNote, Fill, Order, Side
from qfin_alphaguard.guard.config import GuardConfig
from qfin_alphaguard.guard.rules import Guard
from qfin_alphaguard.live.engine import EngineError, SessionStart, Stop, replay_day
from qfin_alphaguard.live.journal import Entry, rebuild
from qfin_alphaguard.live.orders import InvalidTransition
from qfin_alphaguard.live.runner import make_engine_for, warmed_strategy
from qfin_alphaguard.momentum import noise_band
from qfin_alphaguard.portfolio import Portfolio
from qfin_alphaguard.sessions import Session

NEW_YORK = ZoneInfo("America/New_York")
GATE = 0.95  # Phase 3's gate: the share of decisions a backtest must make the same
HALF_SPREAD_BPS = 0.3  # QQQ's measured half spread, as the day book's backtests use


@dataclass(frozen=True)
class OrderRow:
    client_order_id: str
    sent_at: datetime
    side: str
    quantity: float
    limit_price: float
    reason: str
    status: str
    filled: float
    average_price: float | None
    reference_price: float | None  # the last price the strategy saw when it decided
    slippage_bps: float | None  # the fills against it: positive costs the account


@dataclass(frozen=True)
class Comparison:
    """A backtest of the session on one feed's minutes, against what the engine did."""

    feed: str
    bars: int
    matched: int
    total: int
    differences: tuple[tuple[datetime, str, str], ...]  # minute, live, backtest
    result_usd: float
    orders: int

    @property
    def share(self) -> float:
        return self.matched / self.total if self.total else 1.0


@dataclass(frozen=True)
class BarCheck:
    """The bars the stream sent against the ones Alpaca keeps for the day."""

    live: int
    kept: int
    same: int
    differing: int
    live_only: int
    kept_only: int


@dataclass(frozen=True)
class DayReport:
    day: date
    start: SessionStart
    replay: str  # "the same" or what differed
    leftover: int  # records a crash cut short
    notes: tuple[DecisionNote, ...]
    orders: tuple[OrderRow, ...]
    stops: tuple[Stop, ...]
    start_value: float
    end_value: float
    bars: BarCheck
    comparisons: tuple[Comparison, ...]
    alerts: tuple[str, ...]

    @property
    def result_usd(self) -> float:
        return self.end_value - self.start_value

    @property
    def result_pct(self) -> float:
        return self.result_usd / self.start_value if self.start_value else 0.0

    def summary(self) -> str:
        """The report in a few lines, for a terminal or an e-mail."""
        leftover = (
            f" ({self.leftover} records a crash cut short)" if self.leftover else ""
        )
        lines = [
            (
                f"Session {self.day}: {self.result_usd:+,.2f} USD "
                f"({self.result_pct:+.2%}) on {self.start_value:,.2f} USD"
            ),
            f"Replayed from its journal: {self.replay}{leftover}",
        ]
        changes, held = [], "flat"
        for note in self.notes:
            if note.choice != held:
                changes.append(f"{note.time.astimezone(NEW_YORK):%H:%M} {note.choice}")
                held = note.choice
        lines.append(
            f"Decisions: {len(self.notes)}; the position, New York time: "
            + (", ".join(changes) if changes else "flat all day")
        )
        filled = [o for o in self.orders if o.filled > 0]
        slips = [o.slippage_bps for o in filled if o.slippage_bps is not None]
        slip = (
            f"; slippage median {statistics.median(slips):+.2f} bps, "
            f"mean {statistics.fmean(slips):+.2f} bps against the price decided on"
            if slips
            else ""
        )
        lines.append(f"Orders: {len(self.orders)}, {len(filled)} filled{slip}")
        for c in self.comparisons:
            verdict = "meets" if c.share >= GATE else "MISSES"
            lines.append(
                f"Backtest on {c.feed}: {c.matched} of {c.total} decisions the same "
                f"({c.share:.0%}, {verdict} the {GATE:.0%} gate); "
                f"its result {c.result_usd:+,.2f} USD, {c.orders} orders"
            )
        b = self.bars
        lines.append(
            f"Bars: {b.live} from the stream, {b.kept} kept by Alpaca; {b.same} the same, "
            f"{b.differing} different, {b.live_only} only live, {b.kept_only} only kept"
        )
        for stop in self.stops:
            when = stop.time.astimezone(NEW_YORK)
            lines.append(f"STOPPED at {when:%H:%M:%S} New York time: {stop.reason}")
        for alert in self.alerts:
            lines.append(f"Alert: {alert}")
        return "\n".join(lines)


def bars_of(frame, ticker: str) -> list[Bar]:
    """Minute bars from a frame as AlpacaClient.minute_bars returns it."""
    return [
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
        for r in frame.sort("start").iter_rows(named=True)
    ]


def _minute(time: datetime) -> datetime:
    return time.replace(second=0, microsecond=0)


def _decisions(notes: Sequence[DecisionNote]) -> dict[datetime, str]:
    """Each decision's choice, by the minute its bar ended."""
    return {_minute(n.time): n.choice for n in notes}


def _value(
    start: SessionStart, fills: Sequence[Fill], prices: Mapping[str, float]
) -> float:
    books = Portfolio(start.cash, start.positions)
    for fill in fills:
        books.apply(fill)
    return books.value({**start.prices, **prices})


def backtest_session(
    start: SessionStart,
    bars: Sequence[Bar],
    past: Sequence[tuple[Session, Sequence[Bar]]],
    limits: GuardConfig,
) -> tuple[list[DecisionNote], int, float]:
    """The backtest's decisions, orders and result on `bars`, from the morning's account.

    It starts from the account's cash: a position carried from an earlier day is not in it.
    """
    strategy = warmed_strategy(limits, past, start.cash)
    broker = NextBarBroker(CostModel(half_spread_bps=HALF_SPREAD_BPS))
    result = run_backtest(bars, strategy, broker, Guard(limits), start.cash)
    notes = [r for r in result.log if isinstance(r, DecisionNote)]
    orders = sum(isinstance(r, Order) for r in result.log)
    closes = [r for r in result.log if isinstance(r, SessionClose)]
    value = closes[-1].value if closes else start.cash
    return notes, orders, value - start.cash


def day_report(
    entries: Sequence[Entry],
    past: Sequence[tuple[Session, Sequence[Bar]]],
    limits: GuardConfig,
    feeds: Mapping[str, Sequence[Bar]],
    alerts: Sequence[str] = (),
) -> DayReport:
    """The report of the session in `entries`, a journal read back.

    `feeds` holds the session's minutes by feed, such as Alpaca's history of IEX and SIP;
    the first is the one the engine traded on, and its bars are checked against the
    stream's.
    """
    if not entries or not isinstance(entries[0].record, SessionStart):
        raise ValueError("a journal starts with its session's start")
    start: SessionStart = entries[0].record
    records = [e.record for e in entries]
    try:
        _, leftover = replay_day(entries, make_engine_for(limits, past))
        replay = "the same"
    except (EngineError, InvalidTransition) as error:
        leftover, replay = [], f"DIFFERS: {error}"
    notes = [r for r in records if isinstance(r, DecisionNote)]
    fills = [r for r in records if isinstance(r, Fill)]
    stops = [r for r in records if isinstance(r, Stop)]
    live_bars = [r for r in records if isinstance(r, Bar)]

    # Each order, with the last price the strategy saw before it was sent.
    orders = rebuild(records).orders
    rows = []
    last_price: float | None = None
    for record in records:
        if isinstance(record, Bar):
            last_price = record.close
        elif isinstance(record, Order):
            order = orders[record.client_order_id]
            reference = last_price or start.prices.get(record.intent.ticker)
            slip = None
            if order.average_price is not None and reference:
                sign = 1 if record.intent.side is Side.BUY else -1
                slip = sign * (order.average_price / reference - 1) * 10_000
            rows.append(
                OrderRow(
                    record.client_order_id,
                    record.sent_at,
                    record.intent.side.value,
                    record.intent.quantity,
                    record.intent.limit_price,
                    record.intent.reason,
                    order.status.value,
                    order.filled_quantity,
                    order.average_price,
                    reference,
                    slip,
                )
            )

    prices = {b.ticker: b.close for b in live_bars}
    start_value = _value(start, [], {})
    end_value = _value(start, fills, prices)

    live_decisions = _decisions(notes)
    comparisons = []
    for feed, bars in feeds.items():
        theirs, n_orders, result = backtest_session(start, bars, past, limits)
        their_decisions = _decisions(theirs)
        minutes = sorted(set(live_decisions) | set(their_decisions))
        differences = tuple(
            (m, live_decisions.get(m, "-"), their_decisions.get(m, "-"))
            for m in minutes
            if live_decisions.get(m) != their_decisions.get(m)
        )
        comparisons.append(
            Comparison(
                feed,
                len(bars),
                len(minutes) - len(differences),
                len(minutes),
                differences,
                result,
                n_orders,
            )
        )

    kept = {b.start: b for b in next(iter(feeds.values()), [])}
    live = {b.start: b for b in live_bars}
    both = set(kept) & set(live)
    same = sum(kept[s] == live[s] for s in both)
    check = BarCheck(
        len(live),
        len(kept),
        same,
        len(both) - same,
        len(live) - len(both),
        len(kept) - len(both),
    )
    return DayReport(
        start.day,
        start,
        replay,
        len(leftover),
        tuple(notes),
        tuple(rows),
        tuple(stops),
        start_value,
        end_value,
        check,
        tuple(comparisons),
        tuple(alerts),
    )


# The page.


def chart_series(
    report: DayReport,
    bars: Sequence[Bar],
    past: Sequence[tuple[Session, Sequence[Bar]]],
    limits: GuardConfig,
    session: Session,
) -> list[dict]:
    """Each minute of the session: price, VWAP and the noise band the strategy used."""
    strategy = warmed_strategy(limits, past, report.start.cash)
    moves, last_close = (
        list(strategy.moves),
        strategy.closes[-1] if strategy.closes else None,
    )
    points = []
    flow = volume = 0.0
    day_open = bars[0].open if bars else None
    for bar in bars:
        minute = int((bar.start - session.open).total_seconds() // 60)
        flow += (bar.high + bar.low + bar.close) / 3 * bar.volume
        volume += bar.volume
        lower = upper = None
        if moves and last_close is not None and day_open is not None:
            low, high = noise_band(moves, minute, day_open, last_close)
            lower = None if math.isnan(low) else round(low, 4)
            upper = None if math.isnan(high) else round(high, 4)
        points.append(
            {
                "t": bar.end.isoformat(),
                "price": bar.close,
                "vwap": round(flow / volume, 4) if volume else None,
                "lower": lower,
                "upper": upper,
            }
        )
    return points


def render_html(report: DayReport, points: Sequence[dict], when_new_york) -> str:
    """The report as a page: the verdicts, the chart, and the tables behind it."""
    e = html.escape
    marks = [
        {
            "t": o.sent_at.isoformat(),
            "price": o.average_price or o.limit_price,
            "side": o.side,
            "label": f"{o.side} {o.filled:g} of {o.quantity:g} at "
            + (f"{o.average_price:.2f}" if o.average_price else "nothing filled"),
        }
        for o in report.orders
    ]
    verdicts = "".join(f"<li>{e(line)}</li>" for line in report.summary().splitlines())
    notes = "".join(
        f"<tr><td>{when_new_york(n.time):%H:%M:%S}</td><td>{e(n.choice)}</td>"
        f"<td>{e(n.reason)}</td><td class=num>{n.seen.get('price', float('nan')):.2f}</td>"
        f"<td class=num>{n.seen.get('lower', float('nan')):.2f}</td>"
        f"<td class=num>{n.seen.get('upper', float('nan')):.2f}</td>"
        f"<td class=num>{n.seen.get('vwap', float('nan')):.2f}</td>"
        f"<td class=num>{n.seen.get('shares', float('nan')):g}</td></tr>"
        for n in report.notes
    )
    orders = "".join(
        f"<tr><td>{when_new_york(o.sent_at):%H:%M:%S}</td><td>{e(o.side)}</td>"
        f"<td class=num>{o.quantity:g}</td><td class=num>{o.limit_price:.2f}</td>"
        f"<td>{e(o.status)}</td><td class=num>{o.filled:g}</td>"
        f"<td class=num>{'' if o.average_price is None else f'{o.average_price:.4f}'}</td>"
        f"<td class=num>{'' if o.slippage_bps is None else f'{o.slippage_bps:+.2f}'}</td>"
        f"<td>{e(o.reason)}</td></tr>"
        for o in report.orders
    )
    differences = (
        "".join(
            f"<tr><td>{e(c.feed)}</td><td>{when_new_york(m):%H:%M}</td>"
            f"<td>{e(live)}</td><td>{e(theirs)}</td></tr>"
            for c in report.comparisons
            for m, live, theirs in c.differences
        )
        or "<tr><td colspan=4>none</td></tr>"
    )
    data = json.dumps({"points": list(points), "marks": marks})
    return PAGE.format(
        day=report.day.isoformat(),
        verdicts=verdicts,
        notes=notes or "<tr><td colspan=8>none</td></tr>",
        orders=orders or "<tr><td colspan=9>none</td></tr>",
        differences=differences,
        data=data.replace("</", "<\\/"),
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QQQ session {day}</title>
<style>
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ef; --grid: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77766f;
  --series-1: #2a78d6; --series-2: #eb6834; --band: #8d8c86;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322; --grid: #33332f;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-1: #3987e5; --series-2: #d95926; --band: #8d8c86;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #232322; --grid: #33332f;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --series-1: #3987e5; --series-2: #d95926; --band: #8d8c86;
}}
body {{ margin: 0; background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 8px; }}
h2 {{ font-size: 16px; margin: 32px 0 8px; }}
ul.verdicts {{ padding-left: 18px; color: var(--text-secondary); }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--text-secondary);
  font-size: 13px; margin: 8px 0; }}
.legend span::before {{ content: ""; display: inline-block; width: 14px; height: 2px;
  margin-right: 6px; vertical-align: middle; background: var(--c); }}
.legend .dashed::before {{ background: none; border-top: 2px dashed var(--c); }}
.chart {{ position: relative; }}
svg {{ width: 100%; height: auto; display: block; }}
.tip {{ position: absolute; pointer-events: none; background: var(--surface-2);
  border: 1px solid var(--grid); border-radius: 6px; padding: 6px 8px; font-size: 12px;
  color: var(--text-primary); white-space: nowrap; display: none; }}
.tables {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--text-secondary); font-weight: 600; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<main>
<h1>QQQ session {day}</h1>
<ul class="verdicts">{verdicts}</ul>
<div class="legend">
  <span style="--c: var(--series-1)">QQQ, the stream's bars</span>
  <span style="--c: var(--series-2)">VWAP</span>
  <span class="dashed" style="--c: var(--band)">noise band</span>
  <span>&#9650; buy &#9660; sell, at the fill</span>
</div>
<div class="chart"><svg id="chart" viewBox="0 0 1000 420" role="img"
  aria-label="QQQ's price through the session, with VWAP, the noise band and the orders"></svg>
<div class="tip" id="tip"></div></div>
<h2>Decisions (New York time)</h2>
<div class="tables"><table><thead><tr><th>Time</th><th>Choice</th><th>Why</th>
<th>Price</th><th>Band low</th><th>Band high</th><th>VWAP</th><th>Shares</th></tr></thead>
<tbody>{notes}</tbody></table></div>
<h2>Orders</h2>
<div class="tables"><table><thead><tr><th>Sent</th><th>Side</th><th>Shares</th>
<th>Limit</th><th>Status</th><th>Filled</th><th>Average price</th><th>Slippage, bps</th>
<th>Reason</th></tr></thead><tbody>{orders}</tbody></table></div>
<h2>Where a backtest decided otherwise</h2>
<div class="tables"><table><thead><tr><th>Backtest on</th><th>Minute</th><th>Live</th>
<th>Backtest</th></tr></thead><tbody>{differences}</tbody></table></div>
</main>
<script type="application/json" id="data">{data}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById("data").textContent);
  const pts = data.points.map(p => ({{...p, x: Date.parse(p.t)}}));
  const svg = document.getElementById("chart"), tip = document.getElementById("tip");
  if (!pts.length) {{ svg.innerHTML = '<text x="20" y="40">no bars</text>'; return; }}
  const W = 1000, H = 420, L = 60, R = 16, T = 12, B = 32;
  const values = pts.flatMap(p => [p.price, p.vwap, p.lower, p.upper]).filter(v => v != null);
  const lo = Math.min(...values), hi = Math.max(...values), pad = (hi - lo) * 0.05 || 1;
  const x0 = pts[0].x, x1 = pts[pts.length - 1].x;
  const X = x => L + (x - x0) / (x1 - x0 || 1) * (W - L - R);
  const Y = v => T + (hi + pad - v) / (hi - lo + 2 * pad) * (H - T - B);
  const ns = "http://www.w3.org/2000/svg";
  const add = (tag, attrs, parent = svg) => {{
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    parent.appendChild(el); return el; }};
  const fmtTime = x => new Date(x).toLocaleTimeString("en-US",
    {{timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false}});
  for (let i = 0; i <= 4; i++) {{
    const v = lo - pad + (hi - lo + 2 * pad) * i / 4;
    add("line", {{x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--grid)"}});
    add("text", {{x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11,
      fill: "var(--text-muted)"}}).textContent = v.toFixed(2);
  }}
  for (let i = 0; i <= 6; i++) {{
    const x = x0 + (x1 - x0) * i / 6;
    add("text", {{x: X(x), y: H - 10, "text-anchor": "middle", "font-size": 11,
      fill: "var(--text-muted)"}}).textContent = fmtTime(x);
  }}
  const path = key => pts.filter(p => p[key] != null)
    .map((p, i) => (i ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p[key]).toFixed(1)).join(" ");
  for (const k of ["lower", "upper"]) add("path", {{d: path(k), fill: "none",
    stroke: "var(--band)", "stroke-width": 1.5, "stroke-dasharray": "5 4"}});
  add("path", {{d: path("vwap"), fill: "none", stroke: "var(--series-2)", "stroke-width": 2}});
  add("path", {{d: path("price"), fill: "none", stroke: "var(--series-1)", "stroke-width": 2}});
  for (const m of data.marks) {{
    const x = X(Date.parse(m.t)), y = Y(m.price), up = m.side === "buy";
    const d = up ? `M${{x}} ${{y - 7}} L${{x + 6}} ${{y + 5}} L${{x - 6}} ${{y + 5}} Z`
                 : `M${{x}} ${{y + 7}} L${{x + 6}} ${{y - 5}} L${{x - 6}} ${{y - 5}} Z`;
    add("path", {{d, fill: "var(--text-primary)", stroke: "var(--surface-1)", "stroke-width": 2}})
      .appendChild(document.createElementNS(ns, "title")).textContent = m.label;
  }}
  const cross = add("line", {{y1: T, y2: H - B, stroke: "var(--text-muted)", visibility: "hidden"}});
  svg.addEventListener("mousemove", ev => {{
    const box = svg.getBoundingClientRect(), sx = (ev.clientX - box.left) / box.width * W;
    const target = x0 + (sx - L) / (W - L - R) * (x1 - x0);
    let best = pts[0];
    for (const p of pts) if (Math.abs(p.x - target) < Math.abs(best.x - target)) best = p;
    cross.setAttribute("x1", X(best.x)); cross.setAttribute("x2", X(best.x));
    cross.setAttribute("visibility", "visible");
    const f = v => v == null ? "-" : v.toFixed(2);
    tip.innerHTML = `${{fmtTime(best.x)}}<br>QQQ ${{f(best.price)}}<br>VWAP ${{f(best.vwap)}}` +
      `<br>band ${{f(best.lower)}} to ${{f(best.upper)}}`;
    tip.style.display = "block";
    const left = (X(best.x) / W) * box.width;
    tip.style.left = Math.min(left + 12, box.width - 150) + "px"; tip.style.top = "8px";
  }});
  svg.addEventListener("mouseleave", () => {{ tip.style.display = "none";
    cross.setAttribute("visibility", "hidden"); }});
}})();
</script>
</body>
</html>
"""
