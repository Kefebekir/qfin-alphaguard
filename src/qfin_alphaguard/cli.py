"""Command line entry point.

    qfin ingest       load, validate and store daily prices (the nightly AWS job)
    qfin intraday     download 1-minute bars for each year's trading universe
    qfin etf-minutes  download ETFs' 1-minute bars from Alpaca, such as QQQ's
    qfin plan         build a day's plan: the portfolio to hold, from the bars before it
    qfin backtest     replay the nightly plan on the stored minutes, against SPY and
                      the same stocks at equal weight
    qfin live         run a session of the QQQ strategy on Alpaca's paper account
    qfin live-report  a paper session's report, against backtests of the same session

The nightly AWS job runs the command in infra/aws/task-definition.template.json;
a test checks that this CLI still accepts it.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from qfin_alphaguard.backtest.report import render, summary
from qfin_alphaguard.backtest.run import Backtest
from qfin_alphaguard.config import Config
from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.eodhd import EodhdClient, api_key
from qfin_alphaguard.data.ingest import (
    load_prices,
    repair_broken_bars,
    sp500_coverage,
)
from qfin_alphaguard.data.intraday import YearOfMinutes, backfill_minutes
from qfin_alphaguard.data.store import (
    prices_path,
    read_prices,
    upload_to_s3,
    write_prices,
)
from qfin_alphaguard.data.universe import sp500, trading_universes
from qfin_alphaguard.data.validate import validate_prices
from qfin_alphaguard.guard.config import load_guard_config
from qfin_alphaguard.plan import build_plan, plan_to_json
from qfin_alphaguard.sessions import session_days, sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qfin",
        description="QFin-AlphaGuard: nightly data, daily plan and backtests.",
    )
    # required=True: a bare `qfin --synthetic`, the old form, fails instead of
    # quietly doing something else.
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="load, validate and store daily prices")
    ingest.add_argument(
        "--synthetic",
        action="store_true",
        help="use generated data instead of downloading",
    )
    ingest.add_argument(
        "--s3-bucket",
        help="also upload the result to this S3 bucket",
    )
    ingest.add_argument(
        "--end-date",
        help="last date to download (YYYY-MM-DD); defaults to today for real data",
    )
    ingest.set_defaults(run=_ingest)

    intraday = commands.add_parser(
        "intraday",
        help="download 1-minute bars for each year's trading universe, and for "
        "January of the stocks that left it",
    )
    intraday.add_argument(
        "--from-year",
        type=int,
        default=2016,
        help="first year; 2016 is the first with a trading universe",
    )
    intraday.add_argument(
        "--to-year", type=int, help="last year; defaults to the current year"
    )
    intraday.add_argument(
        "--workers", type=int, default=4, help="requests to EODHD at a time"
    )
    intraday.set_defaults(run=_intraday)

    etf = commands.add_parser(
        "etf-minutes",
        help="download ETFs' 1-minute bars from Alpaca, checked against EODHD's "
        "daily closes",
    )
    etf.add_argument("symbols", nargs="+", help="the ETFs, such as QQQ")
    etf.add_argument(
        "--from-year",
        type=int,
        default=2016,
        help="first year; Alpaca's history starts in 2016",
    )
    etf.add_argument(
        "--to-year", type=int, help="last year; defaults to the year of the last day"
    )
    etf.set_defaults(run=_etf_minutes)

    plan = commands.add_parser(
        "plan", help="build a day's plan from the daily bars before it"
    )
    plan.add_argument(
        "--date",
        type=date.fromisoformat,
        help="trading day (YYYY-MM-DD); defaults to the session after the last bars",
    )
    plan.add_argument(
        "--capital",
        type=float,
        default=30_000.0,
        help="account value in USD; only a tiny one limits the stocks held",
    )
    plan.add_argument(
        "--synthetic", action="store_true", help="plan from the synthetic bars"
    )
    plan.add_argument("--guard", default="guard.yaml", help="the risk limits file")
    plan.add_argument("--out", help="where to write the plan's JSON")
    plan.set_defaults(run=_plan)

    backtest = commands.add_parser(
        "backtest",
        help="replay the nightly plan on the stored minutes, against SPY and the "
        "same stocks at equal weight",
    )
    backtest.add_argument(
        "--from",
        dest="first",
        type=date.fromisoformat,
        help="first day (YYYY-MM-DD); defaults to the first with a year of daily "
        "returns before it",
    )
    backtest.add_argument(
        "--to",
        dest="last",
        type=date.fromisoformat,
        help="last day; defaults to the last with daily bars and minutes",
    )
    backtest.add_argument(
        "--capital",
        type=float,
        nargs="+",
        default=[500.0, 5_000.0, 30_000.0],
        help="starting cash in USD; one run each",
    )
    backtest.add_argument("--guard", default="guard.yaml", help="the risk limits file")
    backtest.add_argument("--out", help="where to write the report's JSON")
    backtest.set_defaults(run=_backtest)

    live = commands.add_parser(
        "live",
        help="run a session of the QQQ strategy on Alpaca's paper account",
    )
    live.add_argument(
        "--day",
        type=date.fromisoformat,
        help="the session (YYYY-MM-DD); defaults to today in New York",
    )
    live.add_argument("--guard", default="guard_qqq.yaml", help="the risk limits file")
    live.add_argument(
        "--root", default="data/live", help="where each day's journal and alerts go"
    )
    live.add_argument("--s3-bucket", help="keep each day's folder in this bucket too")
    live.add_argument("--alert-topic", help="an SNS topic's ARN to e-mail alerts to")
    live.add_argument(
        "--check",
        action="store_true",
        help="the morning, a reconciliation and both streams, then out; nothing traded",
    )
    live.set_defaults(run=_live)

    report = commands.add_parser(
        "live-report",
        help="a paper session's report: what the engine did, and whether a backtest agrees",
    )
    report.add_argument(
        "--day",
        type=date.fromisoformat,
        help="the session (YYYY-MM-DD); defaults to today in New York",
    )
    report.add_argument(
        "--guard", default="guard_qqq.yaml", help="the risk limits file"
    )
    report.add_argument(
        "--root", default="data/live", help="where each day's folder is"
    )
    report.add_argument(
        "--s3-bucket",
        help="fetch the day's folder from this bucket first, and keep the report there",
    )
    report.add_argument(
        "--alert-topic", help="an SNS topic's ARN to e-mail the summary to"
    )
    report.set_defaults(run=_live_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


def _ingest(args: argparse.Namespace) -> int:
    if args.end_date:
        end_date = args.end_date
    elif args.synthetic:
        end_date = Config().end_date
    else:
        end_date = datetime.now(UTC).date().isoformat()

    config = Config(synthetic=args.synthetic, end_date=end_date)

    df, broken = repair_broken_bars(load_prices(config))
    print(f"Loaded {len(df)} rows for {df['ticker'].n_unique()} tickers")
    if not broken.is_empty():
        cases = ", ".join(
            f"{ticker} {day}" for ticker, day in broken.select("ticker", "date").rows()
        )
        print(
            f"[warning] repaired_bars: {broken.height} bars had an open or close "
            f"outside low..high; high and low were widened to them: {cases}"
        )
    if not args.synthetic:
        print(f"S&P 500 member-days with prices: {sp500_coverage(df, sp500()):.1%}")

    report = validate_prices(df)
    for issue in report.issues:
        print(f"[{issue.severity}] {issue.check}: {issue.detail}")
    if not report.ok:
        print("validation failed; nothing written")
        return 1

    path = write_prices(df, prices_path(config))
    print(f"wrote {path}")
    if args.s3_bucket:
        uri = upload_to_s3(path, args.s3_bucket)
        print(f"uploaded {uri}")
    return 0


def _intraday(args: argparse.Namespace) -> int:
    config = Config()
    daily_path = prices_path(config)
    if not daily_path.exists():
        print(f"{daily_path} not found; run `qfin ingest` first", file=sys.stderr)
        return 1
    daily = read_prices(daily_path)
    last_day = _last_minute_day()
    years = range(args.from_year, (args.to_year or last_day.year) + 1)
    universes = trading_universes(daily, years)
    done = backfill_minutes(
        # EODHD first for the stocks; Alpaca fills its holes (decision 0008).
        (EodhdClient(api_key()), AlpacaClient(*api_keys())),
        daily,
        sp500(),
        universes,
        config.raw_dir,
        last_day,
        workers=args.workers,
        report=_report_year,
    )
    return _summary(done)


def _etf_minutes(args: argparse.Namespace) -> int:
    config = Config()
    last_day = _last_minute_day()
    years = range(args.from_year, (args.to_year or last_day.year) + 1)
    eodhd = EodhdClient(api_key())
    alpaca = AlpacaClient(*api_keys())
    done = []
    for symbol in args.symbols:
        # EODHD's daily bars say which days had a session, and each day's
        # last minute must match their close, as for the stocks.
        daily = eodhd.daily_bars(symbol, date(years.start, 1, 1), last_day)
        done += backfill_minutes(
            (alpaca, eodhd),  # Alpaca first for an ETF (decision 0011)
            daily,
            sp500(),  # an ETF is not in the index: its own code is its only source
            {year: (symbol,) for year in years},
            config.raw_dir,
            last_day,
            workers=2,  # Alpaca allows 200 requests a minute
            report=_report_year,
        )
    return _summary(done)


def _last_minute_day() -> date:
    """The last day whose minutes both providers serve.

    Alpaca's free plan refuses SIP minutes less than 15 minutes old, and a
    request runs to the end of its last UTC day. So the last day is the one
    before the UTC day of 16 minutes ago: a session is stored from the next
    UTC day on.
    """
    return (datetime.now(UTC) - timedelta(minutes=16)).date() - timedelta(1)


def _report_year(stored: YearOfMinutes) -> None:
    if stored.error:
        print(f"{stored.year} {stored.ticker:9} failed: {stored.error}", flush=True)
        return
    parts = ", ".join(f"{name} ({code}) {days}" for name, code, days in stored.parts)
    dropped = f", {stored.dropped} broken minutes dropped" if stored.dropped else ""
    print(
        f"{stored.year} {stored.ticker:9} {stored.rows:>7} minutes on {stored.days}"
        f" of {stored.trading_days} days, days from {parts or 'nowhere'};"
        f" {stored.matching:.0%} of them match the daily close{dropped}",
        flush=True,
    )


def _summary(done: list[YearOfMinutes]) -> int:
    """Print what was stored and what was not; 1 if a download failed."""
    failed = [f"{stored.ticker} {stored.year}" for stored in done if stored.error]
    missing = [
        f"{stored.ticker} {stored.year}"
        for stored in done
        if not stored.parts and not stored.error
    ]
    short = [
        f"{stored.ticker} {stored.year} ({stored.days} of {stored.trading_days} days)"
        for stored in done
        if stored.parts and stored.days < stored.trading_days
    ]
    by_provider = Counter()
    for stored in done:
        for name, _, days in stored.parts:
            by_provider[name] += days
    provided = ", ".join(f"{name} {days:,}" for name, days in by_provider.items())
    print(
        f"stored {len(done) - len(failed) - len(missing)} ticker-years; "
        f"days from {provided or 'nowhere'}; "
        f"without matching minutes: {missing or 'none'}; failed: {failed or 'none'}; "
        f"days without minutes: {short or 'none'}"
    )
    return 1 if failed else 0


def _plan(args: argparse.Namespace) -> int:
    config = Config(synthetic=args.synthetic)
    daily_path = prices_path(config)
    if not daily_path.exists():
        print(f"{daily_path} not found; run `qfin ingest` first", file=sys.stderr)
        return 1
    prices = read_prices(daily_path)
    if args.date:
        day = args.date
        if session_days(day, day) != [day]:
            print(f"{day} is not a trading day", file=sys.stderr)
            return 1
    else:
        last = prices["date"].max()
        day = session_days(last + timedelta(1), last + timedelta(10))[0]
    # A day's plan is made from the bars of the session before it; from older
    # bars it would be the plan of an earlier day.
    before = session_days(day - timedelta(10), day - timedelta(1))[-1]
    if not (prices["date"] == before).any():
        print(
            f"no bars for {before}, the session before {day}; run `qfin ingest` first",
            file=sys.stderr,
        )
        return 1
    max_weight = load_guard_config(args.guard).max_weight
    plan = build_plan(prices, day, args.capital, max_weight)
    out = Path(args.out) if args.out else Path("data/plans") / f"daily_plan_{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plan_to_json(plan), encoding="utf-8")
    weights = sorted(plan.target_weights.values(), reverse=True)
    print(
        f"plan for {day}: {len(weights)} stocks, largest weight {weights[0]:.1%}, "
        f"invested {sum(weights):.1%}; wrote {out}"
    )
    return 0


def _backtest(args: argparse.Namespace) -> int:
    config = Config()
    daily_path = prices_path(config)
    if not daily_path.exists():
        print(f"{daily_path} not found; run `qfin ingest` first", file=sys.stderr)
        return 1
    daily = read_prices(daily_path)
    limits = load_guard_config(args.guard)
    started = time.monotonic()
    test = Backtest(daily, config.raw_dir, limits, args.first, args.last)
    days = test.inputs.days
    print(f"backtest from {days[0]} to {days[-1]}: {len(days):,} sessions", flush=True)
    runs = {}
    for capital in args.capital:
        runs[capital] = test.run(capital)
        elapsed = time.monotonic() - started
        print(f"ran {capital:,.0f} USD; {elapsed:.0f}s so far", flush=True)
    plans = test.plans(args.capital[0])
    report = summary(daily, test.inputs, plans, runs, test.costs)
    print(render(report))
    name = f"backtest_{days[0]}_{days[-1]}.json"
    out = Path(args.out) if args.out else Path("data/backtests") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _live(args: argparse.Namespace) -> int:
    from zoneinfo import ZoneInfo

    from qfin_alphaguard.live.runner import DayFiles, check_session, run_session

    day = args.day or datetime.now(ZoneInfo("America/New_York")).date()
    if not sessions(day, day):
        print(f"no session on {day}")
        return 0
    folder = DayFiles(Path(args.root), day).folder
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(folder / "engine.log")],
    )
    limits = load_guard_config(args.guard)
    if args.check:
        print(asyncio.run(check_session(day, Path(args.root), limits, api_keys())))
        return 0
    outcome = asyncio.run(
        run_session(
            day,
            Path(args.root),
            limits,
            api_keys(),
            bucket=args.s3_bucket,
            topic=args.alert_topic,
        )
    )
    print(f"session {day}: {outcome}")
    return 0 if outcome == "closed" else 1


def _live_report(args: argparse.Namespace) -> int:
    from zoneinfo import ZoneInfo

    from qfin_alphaguard.data.intraday import regular_session
    from qfin_alphaguard.events import Bar
    from qfin_alphaguard.live.journal import read_journal
    from qfin_alphaguard.live.report import (
        bars_of,
        chart_series,
        day_report,
        render_html,
    )
    from qfin_alphaguard.live.runner import TICKER, Backup, DayFiles, past_sessions

    new_york = ZoneInfo("America/New_York")
    day = args.day or datetime.now(new_york).date()
    files = DayFiles(Path(args.root), day)
    backup = Backup(files, args.s3_bucket) if args.s3_bucket else None
    if backup:
        backup.pull()
    if not files.journal.exists():
        print(f"no journal for {day} in {files.folder}")
        return 0 if not sessions(day, day) else 1
    entries = read_journal(files.journal)
    client = AlpacaClient(*api_keys())
    past = past_sessions(client, day, files.warm_up)
    feeds = {}
    for name, feed in (("IEX's history", "iex"), ("SIP", "sip")):
        try:
            frame = regular_session(client.minute_bars(TICKER, day, day, feed=feed))
        except RuntimeError as error:  # SIP's last 15 minutes are not served yet
            print(f"{name} left out: {error}")
            continue
        feeds[name] = bars_of(frame, TICKER)
    alerts = files.alert.read_text().splitlines() if files.alert.exists() else []
    limits = load_guard_config(args.guard)
    report = day_report(entries, past, limits, feeds, alerts)
    live_bars = [e.record for e in entries if isinstance(e.record, Bar)]
    session = sessions(day, day)[0]
    points = chart_series(report, live_bars, past, limits, session)
    (files.folder / "report.txt").write_text(report.summary() + "\n", encoding="utf-8")
    page = render_html(report, points, lambda t: t.astimezone(new_york))
    (files.folder / "report.html").write_text(page, encoding="utf-8")
    print(report.summary())
    print(f"page: {files.folder / 'report.html'}")
    if backup:
        backup.push()
    if args.alert_topic:
        import boto3

        boto3.client("sns").publish(
            TopicArn=args.alert_topic,
            Subject=f"qfin live {day}: report",
            Message=report.summary(),
        )
    return 0
