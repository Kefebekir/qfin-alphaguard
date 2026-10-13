"""Command line entry point.

    qfin ingest       load, validate and store daily prices (the nightly AWS job)
    qfin intraday     download 1-minute bars for each year's trading universe
    qfin etf-minutes  download ETFs' 1-minute bars from Alpaca, such as QQQ's
    qfin plan         build a day's plan: the portfolio to hold, from the bars before it
    qfin backtest     run the event-driven backtester (Phase 1, not built yet)

The nightly AWS job runs the command in infra/aws/task-definition.template.json;
a test checks that this CLI still accepts it.
"""

import argparse
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.eodhd import EodhdClient, api_key
from qfin_alphaguard.data.ingest import load_prices, sp500_coverage
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
from qfin_alphaguard.sessions import session_days


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
        "backtest", help="run the event-driven backtester (not built yet)"
    )
    backtest.set_defaults(run=_not_built_yet)

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

    df = load_prices(config)
    print(f"Loaded {len(df)} rows for {df['ticker'].n_unique()} tickers")
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


def _not_built_yet(args: argparse.Namespace) -> int:
    print(
        f"qfin {args.command} is not built yet; it arrives in Phase 1", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
