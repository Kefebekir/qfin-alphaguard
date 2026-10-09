"""Command line entry point.

    qfin ingest     load, validate and store daily prices (the nightly AWS job)
    qfin intraday   download 1-minute bars for each year's trading universe
    qfin plan       build a day's plan: the portfolio to hold, from the bars before it
    qfin backtest   run the event-driven backtester (Phase 1, not built yet)

The nightly AWS job runs the command in infra/aws/task-definition.template.json;
a test checks that this CLI still accepts it.
"""

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from qfin_alphaguard.config import Config
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
        help="account value in USD; it sets how many stocks are held",
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
    today = datetime.now(UTC).date()
    years = range(args.from_year, (args.to_year or today.year) + 1)
    universes = trading_universes(daily, years)

    def report(stored: YearOfMinutes) -> None:
        if stored.error:
            print(f"{stored.year} {stored.ticker:9} failed: {stored.error}", flush=True)
            return
        source = stored.source or "nowhere"
        dropped = f", {stored.dropped} broken minutes dropped" if stored.dropped else ""
        print(
            f"{stored.year} {stored.ticker:9} {stored.rows:>7} minutes from {source:9}"
            f" on {stored.days} of {stored.trading_days} days,"
            f" {stored.matching:.0%} of them match the daily close{dropped}",
            flush=True,
        )

    done = backfill_minutes(
        EodhdClient(api_key()),
        daily,
        sp500(),
        universes,
        config.raw_dir,
        today,
        workers=args.workers,
        report=report,
    )
    failed = [f"{stored.ticker} {stored.year}" for stored in done if stored.error]
    missing = [
        f"{stored.ticker} {stored.year}"
        for stored in done
        if not stored.source and not stored.error
    ]
    short = [
        f"{stored.ticker} {stored.year} ({stored.days} of {stored.trading_days} days)"
        for stored in done
        if stored.source and stored.days < stored.trading_days
    ]
    print(
        f"stored {len(done) - len(failed) - len(missing)} stock-years; "
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
