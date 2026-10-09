"""Command line entry point.

    qfin ingest     load, validate and store daily prices (the nightly AWS job)
    qfin intraday   download 1-minute bars for each year's trading universe
    qfin plan       build the next day's daily_plan.json (Phase 1, not built yet)
    qfin backtest   run the event-driven backtester (Phase 1, not built yet)

The nightly AWS job runs the command in infra/aws/task-definition.template.json;
a test checks that this CLI still accepts it.
"""

import argparse
import sys
from datetime import UTC, datetime

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
from qfin_alphaguard.data.universe import (
    TRADING_UNIVERSE_SIZE,
    sp500,
    trading_universe,
)
from qfin_alphaguard.data.validate import validate_prices


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
        "intraday", help="download 1-minute bars for each year's trading universe"
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
        "plan", help="build the next day's daily_plan.json (not built yet)"
    )
    plan.set_defaults(run=_not_built_yet)

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
    universes = {
        year: trading_universe(daily, year, TRADING_UNIVERSE_SIZE) for year in years
    }

    def report(stored: YearOfMinutes) -> None:
        if stored.error:
            print(f"{stored.year} {stored.ticker:9} failed: {stored.error}", flush=True)
            return
        source = stored.source or "nowhere"
        print(
            f"{stored.year} {stored.ticker:9} {stored.rows:>7} minutes from {source:9}"
            f" {stored.matching:.0%} of days match the daily close",
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
    print(
        f"stored {len(done) - len(failed) - len(missing)} stock-years; "
        f"without matching minutes: {missing or 'none'}; failed: {failed or 'none'}"
    )
    return 1 if failed else 0


def _not_built_yet(args: argparse.Namespace) -> int:
    print(
        f"qfin {args.command} is not built yet; it arrives in Phase 1", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
