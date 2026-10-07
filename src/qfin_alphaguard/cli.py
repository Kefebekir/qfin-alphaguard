"""Command line entry point: load, validate and store price data."""

import argparse
from datetime import UTC, datetime

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices
from qfin_alphaguard.data.store import prices_path, upload_to_s3, write_prices
from qfin_alphaguard.data.validate import validate_prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qfin",
        description="Load, validate and store price data.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use generated data instead of downloading",
    )
    parser.add_argument(
        "--s3-bucket",
        help="also upload the result to this S3 bucket",
    )
    parser.add_argument(
        "--end-date",
        help="last date to download (YYYY-MM-DD); defaults to today for real data",
    )
    args = parser.parse_args(argv)

    if args.end_date:
        end_date = args.end_date
    elif args.synthetic:
        end_date = Config().end_date
    else:
        end_date = datetime.now(UTC).date().isoformat()

    config = Config(synthetic=args.synthetic, end_date=end_date)

    df = load_prices(config)
    print(f"Loaded {len(df)} rows for {df['ticker'].n_unique()} tickers")

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


if __name__ == "__main__":
    raise SystemExit(main())
