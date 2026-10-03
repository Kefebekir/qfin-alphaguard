"""Command line entry point:load,validate and store price data."""

import argparse

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices
from qfin_alphaguard.data.store import prices_path, upload_to_s3,write_prices
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
    args = parser.parse_args(argv)

    config = Config(synthetic=args.synthetic)

    df = load_prices(config)
    print(f"Load {len(df)} rows for {df['ticker'].n_unique()} tickers")

    report = validate_prices(df)
    for issue in report.issues:
        print(f"[{issue.severity}] {issue.check}: {issue.detail}")
    if not report.ok:
        print("validation failed; nothing written")
        return 1

    path = write_prices(df, prices_path(config))
    print(f"wrote {path}")
    if args.s3_bucket:
        uri=upload_to_s3(path,args.s3_bucket)
        print(f"uploaded {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
