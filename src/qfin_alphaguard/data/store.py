"""Read and write price data as Parquet files."""

from pathlib import Path

import boto3
import polars as pl

from qfin_alphaguard.config import Config


def prices_path(config: Config) -> Path:
    """Where raw prices file for this config lives."""
    name = "prices_synthetic.parquet" if config.synthetic else "prices.parquet"
    return config.raw_dir / name


def write_prices(df: pl.DataFrame, path: Path) -> Path:
    """Write prices to parquet, creating parent folders if needed"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_prices(path: Path) -> pl.DataFrame:
    """Read prices back from Parquet."""
    return pl.read_parquet(path)


def upload_to_s3(path: Path, bucket: str) -> str:
    """Upload a local file to S3 under the same relative path. Returns the S3 URI."""
    key = path.as_posix()
    boto3.client("s3").upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"
