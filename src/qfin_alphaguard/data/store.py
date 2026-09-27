"""Read and write price data as Parquet files."""

from pathlib import Path

import polars as pl
from qfin_alphaguard.config import Config

def prices_path(config:Config) -> Path:
    """Where raw prices file for this config lives."""
    name ="prices_synthetic.parquet" if config.synthetic else "prices.parquet"
    return config.raw_dir / name

def write_prices(df:pl.DataFrame,path:Path) -> Path:
    """Write prices to parquet, creating parent folders if needed"""
    path.parent.mkdir(parents=True,exist_ok=True) 
    df.write_parquet(path)
    return path

def read_prices(path:Path ) ->pl.DataFrame:
    """Read prices back from Parquet."""
    return pl.read_parquet(path)