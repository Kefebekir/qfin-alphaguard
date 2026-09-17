"""Configuration for the QFin-AlphaGuard pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # which assests to pull and over what window
    tickers: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "NVDA",
        "ORCL",
        "CSCO",
        "JNJ",
        "PFE",
        "UNH",
        "ABT",
        "MRK",
        "JPM",
        "BAC",
        "GS",
        "AXP",
        "BLK",
        "XOM",
        "CVX",
        "COP",
        "NEE",
        "DUK",
        "PG",
        "KO",
        "WMT",
        "MCD",
        "NKE",
        "CAT",
        "HON",
        "UNP",
        "LMT",
        "GE",
    )
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # where data lands on disk

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # use generated data instead of hitting the network
    synthetic: bool = False
    seed: int = 42
