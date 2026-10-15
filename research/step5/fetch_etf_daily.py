"""Dividend-adjusted daily bars of the 16 stored ETFs from EODHD, since their start."""

from datetime import date
from pathlib import Path

from qfin_alphaguard.data.eodhd import EodhdClient, api_key

OUT = Path("data/raw/etf_daily")
client = EodhdClient(api_key())
for t in [
    "QQQ",
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "HYG",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]:
    d = client.daily_bars(t, date(1993, 1, 1), date(2026, 10, 14))
    d.write_parquet(OUT / f"{t}.parquet")
    print(t, d.height, d["date"].min(), d["date"].max())
