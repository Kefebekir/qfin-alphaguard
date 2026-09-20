import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices

EXPECTED_SCHEMA = {"date": pl.Date, "ticker": pl.String, "close": pl.Float64}


def test_load_prices_synthetic():
    df = load_prices(Config(synthetic=True))
    assert dict(df.schema) == EXPECTED_SCHEMA


def test_load_prices_network():
    df = load_prices(Config(synthetic=True))
    assert df.equals(df.sort(["ticker", "date"]))
