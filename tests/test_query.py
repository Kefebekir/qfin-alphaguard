from datetime import date

import polars as pl
import pytest

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices
from qfin_alphaguard.data.query import latest_closes, row_counts
from qfin_alphaguard.data.store import write_prices


@pytest.fixture
def prices_file(tmp_path):
    df = load_prices(Config(synthetic=True))
    return write_prices(df, tmp_path / "prices.parquet")


def test_row_counts_one_row_per_ticker(prices_file):
    result = row_counts(prices_file)
    assert len(result) == len(Config().tickers)
    assert result["n_rows"].n_unique() == 1


def test_latest_closes_all_on_last_date(prices_file):
    result = latest_closes(prices_file)
    assert len(result) == len(Config().tickers)
    assert result["date"].n_unique() == 1
    assert (result["close"] > 0).all()


def test_latest_closes_picks_last_date_not_highest_price(tmp_path):
    df = pl.DataFrame(
        {
            "date": [
                date(2024, 12, 27),
                date(2024, 12, 30),
                date(2024, 12, 31),
                date(2024, 12, 27),
                date(2024, 12, 30),
                date(2024, 12, 31),
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "close": [150.0, 148.0, 149.0, 10.0, 20.0, 15.0],
        }
    )
    path = write_prices(df, tmp_path / "prices.parquet")

    result = latest_closes(path)

    assert result["ticker"].to_list() == ["AAA", "BBB"]
    assert result["close"].to_list() == [149.0, 15.0]
    assert result["date"].to_list() == [date(2024, 12, 31), date(2024, 12, 31)]
