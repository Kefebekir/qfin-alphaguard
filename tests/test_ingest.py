from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices, snap_rounding
from qfin_alphaguard.data.validate import validate_prices

EXPECTED_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}


def test_load_prices_synthetic():
    df = load_prices(Config(synthetic=True))
    assert dict(df.schema) == EXPECTED_SCHEMA


def test_load_prices_is_sorted_by_ticker_then_date():
    df = load_prices(Config(synthetic=True))
    assert df.equals(df.sort(["ticker", "date"]))


def fake_download(*args, **kwargs):
    """A frame shaped like yf.download's: one row per date, (field, ticker) columns.

    BBB is not listed yet on the first day, so all its fields are NaN there.
    """
    columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], ["AAA", "BBB"]],
        names=["Price", "Ticker"],
    )
    nan = np.nan
    rows = [
        # Close       High          Low          Open          Volume
        [101.0, nan, 102.0, nan, 99.0, nan, 100.0, nan, 1000.0, nan],
        [102.0, 51.0, 103.0, 52.0, 100.5, 49.0, 101.0, 50.0, 1100.0, 500.0],
    ]
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    return pd.DataFrame(rows, index=index, columns=columns)


def test_download_gives_one_row_per_date_and_ticker(monkeypatch):
    monkeypatch.setattr("qfin_alphaguard.data.ingest.yf.download", fake_download)

    df = load_prices(Config(tickers=("AAA", "BBB")))

    assert dict(df.schema) == EXPECTED_SCHEMA
    assert df.select("ticker", "date").rows() == [
        ("AAA", date(2024, 1, 2)),
        ("AAA", date(2024, 1, 3)),
        ("BBB", date(2024, 1, 3)),  # the NaN day before the listing is dropped
    ]
    assert df.row(2, named=True) == {
        "date": date(2024, 1, 3),
        "ticker": "BBB",
        "open": 50.0,
        "high": 52.0,
        "low": 49.0,
        "close": 51.0,
        "volume": 500,
    }


def one_bar(**prices):
    return pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "ticker": ["AAA"],
            **{name: [value] for name, value in prices.items()},
            "volume": [1000],
        }
    )


def test_close_a_rounding_step_above_the_high_is_snapped():
    close = np.nextafter(100.0, 101.0)  # the next float above 100
    bars = one_bar(open=99.0, high=100.0, low=98.0, close=close)

    snapped = snap_rounding(bars)

    assert snapped["high"][0] == close
    assert validate_prices(snapped).ok


def test_open_a_rounding_step_below_the_low_is_snapped():
    low = np.nextafter(98.0, 99.0)  # the next float above the open
    bars = one_bar(open=98.0, high=100.0, low=low, close=99.0)

    snapped = snap_rounding(bars)

    assert snapped["low"][0] == 98.0
    assert validate_prices(snapped).ok


def test_a_real_inconsistency_is_left_for_validation():
    bars = one_bar(open=99.0, high=100.0, low=98.0, close=101.0)

    snapped = snap_rounding(bars)

    assert snapped.equals(bars)
    assert not validate_prices(snapped).ok
