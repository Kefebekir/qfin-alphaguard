from datetime import date

import polars as pl
import pytest

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices
from qfin_alphaguard.data.validate import validate_prices


def errors(report):
    return {issue.check for issue in report.issues if issue.severity == "error"}


def set_first_row(df, column, value):
    return df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(value)
        .otherwise(pl.col(column))
        .alias(column)
    )


def test_clean_data_has_no_issues():
    report = validate_prices(load_prices(Config(synthetic=True)))
    assert report.ok


def test_data_with_duplicates_has_error():
    df = load_prices(Config(synthetic=True))
    broken = pl.concat([df, df.head(1)])
    assert "duplicates" in errors(validate_prices(broken))


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_non_positive_prices_has_error(column):
    broken = set_first_row(load_prices(Config(synthetic=True)), column, -5.0)
    assert "positive_prices" in errors(validate_prices(broken))


def test_close_above_the_high_has_error():
    df = load_prices(Config(synthetic=True))
    broken = set_first_row(df, "close", df["high"][0] * 1.01)
    assert "ohlc_consistency" in errors(validate_prices(broken))


def test_infinite_adjustment_has_error():
    # adjusted close / close is infinite when EODHD reports a close of zero
    df = load_prices(Config(synthetic=True))
    broken = set_first_row(df, "adjustment", float("inf"))
    assert "positive_prices" in errors(validate_prices(broken))


def test_nan_price_has_error():
    broken = set_first_row(load_prices(Config(synthetic=True)), "open", float("nan"))
    assert "missing_values" in errors(validate_prices(broken))


def test_negative_volume_has_error():
    broken = set_first_row(load_prices(Config(synthetic=True)), "volume", -1)
    assert "volume" in errors(validate_prices(broken))


def test_constant_series_is_flagged():
    days = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    df = pl.DataFrame(
        {
            "date": days,
            "ticker": ["AAA"] * 3,
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
            "volume": [1000] * 3,
            "adjustment": [1.0] * 3,
            "sp500": [True] * 3,
        }
    )
    report = validate_prices(df)
    assert report.ok
    assert any(i.check == "constant_series" for i in report.issues)
