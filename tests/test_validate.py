from datetime import date

import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.ingest import load_prices
from qfin_alphaguard.data.validate import validate_prices


def test_clean_data_has_no_issues():
    report = validate_prices(load_prices(Config(synthetic=True)))
    assert report.ok


def test_data_with_duplicates_has_error():
    df = load_prices(Config(synthetic=True))
    broken = pl.concat([df, df.head(1)])
    report = validate_prices(broken)
    assert not report.ok
    assert any(i.check == "duplicates" for i in report.issues)


def test_non_positive_prices_has_error():
    df = load_prices(Config(synthetic=True))
    broken = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(-5.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    report = validate_prices(broken)
    assert not report.ok
    assert any(i.check == "positive_prices" for i in report.issues)


def test_constant_series_has_flagged():
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "ticker": ["AAA", "AAA", "AAA"],
            "close": [100.0, 100.0, 100.0],
        }
    )
    report = validate_prices(df)
    assert report.ok
    assert any(i.check == "constant_series" for i in report.issues)
