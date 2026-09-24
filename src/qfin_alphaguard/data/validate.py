"""Data quality checks.Reports problems but does not deciede what to do about them."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Issue:
    check: str
    detail: str
    severity: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


def validate_prices(df: pl.DataFrame) -> ValidationReport:
    issues: list[Issue] = []
    issues.extend(_check_equal_row_counts(df))
    issues.extend(_check_duplicate(df))
    issues.extend(_check_positive_prices(df))
    issues.extend(_check_constant_series(df))
    return ValidationReport(issues=tuple(issues))


def _check_equal_row_counts(df: pl.DataFrame) -> list[Issue]:
    counts = df.group_by("ticker").len()
    if counts["len"].n_unique() == 1:
        return []
    shortest = counts.sort("len").head(1)
    return [
        Issue(
            check="equal_row_counts",
            detail=(
                f"tickers have different row counts: "
                f"min {counts['len'].min()} ({shortest['ticker'][0]}), "
                f"max {counts['len'].max()}"
            ),
            severity="warning",
        )
    ]


def _check_duplicate(df: pl.DataFrame) -> list[Issue]:
    n_duplicates = len(df) - len(df.unique(subset=["ticker", "date"]))
    if n_duplicates == 0:
        return []
    return [
        Issue(
            check="duplicates",
            detail=f"{n_duplicates} duplicate ticker/data rows",
            severity="error",
        )
    ]


def _check_positive_prices(df: pl.DataFrame) -> list[Issue]:
    positive_prices = df.filter(pl.col("close") <= 0)
    if len(positive_prices) == 0:
        return []
    return [
        Issue(
            check="positive_prices",
            detail=f"{len(positive_prices)} rows with non-positive prices",
            severity="error",
        )
    ]


def _check_constant_series(df: pl.DataFrame) -> list[Issue]:
    constant_series = df.group_by("ticker").agg(pl.col("close").n_unique())
    if constant_series.filter(pl.col("close") == 1).is_empty():
        return []
    return [
        Issue(
            check="constant_series",
            detail=f"{len(constant_series.filter(pl.col('close') == 1))} tickers with constant price series",
            severity="warning",
        )
    ]
