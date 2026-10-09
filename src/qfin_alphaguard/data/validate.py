"""Data quality checks. Reports problems but does not decide what to do about them."""

from dataclasses import dataclass

import polars as pl

PRICE_COLUMNS = ("open", "high", "low", "close")
# Prices and the split and dividend factors must all be finite and positive.
POSITIVE_COLUMNS = (*PRICE_COLUMNS, "adjustment", "split_factor")
# A one-day move this large is rare enough to look at every time it happens.
LARGE_MOVE = 0.5


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
    issues.extend(_check_missing_values(df))
    issues.extend(_check_equal_row_counts(df))
    issues.extend(_check_duplicate(df))
    issues.extend(_check_positive_prices(df))
    issues.extend(_check_ohlc_consistency(df))
    issues.extend(_check_volume(df))
    issues.extend(_check_constant_series(df))
    issues.extend(_check_large_moves(df))
    return ValidationReport(issues=tuple(issues))


def _check_missing_values(df: pl.DataFrame) -> list[Issue]:
    missing = df.filter(
        pl.any_horizontal(pl.all().is_null())
        | pl.any_horizontal(pl.col(POSITIVE_COLUMNS).is_nan())
    )
    if missing.is_empty():
        return []
    return [
        Issue(
            check="missing_values",
            detail=f"{len(missing)} rows with an empty or NaN value",
            severity="error",
        )
    ]


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
            detail=f"{n_duplicates} duplicate ticker/date rows",
            severity="error",
        )
    ]


def _check_positive_prices(df: pl.DataFrame) -> list[Issue]:
    positive_prices = df.filter(
        (pl.min_horizontal(POSITIVE_COLUMNS) <= 0)
        | pl.any_horizontal(pl.col(POSITIVE_COLUMNS).is_infinite())
    )
    if len(positive_prices) == 0:
        return []
    return [
        Issue(
            check="positive_prices",
            detail=(
                f"{len(positive_prices)} rows with a price or adjustment that is "
                "zero, negative or infinite"
            ),
            severity="error",
        )
    ]


def _check_ohlc_consistency(df: pl.DataFrame) -> list[Issue]:
    # The same rule as the Bar event: low <= open, close <= high.
    broken = df.filter(
        (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.max_horizontal("open", "close"))
    )
    if broken.is_empty():
        return []
    first = broken.row(0, named=True)
    return [
        Issue(
            check="ohlc_consistency",
            detail=(
                f"{len(broken)} rows where open or close is outside low..high, "
                f"first {first['ticker']} on {first['date']}"
            ),
            severity="error",
        )
    ]


def _check_volume(df: pl.DataFrame) -> list[Issue]:
    negative = df.filter(pl.col("volume") < 0)
    if negative.is_empty():
        return []
    return [
        Issue(
            check="volume",
            detail=f"{len(negative)} rows with negative volume",
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


def _check_large_moves(df: pl.DataFrame) -> list[Issue]:
    # Most are real (SVB fell 60% on 9 March 2023), but a bad price looks the
    # same, so the latest one is named for a person to check.
    moves = (
        df.sort(["ticker", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias(
                "move"
            )
        )
        .filter(pl.col("sp500") & (pl.col("move").abs() > LARGE_MOVE))
    )
    if moves.is_empty():
        return []
    latest = moves.sort("date").row(-1, named=True)
    return [
        Issue(
            check="large_moves",
            detail=(
                f"{len(moves)} index-member days moved more than {LARGE_MOVE:.0%} "
                f"in a day; latest {latest['ticker']} on {latest['date']}"
            ),
            severity="warning",
        )
    ]
