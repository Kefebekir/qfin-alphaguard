"""Daily prices from EODHD (eodhd.com), including stocks that have left the exchange.

The API key is read from the EODHD_API_KEY environment variable. Locally it
lives in the git-ignored .env file (`uv run --env-file .env qfin ingest`); on
AWS it comes from an SSM parameter (infra/aws/task-definition.template.json).
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from datetime import date

import polars as pl

BASE_URL = "https://eodhd.com/api"
ATTEMPTS = 4  # for a rate limit (429), a server error (5xx) or a network error

BARS_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adjustment": pl.Float64,
}


def api_key() -> str:
    key = os.environ.get("EODHD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "EODHD_API_KEY is not set; locally, run `uv run --env-file .env qfin ingest`"
        )
    return key


def eodhd_code(ticker: str) -> str:
    """Index lists write class shares with a dot (BRK.B); EODHD uses a dash (BRK-B)."""
    return ticker.replace(".", "-")


class EodhdClient:
    def __init__(
        self,
        key: str,
        urlopen: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = key
        self._urlopen = urlopen
        self._sleep = sleep

    def _get(self, path: str, **params: str) -> object:
        """One API call; None if EODHD does not know the symbol (404)."""
        query = urllib.parse.urlencode(
            {**params, "api_token": self._key, "fmt": "json"}
        )
        url = f"{BASE_URL}/{path}?{query}"
        for attempt in range(1, ATTEMPTS + 1):
            try:
                with self._urlopen(url, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                problem = f"HTTP {error.code}"
                if error.code != 429 and error.code < 500:
                    break  # a bad request or key does not get better by retrying
            except OSError as error:  # network errors and timeouts
                problem = str(getattr(error, "reason", error))
            if attempt < ATTEMPTS:
                self._sleep(2**attempt)
        # Raised outside the except blocks so the traceback does not carry the
        # URL, which contains the key.
        raise RuntimeError(f"EODHD request for {path} failed: {problem}")

    def daily_bars(self, code: str, start: date, end: date) -> pl.DataFrame:
        """Daily bars for one EODHD code, adjusted for splits and dividends.

        EODHD reports open, high, low and close as traded, plus an adjusted
        close. All four prices are scaled by adjusted close / close, kept as
        `adjustment`, so the price as traded is the adjusted price divided by
        it. Volume comes adjusted for splits already.

        Days without trades are dropped: EODHD repeats the last price with zero
        volume on them, for example for weeks after a delisting.
        """
        rows = self._get(
            f"eod/{code}.US", **{"from": start.isoformat(), "to": end.isoformat()}
        )
        if not rows:
            return pl.DataFrame(schema=BARS_SCHEMA)
        traded = (
            pl.DataFrame(rows, infer_schema_length=None)
            .select(
                pl.col("date").str.to_date(),
                pl.col("open", "high", "low", "close", "adjusted_close").cast(
                    pl.Float64
                ),
                pl.col("volume").cast(pl.Int64),
            )
            .filter(pl.col("volume") > 0)
        )
        adjustment = pl.col("adjusted_close") / pl.col("close")
        return traded.select(
            "date",
            pl.lit(code).alias("ticker"),
            (pl.col("open") * adjustment).alias("open"),
            (pl.col("high") * adjustment).alias("high"),
            (pl.col("low") * adjustment).alias("low"),
            pl.col("adjusted_close").alias("close"),
            "volume",
            adjustment.alias("adjustment"),
        )

    def old_codes(self) -> dict[str, tuple[str, ...]]:
        """EODHD codes of earlier companies whose ticker was later reused, by ticker.

        EODHD keeps them as TICKER_old, TICKER_old1, TICKER_old2 and so on:
        DOW_old is Dow Chemical, while DOW is today's Dow Inc.
        """
        listing = self._get("exchange-symbol-list/US", delisted="1") or []
        found: dict[str, list[str]] = defaultdict(list)
        for row in listing:
            match = re.fullmatch(r"(.+)_old\d*", row.get("Code") or "")
            if match:
                found[match.group(1)].append(row["Code"])
        return {ticker: tuple(sorted(codes)) for ticker, codes in found.items()}
