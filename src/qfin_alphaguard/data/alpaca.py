"""Bars and corporate actions from Alpaca's market data API (alpaca.markets), read only.

Alpaca's free plan serves the consolidated tape (SIP: trades on every US
exchange) since 2016, once it is at least 15 minutes old (decision 0008).
The keys are read from the ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY
environment variables; locally they live in the git-ignored .env file
(`uv run --env-file .env qfin etf-minutes QQQ`). They travel in request
headers, never in the URL. This client only reads market data: it has no way
to place an order.
"""

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime

import polars as pl

from qfin_alphaguard.data.eodhd import MINUTE_SCHEMA
from qfin_alphaguard.data.universe import IndexHistory

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
NEW_YORK = "America/New_York"  # a daily bar's session is its New York date
ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
ATTEMPTS = 4  # for a rate limit (429), a server error (5xx) or a network error
PAGE_SIZE = 10_000  # the most bars Alpaca sends in one answer
ACTIONS_PAGE_SIZE = 1_000  # the most corporate actions in one answer
SYMBOLS_A_REQUEST = 100  # keeps the address of a request short

# The corporate actions that change a stock's price series. Mergers and name
# changes are not asked for.
ACTION_TYPES = (
    "forward_split",
    "reverse_split",
    "stock_dividend",
    "cash_dividend",
    "spin_off",
)

DAILY_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}

# Alpaca follows a company through a change of ticker: Priceline's minutes are
# under PCLN and BKNG alike, and Facebook's under FB and META. Altaba's are the
# exception found: before June 2017 they are only under Yahoo's ticker.
EARLIER_TICKERS = {"AABA": ("YHOO",)}


def api_keys() -> tuple[str, str]:
    """The key ID and the secret key."""
    key_id = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    secret = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    if not (key_id and secret):
        raise RuntimeError(
            "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must both be set; "
            "locally, run `uv run --env-file .env qfin etf-minutes QQQ`"
        )
    return key_id, secret


class AlpacaClient:
    name = "Alpaca"

    def __init__(
        self,
        key_id: str,
        secret: str,
        urlopen: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self._urlopen = urlopen
        self._sleep = sleep

    def _get(self, params: dict[str, str], url: str = BARS_URL) -> dict:
        """One page of bars, or of corporate actions from ACTIONS_URL."""
        request = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}", headers=self._headers
        )
        for attempt in range(1, ATTEMPTS + 1):
            try:
                with self._urlopen(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                # Alpaca says why in the answer, such as "subscription does
                # not permit querying recent SIP data".
                reason = error.read(200).decode(errors="replace")
                problem = f"HTTP {error.code} {reason}".strip()
                if error.code != 429 and error.code < 500:
                    break  # a bad request or key does not get better by retrying
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
                problem = str(getattr(error, "reason", error)) or type(error).__name__
            if attempt < ATTEMPTS:
                self._sleep(2**attempt)
        # Raised outside the except blocks, so nothing chained to it carries
        # the request and its keys.
        symbols = params["symbols"].split(",")
        named = (
            symbols[0]
            if len(symbols) == 1
            else f"{symbols[0]} and {len(symbols) - 1} more"
        )
        raise RuntimeError(f"Alpaca request for {named} failed: {problem}")

    def minute_codes(self, code: str, history: IndexHistory) -> tuple[str, ...]:
        """Alpaca's tickers that may hold `code`'s 1-minute bars, to try in order.

        The daily data's code without its _old suffix (CHK_old's minutes are
        under CHK) and with a dot for a class share (BRK.B), then any earlier
        ticker Alpaca keeps apart. The index history is not needed.
        """
        ticker = re.sub(r"_old\d*$", "", code).replace("-", ".")
        return (ticker, *EARLIER_TICKERS.get(code, ()))

    def minute_bars(
        self, code: str, first: date, last: date, feed: str = "sip"
    ) -> pl.DataFrame:
        """1-minute bars from `first` to `last` (UTC days), both included, as traded.

        Like EodhdClient.minute_bars, a bar is labelled with the minute it
        starts, in UTC, and pre-market and after-hours trading are included:
        cut to the regular session with intraday.regular_session. Prices and
        volume are not adjusted for splits or dividends. Alpaca refuses the
        request if the end of `last`, at midnight UTC, is less than 15
        minutes ago. Long answers come in pages, oldest first, followed one by
        one.

        `feed` is "sip", the consolidated tape (every exchange), or "iex", IEX's trades
        alone: the only live feed of Alpaca's free plan, kept back to 2021.
        """
        if feed not in ("sip", "iex"):
            raise ValueError(f"feed must be sip or iex, got {feed!r}")
        params = {
            "symbols": code,
            "timeframe": "1Min",
            "start": f"{first.isoformat()}T00:00:00Z",
            "end": f"{last.isoformat()}T23:59:59Z",
            "feed": feed,
            "adjustment": "raw",
            "sort": "asc",
            "limit": str(PAGE_SIZE),
        }
        rows = []
        while True:
            page = self._get(params)
            rows += (page.get("bars") or {}).get(code, [])
            if not page.get("next_page_token"):
                break
            params = {**params, "page_token": page["next_page_token"]}
        if not rows:
            return pl.DataFrame(schema=MINUTE_SCHEMA)
        return pl.DataFrame(rows, infer_schema_length=None).select(
            pl.col("t")
            .str.to_datetime(time_zone="UTC")
            .dt.cast_time_unit("us")
            .alias("start"),
            pl.lit(code).alias("ticker"),
            pl.col("o").cast(pl.Float64).alias("open"),
            pl.col("h").cast(pl.Float64).alias("high"),
            pl.col("l").cast(pl.Float64).alias("low"),
            pl.col("c").cast(pl.Float64).alias("close"),
            pl.col("v").cast(pl.Int64).alias("volume"),
        )

    def daily_bars(
        self,
        symbols: Sequence[str],
        first: date,
        end: datetime,
        adjustment: str = "raw",
    ) -> pl.DataFrame:
        """Daily bars on the consolidated tape from `first` to before `end`.

        Each bar is labelled with its New York date. With `adjustment` "raw"
        prices and volume are as traded; Alpaca's other adjustments, such as
        "spin-off", round prices to the cent. Alpaca refuses an `end` less than
        15 minutes ago, and a session that is still open comes as its bar so far:
        the caller keeps only the sessions that are over. The symbols are asked
        for SYMBOLS_A_REQUEST at a time.
        """
        rows = []
        for start in range(0, len(symbols), SYMBOLS_A_REQUEST):
            params = {
                "symbols": ",".join(symbols[start : start + SYMBOLS_A_REQUEST]),
                "timeframe": "1Day",
                "start": first.isoformat(),
                "end": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "sip",
                "adjustment": adjustment,
                "sort": "asc",
                "limit": str(PAGE_SIZE),
            }
            while True:
                page = self._get(params)
                for symbol, bars in (page.get("bars") or {}).items():
                    rows += [bar | {"symbol": symbol} for bar in bars]
                if not page.get("next_page_token"):
                    break
                params = {**params, "page_token": page["next_page_token"]}
        if not rows:
            return pl.DataFrame(schema=DAILY_SCHEMA)
        return pl.DataFrame(rows, infer_schema_length=None).select(
            pl.col("t")
            .str.to_datetime(time_zone="UTC")
            .dt.convert_time_zone(NEW_YORK)
            .dt.date()
            .alias("date"),
            pl.col("symbol").alias("ticker"),
            pl.col("o").cast(pl.Float64).alias("open"),
            pl.col("h").cast(pl.Float64).alias("high"),
            pl.col("l").cast(pl.Float64).alias("low"),
            pl.col("c").cast(pl.Float64).alias("close"),
            pl.col("v").cast(pl.Int64).alias("volume"),
        )

    def corporate_actions(
        self, symbols: Sequence[str], first: date, last: date
    ) -> list[dict]:
        """The splits, dividends and spin-offs of `symbols` that Alpaca processed from
        `first` to `last`, both included, as Alpaca sends them.

        Each comes with its `kind`, Alpaca's name for its list (such as
        "cash_dividends"), and its `symbol`; a spin-off's is the company spun
        from. Alpaca lists an action by the day it processed it, which for a
        dividend is the day it is paid, weeks after its ex-date.
        """
        actions = []
        for start in range(0, len(symbols), SYMBOLS_A_REQUEST):
            chunk = symbols[start : start + SYMBOLS_A_REQUEST]
            params = {
                "symbols": ",".join(chunk),
                "types": ",".join(ACTION_TYPES),
                "start": first.isoformat(),
                "end": last.isoformat(),
                "limit": str(ACTIONS_PAGE_SIZE),
            }
            while True:
                page = self._get(params, ACTIONS_URL)
                for kind, listed in (page.get("corporate_actions") or {}).items():
                    for action in listed:
                        symbol = action.get("symbol") or action.get("source_symbol")
                        # Asked about a company, Alpaca also lists the spin-offs
                        # that created it.
                        if symbol in chunk:
                            actions.append(action | {"kind": kind, "symbol": symbol})
                if not page.get("next_page_token"):
                    break
                params = {**params, "page_token": page["next_page_token"]}
        return actions
