"""Load price data: EODHD's history continued by Alpaca, or the synthetic generator."""

import re
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

import polars as pl

from qfin_alphaguard.config import Config
from qfin_alphaguard.data.alpaca import AlpacaClient, api_keys
from qfin_alphaguard.data.eodhd import BARS_SCHEMA, EodhdClient, eodhd_code
from qfin_alphaguard.data.store import read_prices
from qfin_alphaguard.data.synthetic import generate_prices
from qfin_alphaguard.data.universe import IndexHistory, Membership, sp500
from qfin_alphaguard.events import Dividend, Split
from qfin_alphaguard.sessions import session_days, sessions

# Scaling prices for splits and dividends can leave a close a hair above the
# high: about 1e-16 relative, from floating-point rounding. Gaps up to this
# size are rounding; anything larger is a real data error and is left for
# validation.
ROUNDING_TOLERANCE = 1e-9

# The S&P 500 ETF the backtest report measures the plan against (step 10). A
# UK investor would hold a UCITS ETF of the same index; SPY's history is the
# longest.
BENCHMARKS = ("SPY",)

# A split shows up as a jump in `adjustment` (10x for a 10-for-1 split). Split
# histories are fetched only for codes whose adjustment jumps by more than
# this from one day to the next; a 5-for-4 split is a 1.25 jump, while
# ordinary dividends move it by a few percent at most.
SPLIT_JUMP = 1.15

# EODHD's daily bars, as downloaded on 11 November 2026, the last day of the
# subscription: every S&P 500 member since 2015 and SPY, to 10 November. The
# file is never written again; Alpaca's bars continue it (decision 0022).
EODHD_HISTORY = "prices_eodhd.parquet"

# Alpaca sends a session's bars once they are 15 minutes old (decision 0008).
DELAY = timedelta(minutes=16)

# Alpaca lists a corporate action by the day it processed it: for a dividend,
# the day it is paid, a few weeks after its ex-date.
PROCESSED_WITHIN = timedelta(days=90)

# EODHD's name for a ticker's earlier company, which no longer trades.
EARLIER_COMPANY = re.compile(r"_old\d*$")

PRICES = ("open", "high", "low", "close")
COLUMNS = (*BARS_SCHEMA, "split_factor", "sp500")


def load_prices(config: Config) -> pl.DataFrame:
    """Daily bars with prices and volume adjusted for splits and dividends.

    Columns: date, ticker, open, high, low, close, volume, adjustment,
    split_factor, sp500. The price as traded that day is the adjusted price
    divided by `adjustment`; the shares traded are `volume / split_factor`.
    `sp500` says whether the stock was in the S&P 500 that day; the benchmark
    ETFs in BENCHMARKS never are. The synthetic data has no benchmark.

    The real data is EODHD's history in EODHD_HISTORY, continued by Alpaca's
    bars to `config.end_date` (continue_prices).
    """
    if config.synthetic:
        return generate_prices(config)
    return continue_prices(
        read_prices(config.raw_dir / EODHD_HISTORY),
        AlpacaClient(*api_keys()),
        sp500(),
        date.fromisoformat(config.end_date),
        datetime.now(UTC),
    )


def continue_prices(
    stored: pl.DataFrame,
    client: AlpacaClient,
    history: IndexHistory,
    last: date,
    now: datetime,
) -> pl.DataFrame:
    """`stored` continued with Alpaca's bars to `last`, or to the last session over.

    Every ticker with a bar on the stored last day continues, unless EODHD
    marks it as an earlier company that no longer trades; so does every
    member since then, and a member with nothing stored comes with its whole
    history. A session counts once it has been closed for DELAY.
    """
    end = now - DELAY
    final = stored["date"].max()
    if min(last, end.date()) <= final:
        return stored
    over = [
        session.day
        for session in sessions(final, min(last, end.date()))
        if session.close <= end
    ]
    if over[-1] <= final:
        return stored
    until = over[-1]

    links = stored.group_by("ticker").agg(pl.col("date").max().alias("link"))
    still = links.filter(
        (pl.col("link") == final)
        & ~pl.col("ticker").str.contains(EARLIER_COMPANY.pattern)
    )["ticker"]
    members = {
        eodhd_code(t) for t in history.tickers_between(final + timedelta(days=1), until)
    }
    known = set(links["ticker"])
    going_on = sorted(set(still) | (members & known))
    joining = sorted(members - known)

    first_new = links.filter(pl.col("ticker").is_in(going_on))[
        "link"
    ].min() + timedelta(days=1)
    first_stored = stored["date"].min()
    new = client.daily_bars([alpaca_symbol(t) for t in going_on], first_new, end)
    actions = client.corporate_actions(
        [alpaca_symbol(t) for t in going_on], first_new, until + PROCESSED_WITHIN
    )
    if joining:
        joined = [alpaca_symbol(t) for t in joining]
        new = pl.concat([new, client.daily_bars(joined, first_stored, end)])
        actions += client.corporate_actions(
            joined, first_stored, until + PROCESSED_WITHIN
        )
    new = new.with_columns(_stored_name(pl.col("ticker")))
    new = new.join(links, on="ticker", how="left").filter(
        (pl.col("link").is_null() | (pl.col("date") > pl.col("link")))
        & (pl.col("date") <= until)
    )
    events = _events(client, actions, end, until)
    return continue_with(stored, new.drop("link"), events, history)


def continue_with(
    stored: pl.DataFrame,
    new: pl.DataFrame,
    events: list[Split | Dividend],
    history: IndexHistory,
) -> pl.DataFrame:
    """`stored`, EODHD's bars, continued with `new`, Alpaca's bars as traded.

    A ticker's new bars follow its last stored day; a ticker with none stored
    has all its bars from `new`. The splits and dividends of `events` that fall
    after a ticker's last stored day, and by its last bar, are applied to its
    bars before them, as EODHD applied the earlier ones:
    - a split of ratio r divides the prices by r and multiplies the volume and
      `split_factor` by r;
    - a dividend D multiplies the prices by 1 - D / P, where P is the traded
      close of the session before its ex-date.
    A spin-off comes as a split, as EODHD stored GE's of April 2024. So a
    ticker's stored returns stay as they were, and corporate.corporate_actions
    finds each event of `events` again, at its day and size. A ticker's stored
    bars are first brought to factors of 1 on its last stored day, as EODHD
    sends them.

    The new bars are marked `sp500` by the spells of `history`.
    """
    tickers = new["ticker"].unique()
    kept = stored.filter(~pl.col("ticker").is_in(tickers.implode())).select(COLUMNS)
    links = (
        stored.filter(pl.col("ticker").is_in(tickers.implode()))
        .group_by("ticker")
        .agg(pl.col("date").max().alias("link"))
    )
    fresh = [
        bars.with_columns(
            pl.lit(1.0).alias("adjustment"),
            pl.lit(1.0).alias("split_factor"),
            _member_of(history, ticker).alias("sp500"),
        )
        for (ticker,), bars in new.partition_by("ticker", as_dict=True).items()
    ]
    # EODHD adjusts a ticker's bars for the events up to its last one, so its
    # factors there are 1; were they not, the events after it would count twice.
    at_link = pl.col("adjustment").last().over("ticker")
    split_at_link = pl.col("split_factor").last().over("ticker")
    going_on = (
        stored.filter(pl.col("ticker").is_in(tickers.implode()))
        .select(COLUMNS)
        .sort("ticker", "date")
        .with_columns(
            *(pl.col(c) / at_link for c in PRICES),
            pl.col("adjustment") / at_link,
            pl.col("split_factor") / split_at_link,
            (pl.col("volume") / split_at_link).round().cast(pl.Int64),
        )
    )
    rows = pl.concat([going_on, *(bars.select(COLUMNS) for bars in fresh)])
    later = _later_factors(rows, events, links)
    rows = (
        rows.with_columns((pl.col("date") + timedelta(days=1)).alias("next_day"))
        .sort("next_day")
        .join_asof(
            later,
            left_on="next_day",
            right_on="day",
            by="ticker",
            check_sortedness=False,  # sorted by the key above
            strategy="forward",
        )
        .with_columns(pl.col("split").fill_null(1.0), pl.col("dividend").fill_null(1.0))
        .with_columns((pl.col("dividend") / pl.col("split")).alias("scale"))
        .with_columns(
            *(pl.col(c) * pl.col("scale") for c in PRICES),
            (pl.col("adjustment") * pl.col("scale")),
            (pl.col("split_factor") * pl.col("split")),
            (pl.col("volume") * pl.col("split")).round().cast(pl.Int64),
        )
        .select(COLUMNS)
    )
    prices = keep_sessions(pl.concat([kept, rows]))
    return snap_rounding(prices).sort(["ticker", "date"])


def alpaca_symbol(ticker: str) -> str:
    """Alpaca writes a class share with a dot, as the index lists do (BRK.B)."""
    return ticker.replace("-", ".")


def _stored_name(symbol: pl.Expr) -> pl.Expr:
    """The stored ticker of an Alpaca symbol: a class share with a dash, as eodhd_code."""
    return symbol.str.replace_all(".", "-", literal=True)


def _member_of(history: IndexHistory, ticker: str) -> pl.Expr:
    spans = history.spans_of(alpaca_symbol(ticker))
    if not spans:
        return pl.lit(False)
    return pl.any_horizontal([_inside(span) for span in spans])


def _later_factors(
    rows: pl.DataFrame, events: list[Split | Dividend], links: pl.DataFrame
) -> pl.DataFrame:
    """For each ticker and event day, the split ratio and the dividend factor of
    all its events on that day or after: (ticker, day, split, dividend)."""
    schema = {
        "ticker": pl.String,
        "day": pl.Date,
        "split": pl.Float64,
        "cash": pl.Float64,
    }
    listed = pl.DataFrame(
        [
            {
                "ticker": e.ticker,
                "day": e.day,
                "split": e.ratio if isinstance(e, Split) else 1.0,
                "cash": e.amount if isinstance(e, Dividend) else 0.0,
            }
            for e in events
        ],
        schema=schema,
    )
    lasts = rows.group_by("ticker").agg(pl.col("date").max().alias("final"))
    listed = (
        listed.join(lasts, on="ticker")
        .join(links, on="ticker", how="left")
        .filter(
            (pl.col("link").is_null() | (pl.col("day") > pl.col("link")))
            & (pl.col("day") <= pl.col("final"))
        )
        .with_columns((pl.col("day") - timedelta(days=1)).alias("eve"))
        .sort("eve")
        # the traded close of the session before the ex-date
        .join_asof(
            rows.select(
                "ticker",
                "date",
                (pl.col("close") / pl.col("adjustment")).alias("before"),
            ).sort("date"),
            left_on="eve",
            right_on="date",
            by="ticker",
            check_sortedness=False,  # sorted by the key above
            strategy="backward",
        )
        .filter(pl.col("before").is_not_null())
        .with_columns((1 - pl.col("cash") / pl.col("before")).alias("dividend"))
    )
    if (listed["dividend"] <= 0).any():
        raise ValueError("a dividend as large as the price before it")
    return (
        listed.group_by("ticker", "day")
        .agg(pl.col("split").product(), pl.col("dividend").product())
        .sort("ticker", pl.col("day"), descending=[False, True])
        .with_columns(
            pl.col("split").cum_prod().over("ticker"),
            pl.col("dividend").cum_prod().over("ticker"),
        )
        .sort("day")
    )


def _events(
    client: AlpacaClient, actions: list[dict], end: datetime, until: date
) -> list[Split | Dividend]:
    """Alpaca's corporate actions as splits and dividends, by `until`.

    A stock dividend of rate x is a split of 1 + x. A spin-off is a split whose
    ratio is the fall in Alpaca's spin-off adjustment across its ex-date, from
    its bars adjusted for spin-offs alone; one Alpaca does not adjust for is left
    out.
    """
    events: list[Split | Dividend] = []
    spun = []
    for action in actions:
        ticker = eodhd_code(action["symbol"])
        day = date.fromisoformat(action["ex_date"])
        if day > until:
            continue
        match action["kind"]:
            case "forward_splits" | "reverse_splits":
                events.append(
                    Split(ticker, day, action["new_rate"] / action["old_rate"])
                )
            case "stock_dividends":
                events.append(Split(ticker, day, 1 + action["rate"]))
            case "cash_dividends" if action["rate"] > 0:
                events.append(Dividend(ticker, day, action["rate"]))
            case "spin_offs":
                spun.append((ticker, day))
    if spun:
        symbols = sorted({alpaca_symbol(t) for t, _ in spun})
        first = min(day for _, day in spun) - timedelta(days=10)
        raw = client.daily_bars(symbols, first, end)
        alone = client.daily_bars(symbols, first, end, adjustment="spin-off")
        factor = raw.join(alone, on=["ticker", "date"], suffix="_spun").select(
            _stored_name(pl.col("ticker")),
            "date",
            (pl.col("close_spun") / pl.col("close")).alias("factor"),
        )
        for ticker, day in spun:
            around = factor.filter(pl.col("ticker") == ticker).sort("date")
            after = around.filter(pl.col("date") == day)["factor"]
            before = around.filter(pl.col("date") < day)["factor"]
            if after.is_empty() or before.is_empty():
                raise ValueError(f"no bars around {ticker}'s spin-off on {day}")
            # Alpaca lists spin-offs of warrants, such as GameStop's in October
            # 2025, and leaves the prices as they were.
            if after[0] != before[-1]:
                events.append(Split(ticker, day, after[0] / before[-1]))
    return events


def download_sp500(
    client: EodhdClient, history: IndexHistory, first: date, last: date
) -> pl.DataFrame:
    """Daily bars for every stock in the index on any day from `first` to `last`.

    One ticker can stand for different companies over time, and EODHD keeps
    the earlier ones as TICKER_old, TICKER_old1 and so on. For each stretch of
    membership, the code with the most bars inside the stretch is taken, and
    its bars on those days are marked `sp500`.
    """
    old_codes = client.old_codes()
    frames = []
    for ticker in history.tickers_between(first, last):
        code = eodhd_code(ticker)
        bars: dict[str, pl.DataFrame] = {}  # downloaded once per code
        stretches: dict[str, list[Membership]] = defaultdict(list)
        for span in history.spans_of(ticker):
            if not span.overlaps(first, last):
                continue
            best, most = None, 0
            for candidate in (code, *old_codes.get(code, ())):
                if candidate not in bars:
                    bars[candidate] = client.daily_bars(candidate, first, last)
                inside = bars[candidate].filter(_inside(span)).height
                if inside > most:
                    best, most = candidate, inside
            if best is not None:
                stretches[best].append(span)
        for candidate, spans in stretches.items():
            member = pl.any_horizontal([_inside(span) for span in spans])
            with_splits = _with_split_factor(client, candidate, bars[candidate])
            frames.append(with_splits.with_columns(member.alias("sp500")))

    if not frames:
        return pl.DataFrame(
            schema={**BARS_SCHEMA, "split_factor": pl.Float64, "sp500": pl.Boolean}
        )
    prices = keep_sessions(pl.concat(frames))
    return snap_rounding(prices.sort(["ticker", "date"]))


def download_benchmarks(
    client: EodhdClient, codes: tuple[str, ...], first: date, last: date
) -> pl.DataFrame:
    """Daily bars for ETFs such as SPY, in the members' columns, never `sp500`."""
    frames = [
        _with_split_factor(client, code, client.daily_bars(code, first, last))
        for code in codes
    ]
    if not frames:
        return pl.DataFrame(
            schema={**BARS_SCHEMA, "split_factor": pl.Float64, "sp500": pl.Boolean}
        )
    prices = pl.concat(frames).with_columns(pl.lit(False).alias("sp500"))
    return snap_rounding(keep_sessions(prices).sort(["ticker", "date"]))


def keep_sessions(prices: pl.DataFrame) -> pl.DataFrame:
    """Drop bars on days the exchange was closed.

    EODHD has a few: by its data, Nordstrom traded on New Year's Day 2025 and
    on 9 January 2025, when the exchange closed for President Carter's funeral.
    """
    if prices.is_empty():
        return prices
    days = pl.Series(session_days(prices["date"].min(), prices["date"].max()))
    return prices.filter(pl.col("date").is_in(days.implode()))


def sp500_coverage(prices: pl.DataFrame, history: IndexHistory) -> float:
    """Share of index member-days, on the dates in `prices`, that have a bar.

    A drop shows missing data, or index changes after the membership history's
    `as_of` date that the file does not know about yet.
    """
    days = prices["date"].unique()
    expected = sum(len(history.members_on(day)) for day in days)
    return prices.filter(pl.col("sp500")).height / expected if expected else 0.0


def snap_rounding(df: pl.DataFrame) -> pl.DataFrame:
    """Move high and low onto open or close where only rounding puts them outside."""
    body_high = pl.max_horizontal("open", "close")
    body_low = pl.min_horizontal("open", "close")
    high_is_rounding = (pl.col("high") < body_high) & (
        body_high - pl.col("high") <= ROUNDING_TOLERANCE * body_high
    )
    low_is_rounding = (pl.col("low") > body_low) & (
        pl.col("low") - body_low <= ROUNDING_TOLERANCE * body_low
    )
    return df.with_columns(
        pl.when(high_is_rounding)
        .then(body_high)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(low_is_rounding).then(body_low).otherwise(pl.col("low")).alias("low"),
    )


def repair_broken_bars(prices: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Bars whose open or close lies outside low..high, widened to cover them.

    EODHD sometimes sends a bar whose high and low are wrong while its open and
    close are right: LUMN's of 12 October 2026 came as open 5.25, high and low
    5.05, close 5.05, where the SIP minutes say 5.25, 5.28, 4.99 and 5.055
    (decision 0016). The open and close carry the returns and are kept; the
    high and low are moved out to them. Dropping the bar instead would leave
    the stock a day short and out of the plans for a year.

    Returns the bars, repaired, and the broken ones as they came.
    """
    broken = (pl.col("low") > pl.min_horizontal("open", "close")) | (
        pl.col("high") < pl.max_horizontal("open", "close")
    )
    repaired = prices.with_columns(
        pl.max_horizontal("open", "high", "low", "close").alias("high"),
        pl.min_horizontal("open", "high", "low", "close").alias("low"),
    )
    return repaired, prices.filter(broken)


def _with_split_factor(
    client: EodhdClient, code: str, bars: pl.DataFrame
) -> pl.DataFrame:
    """Add `split_factor`: how many of today's shares one share of that day became.

    It is 10 for Netflix before its 10-for-1 split in November 2025 and 1 after,
    so dividing the split-adjusted volume by it gives the shares traded that day.
    """
    bars = bars.sort("date")
    step = pl.col("adjustment") / pl.col("adjustment").shift(1)
    jumps = bars.select(((step > SPLIT_JUMP) | (step < 1 / SPLIT_JUMP)).any()).item()
    factor = pl.lit(1.0)
    if jumps:
        for day, ratio in client.splits(code):
            factor = factor * pl.when(pl.col("date") < day).then(ratio).otherwise(1.0)
    return bars.with_columns(factor.alias("split_factor"))


def _inside(span: Membership) -> pl.Expr:
    inside = pl.col("date") >= span.start
    if span.end is not None:
        inside = inside & (pl.col("date") < span.end)
    return inside
