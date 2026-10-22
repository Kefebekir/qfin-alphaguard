"""N6's context: what a member's prices, the market and the calendar said before a session.

For each S&P 500 member on each session `day` from 2016, every number is known by the close of
the session before `day` (N6's first data uses nothing of the day itself), except the
calendar's, which is published ahead. The closes are the stored ones, adjusted for splits and
dividends, and so are the highs, lows and opens.

The member:
- r1, r5, r20, r60: its return over the last 1, 5, 20 and 60 sessions;
- mom: from 252 sessions back to 21 back (a year's momentum without the last month);
- vol20, vol60: the standard deviation of its daily returns over 20 and 60 sessions;
- range: the last session's high less its low, over its close; gap: that session's open over
  the close before it, less one;
- dollars: the log of its mean traded value over 20 sessions (the price as traded times the
  shares, as data/ingest.py defines them); turnover: the last session's over that mean, logged;
- high52: its close over its highest high of 252 sessions (at least 200), less one;
- beta: the slope of its daily returns on SPY's over 60 sessions; corr60: their correlation;
  resid5: r5 less beta times SPY's r5;
- sector: of the nine SPDR sector funds with prices since 1998 (FUNDS), the one its daily
  returns correlated with most over 252 sessions (at least 200), as 0 to 8; sector_r1 and
  sector_r5, that fund's returns; rel5: r5 less sector_r5.
The market, the same for every member: SPY's r1, r5, r20 and vol20; VIX's close and the log
of its change over 5 sessions; VIX9D over VIX, VIX over VIX3M, and VVIX's close: each Cboe
series' last close by the session before. And how much the members moved apart: dispersion,
the standard deviation of the members' last daily returns across them, and dispersion20, its
mean over 20 sessions; comove60, the members' mean corr60 (in 2026 it fell to its lowest:
members moved with SPY far less than before, which Alpaca's own closes confirm).
The calendar: the weekday (0 Monday), and whether the day has an FOMC decision, a CPI or a
payrolls release (events.csv, which lists them as scheduled; emergency meetings left out).

Output: data/news_reader/n6/context.parquet, one row a member-day. A member with too little
history has nulls where a window is not full.

Checked on the real data, printed: for three days, every price, fund close and Cboe close from
that day on is scrambled and the table built again; every number of that day and the days
before must be unchanged, and the next session's must change. Then a few members' sectors.

Usage: uv run python research/news_reader/n6_context.py
"""

from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import polars as pl

from qfin_alphaguard.sessions import sessions

PRICES = Path("data/raw/prices.parquet")
ETFS = Path("data/raw/etf_daily")
CBOE = Path("data/raw/cboe")
EVENTS = Path("data/raw/calendar/events.csv")
OUT = Path("data/news_reader/n6/context.parquet")
FUNDS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
VIX = ("VIX", "VIX9D", "VIX3M", "VVIX")
EVENT_KINDS = ("fomc", "cpi", "nfp")
FIRST = date(2016, 1, 1)
YEAR, ENOUGH = 252, 200
MEMBER = (
    "r1",
    "r5",
    "r20",
    "r60",
    "mom",
    "vol20",
    "vol60",
    "range",
    "gap",
    "dollars",
    "turnover",
    "high52",
    "beta",
    "corr60",
    "resid5",
    "sector",
    "sector_r1",
    "sector_r5",
    "rel5",
)
MARKET = (
    "spy_r1",
    "spy_r5",
    "spy_r20",
    "spy_vol20",
    "vix",
    "vix_change5",
    "vix9d_ratio",
    "vix3m_ratio",
    "vvix",
    "dispersion",
    "dispersion20",
    "comove60",
)
CALENDAR = ("weekday", "fomc", "cpi", "nfp")


def _back(n: int) -> pl.Expr:
    return pl.col("close").shift(n).over("ticker")


def member_at_close(prices: pl.DataFrame, funds: pl.DataFrame) -> pl.DataFrame:
    """Each member's numbers as of a session's close, keyed by that session (`date`).
    `funds` holds date and the daily returns of SPY and each of FUNDS, by name."""
    traded = (pl.col("close") / pl.col("adjustment")) * (
        pl.col("volume") / pl.col("split_factor")
    )
    p = (
        prices.sort("ticker", "date")
        .with_columns(
            r=pl.col("close") / _back(1) - 1,
            traded=traded,
        )
        .join(funds, on="date", how="left")
    )
    mean_traded = pl.col("traded").rolling_mean(20).over("ticker")
    p = p.with_columns(
        r1=pl.col("r"),
        r5=pl.col("close") / _back(5) - 1,
        r20=pl.col("close") / _back(20) - 1,
        r60=pl.col("close") / _back(60) - 1,
        mom=_back(21) / _back(252) - 1,
        vol20=pl.col("r").rolling_std(20).over("ticker"),
        vol60=pl.col("r").rolling_std(60).over("ticker"),
        range=(pl.col("high") - pl.col("low")) / pl.col("close"),
        gap=pl.col("open") / _back(1) - 1,
        dollars=mean_traded.log(),
        turnover=(pl.col("traded") / mean_traded).log(),
        high52=pl.col("close")
        / pl.col("high").rolling_max(YEAR, min_samples=ENOUGH).over("ticker")
        - 1,
        beta=pl.rolling_cov("r", "SPY", window_size=60).over("ticker")
        / pl.col("SPY").rolling_var(60).over("ticker"),
        corr60=pl.rolling_corr("r", "SPY", window_size=60).over("ticker"),
        **{
            f"corr_{fund}": pl.rolling_corr(
                "r", fund, window_size=YEAR, min_samples=ENOUGH
            ).over("ticker")
            for fund in FUNDS
        },
    )
    corrs = pl.concat_list([pl.col(f"corr_{f}").fill_nan(None) for f in FUNDS])
    p = p.with_columns(
        sector=pl.when(corrs.list.drop_nulls().list.len() == len(FUNDS)).then(
            corrs.list.arg_max()
        )
    )
    pick = pl.concat_list([pl.col(f) for f in FUNDS])
    pick5 = pl.concat_list([pl.col(f"{f}_r5") for f in FUNDS])
    p = p.with_columns(
        sector_r1=pick.list.get(pl.col("sector"), null_on_oob=True),
        sector_r5=pick5.list.get(pl.col("sector"), null_on_oob=True),
        resid5=pl.col("r5") - pl.col("beta") * pl.col("SPY_r5"),
    ).with_columns(rel5=pl.col("r5") - pl.col("sector_r5"))
    return p.select("date", "ticker", "sp500", *MEMBER)


def fund_returns(etfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """date, then each fund's (and SPY's) daily return by its name, and its return over 5
    sessions as <name>_r5."""
    out = None
    for name, frame in etfs.items():
        r = (
            frame.sort("date")
            .select(
                "date",
                (pl.col("close") / pl.col("close").shift(1) - 1).alias(name),
                (pl.col("close") / pl.col("close").shift(5) - 1).alias(f"{name}_r5"),
            )
            .drop_nulls(name)
        )
        out = r if out is None else out.join(r, on="date", how="full", coalesce=True)
    return out.sort("date")


def market_at_close(spy: pl.DataFrame, vix: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The market's numbers as of a session's close, keyed by SPY's sessions (`date`)."""
    s = spy.sort("date").with_columns(r=pl.col("close") / pl.col("close").shift(1) - 1)
    s = s.select(
        "date",
        spy_r1=pl.col("r"),
        spy_r5=pl.col("close") / pl.col("close").shift(5) - 1,
        spy_r20=pl.col("close") / pl.col("close").shift(20) - 1,
        spy_vol20=pl.col("r").rolling_std(20),
    )
    for name, frame in vix.items():
        s = s.join_asof(
            frame.sort("date").select("date", pl.col("close").alias(name)),
            on="date",
            strategy="backward",
        )
    vix5 = s.select("date", "VIX").with_columns(pl.col("VIX").shift(5).alias("vix5"))
    return s.join(vix5.drop("VIX"), on="date").select(
        "date",
        "spy_r1",
        "spy_r5",
        "spy_r20",
        "spy_vol20",
        vix=pl.col("VIX"),
        vix_change5=(pl.col("VIX") / pl.col("vix5")).log(),
        vix9d_ratio=pl.col("VIX9D") / pl.col("VIX"),
        vix3m_ratio=pl.col("VIX") / pl.col("VIX3M"),
        vvix=pl.col("VVIX"),
    )


def context(
    prices: pl.DataFrame,
    etfs: dict[str, pl.DataFrame],
    vix: dict[str, pl.DataFrame],
    events: pl.DataFrame,
    first: date = FIRST,
) -> pl.DataFrame:
    """One row a member-day from `first`: the member's and the market's numbers at the close
    of the session before, and the day's calendar."""
    last = prices["date"].max()
    days = [s.day for s in sessions(prices["date"].min(), last + timedelta(days=10))]
    following = dict(pairwise(days))
    member = member_at_close(prices, fund_returns(etfs))
    market = market_at_close(etfs["SPY"], vix)
    before = pl.col("date").replace_strict(
        following, default=None, return_dtype=pl.Date
    )
    breadth = (
        member.filter(pl.col("sp500"))
        .group_by("date")
        .agg(
            dispersion=pl.col("r1").std(),
            comove60=pl.col("corr60").fill_nan(None).mean(),
        )
        .sort("date")
        .with_columns(dispersion20=pl.col("dispersion").rolling_mean(20))
    )
    market = market.join(breadth, on="date", how="left")
    member = member.with_columns(day=before).drop("date", "sp500")
    market = market.with_columns(day=before).drop("date")
    flags = events.filter(pl.col("event").is_in(EVENT_KINDS))
    members = prices.filter(pl.col("sp500") & (pl.col("date") >= first)).select(
        pl.col("date").alias("day"), "ticker"
    )
    out = (
        members.join(member, on=["day", "ticker"], how="left")
        .join(market, on="day", how="left")
        .with_columns(
            weekday=pl.col("day").dt.weekday() - 1,
            **{
                kind: pl.col("day").is_in(
                    flags.filter(pl.col("event") == kind)["day"].to_list()
                )
                for kind in EVENT_KINDS
            },
        )
    )
    return out.select("day", "ticker", *MEMBER, *MARKET, *CALENDAR).sort(
        "day", "ticker"
    )


def scrambled(frame: pl.DataFrame, start: date, seed: int) -> pl.DataFrame:
    """`frame` with every price and volume from `start` on multiplied by noise."""
    columns = [
        c for c in ("open", "high", "low", "close", "volume") if c in frame.columns
    ]
    noise = pl.Series(
        __import__("numpy").random.default_rng(seed).uniform(0.5, 1.5, frame.height)
    )
    later = pl.col("date") >= start
    return frame.with_columns(
        [
            pl.when(later)
            .then(pl.col(c) * noise)
            .otherwise(pl.col(c))
            .cast(frame.schema[c])
            for c in columns
        ]
    )


def unchanged_before(
    prices, etfs, vix, events, table: pl.DataFrame, start: date, first: date = FIRST
) -> tuple[bool, bool]:
    """Whether the table to `start` survives scrambling everything from `start` on, and
    whether the session after `start` does not."""
    again = context(
        scrambled(prices, start, 1),
        {k: scrambled(v, start, 2 + i) for i, (k, v) in enumerate(etfs.items())},
        {k: scrambled(v, start, 20 + i) for i, (k, v) in enumerate(vix.items())},
        events,
        first,
    )
    to = pl.col("day") <= start
    after = pl.col("day") == table.filter(pl.col("day") > start)["day"].min()
    kept = table.filter(to).equals(again.filter(to), null_equal=True)
    moved = not table.filter(after).equals(again.filter(after), null_equal=True)
    return kept, moved


def read_cboe(path: Path) -> pl.DataFrame:
    """A Cboe history file: date and its last column, the close."""
    raw = pl.read_csv(path)
    return raw.select(
        pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y").alias("date"),
        pl.col(raw.columns[-1]).cast(pl.Float64).alias("close"),
    )


def main() -> None:
    prices = pl.read_parquet(PRICES)
    etfs = {name: pl.read_parquet(ETFS / f"{name}.parquet") for name in ("SPY", *FUNDS)}
    vix = {name: read_cboe(CBOE / f"{name}_History.csv") for name in VIX}
    events = pl.read_csv(EVENTS, try_parse_dates=True)
    table = context(prices, etfs, vix, events)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(OUT)
    print(
        f"{OUT}: {table.height:,} member-days, {table['day'].min()} to {table['day'].max()}"
    )
    nulls = table.select(pl.all().exclude("day", "ticker").null_count() / table.height)
    print("share null:", {k: round(v, 4) for k, v in nulls.row(0, named=True).items()})
    for start in (date(2018, 3, 1), date(2021, 6, 1), date(2026, 10, 1)):
        kept, moved = unchanged_before(prices, etfs, vix, events, table, start)
        print(
            f"everything from {start} on scrambled: the days to it unchanged {kept}, "
            f"the session after changed {moved}"
        )
    day = date(2026, 10, 1)
    some = table.filter(
        (pl.col("day") == day) & pl.col("ticker").is_in(["AAPL", "XOM", "JPM", "NEE"])
    ).sort("ticker")
    print(
        f"sectors on {day}:",
        {t: FUNDS[s] for t, s in zip(some["ticker"], some["sector"], strict=True)},
        f"median beta {table.filter(pl.col('day') == day)['beta'].median():.2f}",
    )
    years = (
        table.group_by(pl.col("day").dt.year().alias("year"))
        .agg(
            pl.col("comove60").mean(),
            pl.col("dispersion20").mean(),
            pl.col("beta").median().alias("median beta"),
        )
        .sort("year")
    )
    print("by year:")
    for year, comove, dispersion, beta in years.iter_rows():
        print(
            f"  {year}: comove60 {comove:.3f}, dispersion20 {dispersion:.4f}, "
            f"median beta {beta:.2f}"
        )


if __name__ == "__main__":
    main()
