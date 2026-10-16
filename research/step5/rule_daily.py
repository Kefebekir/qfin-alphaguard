"""The QQQ noise-area rule's daily result after costs, from policy_data.py's decisions.

Shared by the exploration scripts. Costs: QQQ's measured half spread (0.3 bps) and the SEC
fee on half the turnover; the close of the day's last position is charged to it.
"""

import polars as pl

COST = 0.3e-4 + 0.0000206 / 2


def rule_daily(
    decisions: str = "data/intraday_pilot/policy/decisions.parquet",
    ticker="QQQ",
    cost=COST,
):
    d = pl.read_parquet(decisions).filter(pl.col("ticker") == ticker).sort("day", "t")
    d = d.with_columns(pl.col("rule").shift(1).over("day").fill_null(0).alias("before"))
    last = pl.col("t") == pl.col("t").max().over("day")
    turn = (pl.col("rule") - pl.col("before")).abs() + pl.when(last).then(
        pl.col("rule").abs()
    ).otherwise(0)
    d = d.with_columns(
        (
            pl.col("size") * pl.col("rule") * pl.col("ret")
            - pl.col("size") * cost * turn
        ).alias("net")
    )
    return d.group_by("day").agg(pl.col("net").sum().alias("pnl")).sort("day")
