"""Exploration, after the pre-registered G1/G2 failed (results/05_real_gamma.txt): the rule on days
whose GEX rank is in its top third against the other days, 2017-2022. This led to the
validation of preregistrations/06_real_gamma_validation.txt."""

import math
import sys

import polars as pl

sys.path.insert(0, "research/step5")
from rule_daily import rule_daily

rule = rule_daily()
for t in ("QQQ", "SPY"):
    g = pl.read_parquet(f"data/step5/gamma_real/gex_{t}.parquet")
    j = (
        rule.join(g, on="day")
        .sort("day")
        .filter(
            pl.col("rank").is_not_nan() & pl.col("day").dt.year().is_between(2017, 2022)
        )
    )
    top = j.filter(pl.col("rank") >= 2 / 3)["pnl"].to_numpy()
    rest = j.filter(pl.col("rank") < 2 / 3)["pnl"].to_numpy()
    t_stat = (top.mean() - rest.mean()) / math.sqrt(
        top.var(ddof=1) / len(top) + rest.var(ddof=1) / len(rest)
    )
    print(
        f"{t}: top third {top.mean() * 1e4:+.2f} bps ({len(top)} days) vs rest {rest.mean() * 1e4:+.2f} bps; t {t_stat:+.2f}"
    )
