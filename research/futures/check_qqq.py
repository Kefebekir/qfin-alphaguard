"""F1's implementation check: session_rule.py on QQQ must reproduce qqq_momentum.py.

QQQ's stored minutes and US sessions, exiting at the last price as qqq_momentum.py does,
against its stored daily results (data/intraday_pilot/qqq_days.parquet): the gross return of
a 1x position each day and the days on which the rule traded. F1 asks for a correlation of at
least 0.99 and the same trading days within 1%.

Usage: check_qqq.py ROOT QQQ_DAYS_PARQUET
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from session_rule import Session, simulate

from qfin_alphaguard.data.intraday import intraday_path
from qfin_alphaguard.sessions import sessions

ROOT, REFERENCE = Path(sys.argv[1]), sys.argv[2]
calendar = {s.day: s for s in sessions(date(2016, 1, 1), date(2026, 12, 31))}
bars = (
    pl.concat(
        [
            pl.read_parquet(intraday_path(ROOT, "QQQ", y)).select(
                "start", "open", "high", "low", "close", "volume"
            )
            for y in range(2016, 2027)
            if intraday_path(ROOT, "QQQ", y).exists()
        ]
    )
    .with_columns(pl.col("start").dt.date().alias("day"))
    .sort("start")
)
days = []
for (day,), g in bars.group_by("day", maintain_order=True):
    s = calendar.get(day)
    if s is None:
        continue
    minute = (g["start"] - s.open).dt.total_minutes().to_numpy()
    keep = minute >= 0
    length = int(minute[keep].max()) + 1  # as qqq_momentum.py: to the last bar
    days.append(
        Session(
            day=day,
            minute=minute[keep].astype(int),
            open=g["open"].to_numpy()[keep],
            high=g["high"].to_numpy()[keep],
            low=g["low"].to_numpy()[keep],
            close=g["close"].to_numpy()[keep],
            volume=g["volume"].to_numpy()[keep].astype(float),
            length=length,
            decisions=list(range(30, length, 30)),
        )
    )

ours = pl.DataFrame(
    [
        {"day": d.day, "gross": d.gross, "traded": d.traded, "size": d.size}
        for d in simulate(days, exit_at="last_price")
    ]
)
ref = pl.read_parquet(REFERENCE).select(
    "day",
    pl.col("gross_1x").alias("ref_gross"),
    pl.col("traded_1x").alias("ref_traded"),
)
both = ours.join(ref, on="day", how="inner")
corr = float(np.corrcoef(both["gross"], both["ref_gross"])[0, 1])
ours_days, ref_days = (
    int((both["traded"] > 0).sum()),
    int((both["ref_traded"] > 0).sum()),
)
print(
    f"{both.height} days in both ({ours.height} ours, {ref.height} reference); "
    f"correlation of the daily gross return {corr:.5f}"
)
print(
    f"trading days: ours {ours_days}, reference {ref_days} "
    f"({abs(ours_days - ref_days) / ref_days:.2%} apart); "
    f"days with a different result: {int((np.abs(both['gross'] - both['ref_gross']) > 1e-9).sum())}"
)
passed = corr >= 0.99 and abs(ours_days - ref_days) <= 0.01 * ref_days
print("check passed" if passed else "CHECK FAILED")
