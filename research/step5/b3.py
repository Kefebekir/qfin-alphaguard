import json
import math

import numpy as np
import polars as pl

with open("data/raw/treasury/auctions_10-Year.json") as f:
    d = json.load(f)
days = sorted(
    {
        x["auctionDate"][:10]
        for x in d
        if (x["securityType"], x["originalSecurityTerm"])
        in (("Note", "10-Year"), ("Bond", "30-Year"))
        and x.get("tips") == "No"
        and x.get("floatingRate") == "No"
    }
)
print(len(days), "10- and 30-year nominal auctions", days[0], "..", days[-1])
tlt = (
    pl.read_parquet("data/raw/etf_daily/TLT.parquet")
    .filter(pl.col("date") >= pl.date(2016, 1, 1))
    .sort("date")
)
dates = tlt["date"].to_list()
close = tlt["close"].to_numpy()
index = {d: i for i, d in enumerate(dates)}
from datetime import date

diffs = []
for s in days:
    a = date.fromisoformat(s)
    i = index.get(a)
    if i is None or i < 6 or i + 5 >= len(close):
        continue
    before = (
        close[i - 1] / close[i - 6] - 1
    )  # the 5 sessions up to the day before the auction
    after = close[i + 5] / close[i] - 1  # from the auction day's close, 5 sessions on
    diffs.append(after - before)
v = np.array(diffs)
t = v.mean() / v.std(ddof=1) * math.sqrt(len(v))
print(
    f"B3: TLT, after minus before: {v.mean() * 1e4:+.1f} bps per auction over {len(v)} auctions, t {t:+.2f} -> {'PASS' if t >= 2.64 else 'fail'}"
)
