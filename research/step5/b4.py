import contextlib
import io
import math
import runpy

import polars as pl

with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("research/step5/calendar_effects.py")
qqq = ns["daily"]("QQQ").select("day", "intraday")
w = pl.read_parquet("data/raw/weather/lga_hourly.parquet").with_columns(
    pl.col("time").dt.convert_time_zone("America/New_York")
)
w = w.filter(pl.col("time").dt.hour().is_between(6, 9)).drop_nulls("cldc")
morning = w.group_by(pl.col("time").dt.date().alias("day")).agg(
    pl.col("cldc").cast(pl.Float64).mean().alias("cloud"), pl.len().alias("hours")
)
j = qqq.join(morning, on="day").sort("day")
x, y = j["cloud"].to_numpy(), j["intraday"].to_numpy()
xc = x - x.mean()
b = xc @ (y - y.mean()) / (xc @ xc)
e = y - y.mean() - b * xc
t = b / math.sqrt(e @ e / (len(y) - 2) / (xc @ xc))
print(
    f"B4: QQQ open-to-close on morning cloud cover (oktas, 06-10 NY, LaGuardia): slope {b * 1e4:+.2f} bps per okta, t {t:+.2f} (n {len(y)}) -> {'PASS' if t <= -2.64 else 'fail'}"
)
clear, overcast = y[x <= 2], y[x >= 6]
print(
    f"   clear mornings (<= 2 oktas) {clear.mean() * 1e4:+.2f} bps ({len(clear)}), overcast (>= 6) {overcast.mean() * 1e4:+.2f} bps ({len(overcast)})"
)
