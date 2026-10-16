import json
import os
import urllib.parse
import urllib.request

import polars as pl

headers = {
    "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
}
rows, token = [], None
while True:
    q = {
        "symbols": "BTC/USD",
        "timeframe": "1Day",
        "start": "2015-01-01T00:00:00Z",
        "end": "2026-10-10T00:00:00Z",
        "limit": "10000",
    }
    if token:
        q["page_token"] = token
    req = urllib.request.Request(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars?"
        + urllib.parse.urlencode(q),
        headers=headers,
    )
    data = json.load(urllib.request.urlopen(req, timeout=60))
    rows += data["bars"].get("BTC/USD", [])
    token = data.get("next_page_token")
    if not token:
        break
d = (
    pl.DataFrame(rows)
    .select(
        pl.col("t").str.to_datetime(time_zone="UTC").alias("time"),
        pl.col("o", "h", "l", "c", "v"),
    )
    .sort("time")
)
d.write_parquet("data/raw/crypto/btc_usd_daily.parquet")
print(d.height, d["time"].min(), d["time"].max())
