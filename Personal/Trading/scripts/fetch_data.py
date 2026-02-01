from engine.data import fetch_ohlcv_binance, save_parquet, utc_now_ms

# ~2 years back in ms (approx)
since_ms = utc_now_ms() - int(2 * 365 * 24 * 60 * 60 * 1000)

df = fetch_ohlcv_binance(
    symbol="BTC/USDT",
    timeframe="1h",
    since_ms=since_ms,
    limit=1000,
    max_batches=1000
)

save_parquet(df, "data/raw/BTCUSDT_1h.parquet")
print(len(df), df.index.min(), df.index.max())
