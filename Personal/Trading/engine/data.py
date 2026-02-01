import ccxt
import pandas as pd
from datetime import datetime, timezone

def fetch_ohlcv_binance(
    symbol="BTC/USDT",
    timeframe="1h",
    since_ms=None,
    limit=1000,
    max_batches=200
):
    ex = ccxt.binance({"enableRateLimit": True})
    all_rows = []
    for _ in range(max_batches):
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + 1  # move forward
        if len(batch) < limit:
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["ts"])
    return df

def save_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def utc_now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)