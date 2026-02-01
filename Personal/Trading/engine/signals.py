import pandas as pd
import numpy as np

def moving_average_signal(df: pd.DataFrame, fast=50, slow=200) -> pd.Series:
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    sig = (fast_ma > slow_ma).astype(int)  # 1 = long, 0 = flat
    return sig.rename("signal")

def realized_vol(df: pd.DataFrame, vol_lookback=48) -> pd.Series:
    # log returns
    r = np.log(df["close"]).diff()
    # realized vol per bar (stdev of returns)
    vol = r.rolling(vol_lookback).std()
    return vol.rename("rv")

def vol_target_position(signal: pd.Series, rv: pd.Series, target_vol=0.01, max_leverage=3.0) -> pd.Series:
    # target_vol is per-bar (e.g., per hour if timeframe=1h). Tune as needed.
    # leverage ~ target_vol / realized_vol
    lev = (target_vol / rv).clip(0, max_leverage)
    pos = signal * lev
    return pos.rename("position")
