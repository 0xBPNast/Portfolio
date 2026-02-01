import numpy as np
import pandas as pd
from engine.features import FeatureBuilder, FeatureSet

class CandleFeatureBuilder(FeatureBuilder):
    def __init__(self, ret_lb=24, vol_lb=48, trend_lb=200, z_lb=24*30):
        self.ret_lb = ret_lb
        self.vol_lb = vol_lb
        self.trend_lb = trend_lb
        self.z_lb = z_lb

    def build(self, market_data: pd.DataFrame) -> FeatureSet:
        df = market_data.copy()
        r = np.log(df["close"]).diff()

        feats = pd.DataFrame(index=df.index)

        # Returns
        feats["ret_1"] = r
        feats["ret_lb"] = r.rolling(self.ret_lb, min_periods=self.ret_lb).sum()

        # Realized volatility
        feats["rv"] = r.rolling(self.vol_lb, min_periods=self.vol_lb).std()

        # Volatility z-score (30d by default)
        rv_mean = feats["rv"].rolling(self.z_lb, min_periods=self.z_lb).mean()
        rv_std  = feats["rv"].rolling(self.z_lb, min_periods=self.z_lb).std()
        feats["rv_z"] = (feats["rv"] - rv_mean) / (rv_std + 1e-12)

        # Trend proxy (MA ratio spread)
        ma_fast = df["close"].rolling(50, min_periods=50).mean()
        ma_slow = df["close"].rolling(self.trend_lb, min_periods=self.trend_lb).mean()
        feats["trend"] = (ma_fast / (ma_slow + 1e-12) - 1.0)

        # chop proxy on returns (stable)
        # gross = sum(|r|), net = |sum(r)|
        gross = r.abs().rolling(self.ret_lb, min_periods=self.ret_lb).sum()
        net = r.rolling(self.ret_lb, min_periods=self.ret_lb).sum().abs()

        chop_raw = gross / (net + 1e-12)

        # log transform compresses huge values
        feats["chop"] = np.log1p(chop_raw)

        # optional: cap extreme outliers (recommended)
        cap = feats["chop"].quantile(0.995)
        feats["chop"] = feats["chop"].clip(upper=cap)
        feats = feats.replace([np.inf, -np.inf], np.nan)

        return FeatureSet(feats)
