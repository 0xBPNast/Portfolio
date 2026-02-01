import pandas as pd
import numpy as np

def backtest_single_asset(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps=4.0,          # 0.04% default-ish; tune to your venue/tier
    slippage_bps=2.0,     # conservative friction
    initial_equity=1_000_000.0
):
    df = df.copy()
    pos = position.reindex(df.index).fillna(0.0)

    # Returns
    r = df["close"].pct_change().fillna(0.0)

    # Turnover cost model: cost on change in position
    dpos = pos.diff().abs().fillna(0.0)
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    costs = dpos * cost_rate

    # Strategy return: position * asset return - costs
    strat_r = (pos.shift(1).fillna(0.0) * r) - costs

    equity = (1.0 + strat_r).cumprod() * initial_equity

    out = pd.DataFrame({
        "close": df["close"],
        "position": pos,
        "asset_r": r,
        "costs": costs,
        "strat_r": strat_r,
        "equity": equity
    })
    return out
