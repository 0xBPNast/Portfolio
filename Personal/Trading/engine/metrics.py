import pandas as pd
import numpy as np

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    mu = returns.mean()
    sd = returns.std(ddof=0)
    if sd == 0:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))

def summary_stats(bt: pd.DataFrame, periods_per_year: float):
    rets = bt["strat_r"]
    eq = bt["equity"]
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    mdd = max_drawdown(eq)
    sh = sharpe(rets, periods_per_year)
    return {
        "total_return": total_return,
        "max_drawdown": mdd,
        "sharpe": sh,
        "turnover_avg": float(bt["position"].diff().abs().mean())
    }
