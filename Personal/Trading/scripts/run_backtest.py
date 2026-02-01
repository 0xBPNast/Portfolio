import matplotlib.pyplot as plt
from engine.data import load_parquet
from engine.signals import moving_average_signal, realized_vol, vol_target_position
from engine.backtest import backtest_single_asset
from engine.metrics import summary_stats

df = load_parquet("data/raw/BTCUSDT_1h.parquet")

sig = moving_average_signal(df, fast=50, slow=200)
rv  = realized_vol(df, vol_lookback=48)
pos = vol_target_position(sig, rv, target_vol=0.008, max_leverage=2.0)

bt = backtest_single_asset(df, pos, fee_bps=4.0, slippage_bps=2.0, initial_equity=1_000_000.0)

bt.to_csv("data/processed/backtest_results.csv", index=True)
print("Saved to data/processed/backtest_results.csv")

stats = summary_stats(bt, periods_per_year=24*365)  # 1h bars
print(stats)

bt["equity"].plot(title="Equity Curve")
plt.show()
