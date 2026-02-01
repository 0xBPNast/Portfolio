# scripts/paper_live.py
import os
import json
import pandas as pd
import numpy as np

from engine.data import load_parquet
from engine.signals import moving_average_signal, realized_vol, vol_target_position
from engine.backtest import backtest_single_asset  # used here only for consistent accounting if you want
from engine.features_candles import CandleFeatureBuilder
from engine.regime import RegimeEngine
from engine.portfolio import GuardrailConfig, PortfolioGuardrails


# --- Singletons (avoid re-instantiating) ---
FB = CandleFeatureBuilder()
RE = RegimeEngine()


def compute_target_position(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    vol_lookback: int,
    target_vol: float,
    max_leverage: float,
) -> pd.Series:
    sig = moving_average_signal(df, fast=fast, slow=slow)
    rv = realized_vol(df, vol_lookback=vol_lookback)

    feats = FB.build(df).df
    reg = RE.score(feats)

    pos = vol_target_position(sig, rv, target_vol=target_vol, max_leverage=max_leverage)
    pos = pos * reg["risk_multiplier"].reindex(pos.index).fillna(0.0)

    # Return the raw “strategy target” series, before portfolio guardrails
    return pos.rename("target_pos")


def ensure_dirs():
    os.makedirs("data/live", exist_ok=True)


def main():
    ensure_dirs()

    # ---- Config (start simple; later pull from YAML) ----
    symbol = "BTCUSDT"
    timeframe = "1h"
    parquet_path = f"data/raw/{symbol}_{timeframe}.parquet"

    fast, slow = 30, 120
    vol_lookback = 48
    target_vol = 0.008
    max_leverage = 2.0

    # Guardrails state persistence
    guard_state_path = "data/live/guard_state.json"
    cfg = GuardrailConfig(
        max_leverage=max_leverage,
        rebalance_deadband=0.02,
        smooth_alpha=0.35,
        max_pos_change_per_bar=0.25,
        dd_half_risk=0.10,
        dd_stop=0.20,
        daily_loss_limit=0.03,
        resume_dd=0.12,
    )
    guards = PortfolioGuardrails.load_state(guard_state_path, default_cfg=cfg)

    # ---- Load market data ----
    df = load_parquet(parquet_path).dropna().copy()
    if len(df) < max(slow, vol_lookback, 24 * 30) + 10:
        raise ValueError("Not enough bars yet to run paper-live safely. Fetch more history first.")

    # ---- Compute target & latest decision ----
    target = compute_target_position(df, fast, slow, vol_lookback, target_vol, max_leverage)
    ts = target.index[-1]
    target_now = float(target.iloc[-1])

    # ---- Paper equity tracking ----
    # For paper-live, we track equity from the log file (so it persists across runs).
    log_path = "data/live/paper_log.csv"
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        last_equity = float(log["equity"].iloc[-1])
        last_pos = float(log["final_pos"].iloc[-1])
    else:
        last_equity = 1_000_000.0
        last_pos = 0.0

    # Approximate “one-bar paper PnL” using close-to-close return (simple).
    # You can upgrade to next-open execution later.
    close = df["close"].astype(float)
    r = float((close.iloc[-1] / close.iloc[-2]) - 1.0)
    equity_now = last_equity * (1.0 + last_pos * r)

    # ---- Apply guardrails to produce final position ----
    final_pos, dbg = guards.apply(ts=pd.Timestamp(ts), equity=equity_now, target_pos=target_now)

    # Persist guard state
    guards.save_state(guard_state_path)

    # ---- Log ----
    row = {
        "timestamp": str(ts),
        "close": float(close.iloc[-1]),
        "ret": float(r),
        "equity": float(equity_now),
        "target_pos": float(target_now),
        "final_pos": float(final_pos),
        "halted": int(dbg["halted"]),
        "drawdown": float(dbg["drawdown"]),
        "daily_loss": float(dbg["daily_loss"]),
        "risk_scalar": float(dbg["risk_scalar"]),
    }

    # Append
    if os.path.exists(log_path):
        out = pd.read_csv(log_path)
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(log_path, index=False)

    # ---- Status JSON for phone monitoring / dashboards ----
    status = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": str(ts),
        "close": row["close"],
        "equity": row["equity"],
        "target_pos": row["target_pos"],
        "final_pos": row["final_pos"],
        "halted": bool(row["halted"]),
        "drawdown": row["drawdown"],
        "daily_loss": row["daily_loss"],
        "risk_scalar": row["risk_scalar"],
        "guard_state_path": guard_state_path,
        "log_path": log_path,
    }
    with open("data/live/status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print("PAPER-LIVE OK:", status)


if __name__ == "__main__":
    main()
