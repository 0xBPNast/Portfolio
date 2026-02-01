import os
import numpy as np
import pandas as pd

from engine.data import load_parquet
from engine.signals import moving_average_signal, realized_vol, vol_target_position
from engine.backtest import backtest_single_asset
from engine.metrics import summary_stats
from engine.features_candles import CandleFeatureBuilder
from engine.regime import RegimeEngine


# ----------------------------
# Globals (instantiate once)
# ----------------------------
FB = CandleFeatureBuilder()

# CandleFeatureBuilder uses a 30-day rolling z-score window by default
FEATURE_NORM_LB = 24 * 30


# ----------------------------
# Helpers
# ----------------------------
def run_strategy(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    vol_lookback: int,
    target_vol: float,
    max_leverage: float,
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float,
    re: RegimeEngine,                    # TRAIN-fitted regime engine (no leakage)
    fb: CandleFeatureBuilder = FB        # feature builder (default global)
) -> pd.DataFrame:
    """
    Full pipeline on the provided df (train or context/test):
      - MA signal
      - realized vol
      - candle features
      - regime risk multiplier (using TRAIN-fitted RegimeEngine)
      - vol-targeted position * regime multiplier
      - backtest
    """
    sig = moving_average_signal(df, fast=fast, slow=slow)
    rv = realized_vol(df, vol_lookback=vol_lookback)

    features = fb.build(df).df
    reg = re.score(features)

    pos = vol_target_position(sig, rv, target_vol=target_vol, max_leverage=max_leverage)
    pos = pos * reg["risk_multiplier"].reindex(pos.index).fillna(0.0)

    bt = backtest_single_asset(
        df=df,
        position=pos,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity
    )

    # Attach useful debug columns for later analysis
    bt["signal"] = sig.reindex(bt.index).fillna(0.0)
    bt["rv"] = rv.reindex(bt.index)
    bt["risk_multiplier"] = reg["risk_multiplier"].reindex(bt.index)
    bt["regime"] = reg["regime"].reindex(bt.index)

    if "shock_active" in reg.columns:
        bt["shock_active"] = reg["shock_active"].reindex(bt.index).fillna(0.0)

    return bt


def pick_best_params(
    train_df: pd.DataFrame,
    param_grid,
    vol_lookback: int,
    target_vol: float,
    max_leverage: float,
    fee_bps: float,
    slippage_bps: float
):
    """
    Grid search on TRAIN only.
    Score = Sharpe, tie-breaker = smaller max drawdown (closer to 0).
    Regime thresholds are fit on TRAIN only once (no leakage).
    """
    # Fit regime thresholds on TRAIN ONLY
    re_train = RegimeEngine().fit(FB.build(train_df).df)

    best = None
    best_score = -np.inf

    for fast, slow in param_grid:
        if fast >= slow:
            continue

        bt_train = run_strategy(
            df=train_df,
            fast=fast,
            slow=slow,
            vol_lookback=vol_lookback,
            target_vol=target_vol,
            max_leverage=max_leverage,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_equity=1_000_000.0,
            re=re_train
        )

        stats = summary_stats(bt_train, periods_per_year=24 * 365)
        score = stats["sharpe"]

        if (score > best_score) or (
            np.isclose(score, best_score)
            and stats["max_drawdown"] > (best["stats"]["max_drawdown"] if best else -1e9)
        ):
            best_score = score
            best = {"fast": fast, "slow": slow, "stats": stats}

    return best


def walk_forward(
    df: pd.DataFrame,
    train_bars: int = 24 * 90,   # 90 days of 1h bars
    test_bars: int = 24 * 30,    # 30 days of 1h bars
    step_bars: int = 24 * 30,    # advance by 30 days each iteration
    optimize: bool = True,
    param_grid=None,
    vol_lookback: int = 48,
    target_vol: float = 0.008,
    max_leverage: float = 2.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    initial_equity: float = 1_000_000.0,
    # Diagnostics tuning for continuous position series
    pos_eps: float = 1e-4,   # "invested" threshold
    dpos_eps: float = 1e-3   # "meaningful position change" threshold
):
    if param_grid is None:
        param_grid = [(20, 80), (30, 120), (40, 160), (50, 200), (60, 240)]

    df = df.dropna().copy()
    n = len(df)

    oos_chunks = []
    selections = []

    start = 0
    iter_id = 0

    while True:
        train_start = start
        train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars

        if test_end > n:
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        # Fit regime thresholds on TRAIN ONLY for this iteration (no leakage)
        re_train = RegimeEngine().fit(FB.build(train_df).df)

        # Select parameters on TRAIN only
        if optimize:
            # Ensure the param picking uses the same train-only-fitted regime engine
            best = None
            best_score = -np.inf

            for fast, slow in param_grid:
                if fast >= slow:
                    continue

                bt_train = run_strategy(
                    df=train_df,
                    fast=fast,
                    slow=slow,
                    vol_lookback=vol_lookback,
                    target_vol=target_vol,
                    max_leverage=max_leverage,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    initial_equity=1_000_000.0,
                    re=re_train
                )

                stats = summary_stats(bt_train, periods_per_year=24 * 365)
                score = stats["sharpe"]

                if (score > best_score) or (
                    np.isclose(score, best_score)
                    and stats["max_drawdown"] > (best["stats"]["max_drawdown"] if best else -1e9)
                ):
                    best_score = score
                    best = {"fast": fast, "slow": slow, "stats": stats}

            fast, slow = best["fast"], best["slow"]
            train_pick_stats = best["stats"]

        else:
            fast, slow = param_grid[0]
            bt_train_tmp = run_strategy(
                df=train_df,
                fast=fast,
                slow=slow,
                vol_lookback=vol_lookback,
                target_vol=target_vol,
                max_leverage=max_leverage,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                initial_equity=1_000_000.0,
                re=re_train
            )
            train_pick_stats = summary_stats(bt_train_tmp, periods_per_year=24 * 365)

        # --- Warmup/context so rolling indicators & feature stats work ---
        warmup = max(slow, vol_lookback, FEATURE_NORM_LB) + 5
        ctx_start = max(0, test_start - warmup)
        ctx_df = df.iloc[ctx_start:test_end]

        bt_ctx = run_strategy(
            df=ctx_df,
            fast=fast,
            slow=slow,
            vol_lookback=vol_lookback,
            target_vol=target_vol,
            max_leverage=max_leverage,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_equity=initial_equity,
            re=re_train
        )

        # Keep only the true out-of-sample slice
        bt_oos = bt_ctx.loc[test_df.index[0]:test_df.index[-1]].copy()

        # Add iteration metadata
        bt_oos["iter_id"] = iter_id
        bt_oos["fast"] = fast
        bt_oos["slow"] = slow

        # Regime distribution + average risk multiplier
        reg_counts = bt_oos["regime"].value_counts(normalize=True).to_dict()
        avg_rm = float(bt_oos["risk_multiplier"].mean())
        print(f"iter {iter_id} regime_frac={reg_counts} avg_rm={avg_rm:.2f}")

        if "shock_active" in bt_oos.columns:
            shock_frac = float((bt_oos["shock_active"] > 0).mean())
            print(f"iter {iter_id} shock_frac={shock_frac:.3f}")

        # Diagnostics suitable for continuous positions
        invested_frac = float((bt_oos["position"].abs() > pos_eps).mean())
        position_changes = int((bt_oos["position"].diff().abs() > dpos_eps).sum())
        turnover = float(bt_oos["position"].diff().abs().fillna(0.0).sum())
        print(
            f"iter {iter_id} | fast={fast} slow={slow} | "
            f"invested_frac={invested_frac:.3f} | changes>{dpos_eps}={position_changes} | turnover={turnover:.2f}"
        )

        selections.append({
            "iter_id": iter_id,
            "train_start": str(train_df.index[0]),
            "train_end": str(train_df.index[-1]),
            "test_start": str(test_df.index[0]),
            "test_end": str(test_df.index[-1]),
            "fast": fast,
            "slow": slow,
            "train_sharpe": float(train_pick_stats["sharpe"]),
            "train_mdd": float(train_pick_stats["max_drawdown"]),
        })

        oos_chunks.append(bt_oos)

        iter_id += 1
        start += step_bars

    if not oos_chunks:
        raise ValueError("Not enough data for the chosen walk-forward window sizes.")

    oos = pd.concat(oos_chunks).sort_index()

    # Chain OOS equity into one continuous curve
    oos["wf_equity"] = (1.0 + oos["strat_r"].fillna(0.0)).cumprod() * initial_equity

    selections_df = pd.DataFrame(selections)
    return oos, selections_df


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    df = load_parquet("data/raw/BTCUSDT_1h.parquet")

    oos, picks = walk_forward(
        df=df,
        train_bars=24 * 90,   # 90 days
        test_bars=24 * 30,    # 30 days
        step_bars=24 * 30,    # 30 days roll forward
        optimize=True,
        param_grid=[(20, 80), (30, 120), (40, 160), (50, 200), (60, 240)],
        vol_lookback=48,
        target_vol=0.008,
        max_leverage=2.0,
        fee_bps=4.0,
        slippage_bps=2.0,
        initial_equity=1_000_000.0,
        pos_eps=1e-4,
        dpos_eps=1e-3
    )

    wf_stats = {
        "oos_total_return": float(oos["wf_equity"].iloc[-1] / oos["wf_equity"].iloc[0] - 1.0),
        "oos_max_drawdown": float((oos["wf_equity"] / oos["wf_equity"].cummax() - 1.0).min()),
        "oos_sharpe": float(
            (oos["strat_r"].mean() / (oos["strat_r"].std(ddof=0) + 1e-12)) * np.sqrt(24 * 365)
        ),
    }

    print("\nWALK-FORWARD OOS STATS:")
    print(wf_stats)

    print("\nPARAM PICKS PER ITERATION:")
    print(picks.head(10))

    os.makedirs("data/processed", exist_ok=True)
    oos.to_csv("data/processed/walkforward_oos.csv")
    picks.to_csv("data/processed/walkforward_param_picks.csv", index=False)
    print("\nSaved:")
    print(" - data/processed/walkforward_oos.csv")
    print(" - data/processed/walkforward_param_picks.csv")

    try:
        import matplotlib.pyplot as plt
        oos["wf_equity"].plot(title="Walk-Forward OOS Equity (Chained)")
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)
