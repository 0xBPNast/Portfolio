# scripts/alpha_audit.py
import os
import numpy as np
import pandas as pd

from engine.metrics import summary_stats


def block_bootstrap_sharpe(r: np.ndarray, blocks: int = 48, n_boot: int = 500, periods_per_year: int = 24 * 365):
    """
    Block bootstrap to preserve autocorrelation.
    r: array of per-bar returns
    blocks: block length in bars (e.g., 48 = 2 days for 1h bars)
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < blocks * 5:
        return (np.nan, np.nan, np.nan)

    sharpe_samples = []
    n_blocks = int(np.ceil(n / blocks))
    for _ in range(n_boot):
        starts = np.random.randint(0, max(1, n - blocks + 1), size=n_blocks)
        sample = np.concatenate([r[s:s + blocks] for s in starts])[:n]
        mu = sample.mean()
        sd = sample.std(ddof=0) + 1e-12
        sh = (mu / sd) * np.sqrt(periods_per_year)
        sharpe_samples.append(sh)

    sharpe_samples = np.sort(np.array(sharpe_samples))
    return (float(np.percentile(sharpe_samples, 5)),
            float(np.percentile(sharpe_samples, 50)),
            float(np.percentile(sharpe_samples, 95)))


def cost_stress(oos: pd.DataFrame, extra_cost_bps: float) -> pd.Series:
    """
    Adds an extra cost penalty proportional to turnover:
      turnover_t = abs(pos_t - pos_{t-1})
      penalty_t  = turnover_t * extra_cost_bps / 1e4

    Note: This is an approximation (useful for robustness checking).
    """
    if "position" not in oos.columns:
        raise ValueError("oos must contain 'position' column for turnover-based cost stress.")

    pos = oos["position"].fillna(0.0).astype(float)
    turnover = pos.diff().abs().fillna(0.0)

    penalty = turnover * (extra_cost_bps / 10000.0)
    r_stressed = oos["strat_r"].fillna(0.0).astype(float) - penalty
    return r_stressed.rename(f"strat_r_plus_{extra_cost_bps:.1f}bps")


def main():
    # Default paths produced by your walkforward script
    oos_path = os.environ.get("OOS_CSV", "data/processed/walkforward_oos.csv")
    if not os.path.exists(oos_path):
        raise FileNotFoundError(f"Cannot find {oos_path}. Set OOS_CSV env var or run walkforward first.")

    oos = pd.read_csv(oos_path)
    # Try to restore datetime index if present
    if "timestamp" in oos.columns:
        oos["timestamp"] = pd.to_datetime(oos["timestamp"])
        oos = oos.set_index("timestamp")
    elif oos.columns[0].lower() in ("date", "datetime", "time"):
        oos[oos.columns[0]] = pd.to_datetime(oos[oos.columns[0]])
        oos = oos.set_index(oos.columns[0])
    else:
        # If file was saved with index, pandas usually loads an "Unnamed: 0"
        if "Unnamed: 0" in oos.columns:
            oos["Unnamed: 0"] = pd.to_datetime(oos["Unnamed: 0"])
            oos = oos.set_index("Unnamed: 0")

    required = {"strat_r", "wf_equity"}
    missing = required - set(oos.columns)
    if missing:
        raise ValueError(f"OOS file missing columns: {missing}. Found: {list(oos.columns)}")

    # Basic stats
    periods_per_year = 24 * 365
    stats = summary_stats(oos, periods_per_year=periods_per_year)

    print("\n=== ALPHA AUDIT: BASE (NET OF BACKTEST COSTS) ===")
    print({k: float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))})

    # PnL concentration by WF iteration
    if "iter_id" in oos.columns:
        by_iter = (
            oos.groupby("iter_id")["strat_r"]
            .apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
            .rename("iter_total_return")
            .to_frame()
        )
        by_iter["abs_contrib"] = by_iter["iter_total_return"].abs()
        by_iter["share_abs_contrib"] = by_iter["abs_contrib"] / (by_iter["abs_contrib"].sum() + 1e-12)

        print("\n=== PnL CONCENTRATION (by walk-forward iter) ===")
        print(by_iter.sort_values("share_abs_contrib", ascending=False).head(10))

        top3 = float(by_iter.sort_values("share_abs_contrib", ascending=False)["share_abs_contrib"].head(3).sum())
        print(f"\nTop-3 window share of absolute return contribution: {top3:.2%}")
        if top3 > 0.65:
            print("WARNING: Returns are highly concentrated. Edge may be regime-luck or overfit.")

    # Cost stress test (extra cost on turnover)
    if "position" in oos.columns:
        print("\n=== COST STRESS TEST (extra bps per unit turnover) ===")
        for extra in [2.0, 5.0, 10.0, 20.0]:
            r2 = cost_stress(oos, extra_cost_bps=extra)
            tmp = oos.copy()
            tmp["strat_r"] = r2
            tmp["wf_equity_stressed"] = (1.0 + tmp["strat_r"].fillna(0.0)).cumprod() * float(oos["wf_equity"].iloc[0])
            st = summary_stats(tmp.rename(columns={"wf_equity_stressed": "wf_equity"}), periods_per_year=periods_per_year)
            print(f" +{extra:>4.1f} bps | sharpe={st['sharpe']:.3f} | mdd={st['max_drawdown']:.3f} | total_return={st['total_return']:.3f}")

    # Bootstrap Sharpe confidence interval
    r = oos["strat_r"].fillna(0.0).values
    lo, med, hi = block_bootstrap_sharpe(r, blocks=48, n_boot=600, periods_per_year=periods_per_year)
    print("\n=== SHARPE BOOTSTRAP (block=48 bars) ===")
    print({"sharpe_p5": lo, "sharpe_p50": med, "sharpe_p95": hi})
    if np.isfinite(lo) and lo < 0.0:
        print("WARNING: Sharpe CI includes < 0. Edge might be weak / not robust.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
