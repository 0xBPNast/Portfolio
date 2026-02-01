import numpy as np
import pandas as pd
from typing import Optional


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class RegimeEngine:
    """
    Conservative regime + risk overlay (Python 3.9 compatible).

    Outputs:
      - regime (label)
      - risk_multiplier (continuous in ~[rm_floor, 1.0], clipped)
      - diagnostic flags

    Conservative profile:
      - Lower baseline in CHOP
      - Stronger, continuous de-risking as rv_z rises
      - Optional chop penalty (continuous)
      - Small, capped trend boost
    """

    def __init__(
        self,
        # Thresholds (can be set directly OR learned via fit()).
        trend_th: Optional[float] = None,
        chop_th: Optional[float] = None,
        stress_rv_z: Optional[float] = None,

        # Percentiles used in fit()
        trend_q: float = 0.75,      # stricter trend
        chop_q: float = 0.50,       # define "chop_ok" around median
        stress_q: float = 0.93,     # more frequent stress detection than 0.95

        # Base multipliers by regime (conservative)
        rm_trend: float = 0.95,
        rm_neutral: float = 0.75,
        rm_chop: float = 0.45,
        rm_stress_base: float = 0.20,

        # Floors / caps
        rm_floor: float = 0.10,
        rm_cap: float = 1.00,

        # Continuous stress de-risk curve (uses rv_z)
        # multiplier_stress ~ exp(-k*(rv_z - stress_th)), clipped.
        stress_k: float = 0.90,     # higher => stronger de-risk
        stress_clip_low: float = 0.12,
        stress_clip_high: float = 1.00,

        # Continuous chop penalty (uses chop)
        # penalty increases when chop > chop_th; penalty is in [chop_penalty_low, 1]
        use_chop_penalty: bool = True,
        chop_slope: float = 1.25,          # bigger => faster penalty past chop_th
        chop_penalty_low: float = 0.55,    # minimum multiplier from chop penalty

        # Trend boost (small, capped)
        use_trend_boost: bool = True,
        trend_boost_max: float = 1.10,     # allow up to +10% boost
        trend_boost_scale: float = 6.0,    # controls how quickly boost ramps

        # Shock brake (requires feats to include ret_1 and rv)
        use_shock_brake: bool = True,
        shock_th: float = 4.5,
        shock_rm: float = 0.60,
        shock_cooldown_bars: int = 6,


        # Optional smoothing to reduce flicker
        smooth_window: int = 3,
        min_periods: int = 1
    ):
        self.trend_th = trend_th
        self.chop_th = chop_th
        self.stress_rv_z = stress_rv_z

        self.trend_q = trend_q
        self.chop_q = chop_q
        self.stress_q = stress_q

        self.rm_trend = rm_trend
        self.rm_neutral = rm_neutral
        self.rm_chop = rm_chop
        self.rm_stress_base = rm_stress_base

        self.rm_floor = rm_floor
        self.rm_cap = rm_cap

        self.stress_k = stress_k
        self.stress_clip_low = stress_clip_low
        self.stress_clip_high = stress_clip_high

        self.use_chop_penalty = use_chop_penalty
        self.chop_slope = chop_slope
        self.chop_penalty_low = chop_penalty_low

        self.use_trend_boost = use_trend_boost
        self.trend_boost_max = trend_boost_max
        self.trend_boost_scale = trend_boost_scale

        self.smooth_window = smooth_window
        self.min_periods = min_periods

        self.use_shock_brake = use_shock_brake
        self.shock_th = shock_th
        self.shock_rm = shock_rm
        self.shock_cooldown_bars = shock_cooldown_bars


    def fit(self, feats: pd.DataFrame) -> "RegimeEngine":
        """
        Fit thresholds using percentiles from TRAIN features.
        """
        f = feats[["trend", "chop", "rv_z"]].dropna()
        if len(f) < 200:
            # fallback defaults if needed
            if self.trend_th is None:
                self.trend_th = 0.001
            if self.chop_th is None:
                self.chop_th = 2.5
            if self.stress_rv_z is None:
                self.stress_rv_z = 1.5
            return self

        if self.trend_th is None:
            self.trend_th = float(f["trend"].quantile(self.trend_q))
        if self.chop_th is None:
            self.chop_th = float(f["chop"].quantile(self.chop_q))
        if self.stress_rv_z is None:
            self.stress_rv_z = float(f["rv_z"].quantile(self.stress_q))

        return self

    def _maybe_smooth(self, feats: pd.DataFrame) -> pd.DataFrame:
        if self.smooth_window is None or self.smooth_window <= 1:
            return feats

        cols = ["trend", "chop", "rv_z"]
        sm = feats[cols].rolling(self.smooth_window, min_periods=self.min_periods).mean()
        out = feats.copy()
        out[cols] = sm
        return out

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        """
        Score regimes + build a conservative continuous risk multiplier.
        Best practice: call fit(train_feats) then score(ctx/test_feats).
        """
        if self.trend_th is None or self.chop_th is None or self.stress_rv_z is None:
            self.fit(feats)

        f = self._maybe_smooth(feats)

        out = pd.DataFrame(index=f.index)

        trend_ok = f["trend"] > self.trend_th
        chop_ok = f["chop"] < self.chop_th
        stress = f["rv_z"] > self.stress_rv_z

        # Labels
        out["regime"] = np.where(
            stress,
            "STRESS",
            np.where(trend_ok & chop_ok, "TREND", np.where(chop_ok, "NEUTRAL", "CHOP"))
        )

        # Base multiplier from regime
        rm = np.full(len(out), self.rm_neutral, dtype=float)
        rm = np.where(out["regime"] == "TREND", self.rm_trend, rm)
        rm = np.where(out["regime"] == "CHOP", self.rm_chop, rm)
        rm = np.where(out["regime"] == "STRESS", self.rm_stress_base, rm)

        # --- Continuous stress de-risk (works in all regimes, strongest in STRESS) ---
        # If rv_z is below stress threshold, factor ~= 1.
        # If above, factor decays exponentially.
        rvz = f["rv_z"].to_numpy(dtype=float)
        stress_excess = np.maximum(0.0, rvz - float(self.stress_rv_z))
        stress_factor = np.exp(-self.stress_k * stress_excess)
        stress_factor = np.clip(stress_factor, self.stress_clip_low, self.stress_clip_high)

        # Apply more aggressively if regime is STRESS, but still apply globally
        is_stress = (out["regime"].to_numpy() == "STRESS")
        # In STRESS: square it (stronger cut). Else: mild cut.
        stress_factor_adj = np.where(is_stress, stress_factor ** 2, stress_factor)
        rm = rm * stress_factor_adj

        # --- Continuous chop penalty (only penalize when chop > chop_th) ---
        if self.use_chop_penalty:
            chop = f["chop"].to_numpy(dtype=float)
            # normalize how far above threshold we are
            excess = (chop - float(self.chop_th))
            excess = np.maximum(0.0, excess)
            # map excess to [low, 1] using sigmoid
            # bigger excess => lower penalty
            penalty = 1.0 - (1.0 - self.chop_penalty_low) * _sigmoid(self.chop_slope * excess)
            rm = rm * penalty

        # --- Small trend boost (only in TREND to avoid over-risking) ---
        if self.use_trend_boost:
            trend = f["trend"].to_numpy(dtype=float)
            # measure "how far above trend_th"
            t_excess = np.maximum(0.0, trend - float(self.trend_th))
            # boost ramps up smoothly and caps at trend_boost_max
            boost = 1.0 + (self.trend_boost_max - 1.0) * (1.0 - np.exp(-self.trend_boost_scale * t_excess))
            boost = np.clip(boost, 1.0, self.trend_boost_max)
            rm = np.where(out["regime"].to_numpy() == "TREND", rm * boost, rm)

        # --- Shock brake (tail-risk cut, gated + robust) ---
        if self.use_shock_brake and ("ret_1" in f.columns) and ("rv" in f.columns):
            ret1 = f["ret_1"].to_numpy(dtype=float)
            rv = f["rv"].to_numpy(dtype=float)

            # Robust floor to prevent tiny-rv false shocks
            rv_med = np.nanmedian(rv)
            rv_floor = 0.50 * rv_med if np.isfinite(rv_med) and rv_med > 0 else 0.0
            rv_safe = np.maximum(rv, rv_floor)

            shock = np.abs(ret1) / (rv_safe + 1e-12)

            # Gate: only brake during elevated risk environments
            rvz = f["rv_z"].to_numpy(dtype=float) if "rv_z" in f.columns else None
            is_stress_regime = (out["regime"].to_numpy() == "STRESS")
            is_vol_elevated = (rvz is not None) and (rvz > float(self.stress_rv_z))

            gate = is_stress_regime | is_vol_elevated

            shock_event = (shock > self.shock_th) & gate

            if self.shock_cooldown_bars > 1:
                shock_active = (
                    pd.Series(shock_event.astype(int), index=f.index)
                    .rolling(self.shock_cooldown_bars, min_periods=1)
                    .max()
                    .to_numpy(dtype=float)
                )
            else:
                shock_active = shock_event.astype(float)

            # Softer haircut (don’t nuke the strategy)
            rm = np.where(shock_active > 0.0, rm * self.shock_rm, rm)

            out["shock_active"] = shock_active
            out["shock"] = shock

        # Clip final multiplier
        rm = np.clip(rm, self.rm_floor, self.rm_cap)
        out["risk_multiplier"] = rm

        # Diagnostics
        out["trend_ok"] = trend_ok.astype(int)
        out["chop_ok"] = chop_ok.astype(int)
        out["stress"] = stress.astype(int)
        out["rv_z"] = f["rv_z"]
        out["trend"] = f["trend"]
        out["chop"] = f["chop"]

        return out
