# engine/portfolio.py
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import json
import os
import numpy as np
import pandas as pd


@dataclass
class GuardrailConfig:
    # Exposure hygiene
    max_leverage: float = 2.0                 # absolute cap on position
    rebalance_deadband: float = 0.02          # ignore tiny target changes (abs pos units)
    smooth_alpha: float = 0.35                # EWMA smoothing strength (0..1); higher = more responsive
    max_pos_change_per_bar: float = 0.25      # cap abs(position_t - position_{t-1}) per bar

    # Risk controls (equity-based)
    dd_half_risk: float = 0.10                # at 10% drawdown -> reduce risk
    dd_stop: float = 0.20                     # at 20% drawdown -> halt trading
    daily_loss_limit: float = 0.03            # at 3% intraday loss -> halt trading

    # If halted, require this recovery from last peak to auto-resume (optional; set None to disable)
    resume_dd: Optional[float] = 0.12         # e.g. resume only once DD < 12%


@dataclass
class GuardrailState:
    peak_equity: float = 1.0
    day_start_equity: float = 1.0
    day_key: str = ""                         # YYYY-MM-DD
    halted: bool = False
    last_pos: float = 0.0                     # last executed position
    last_target: float = 0.0                  # last raw target before guards


def _date_key(ts: pd.Timestamp) -> str:
    # Use UTC date if tz-aware; else local date as provided
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]


class PortfolioGuardrails:
    """
    Stateful guardrails intended for live/paper-live usage.

    Inputs:
      - timestamp (pd.Timestamp)
      - current_equity (float)
      - target_pos (float): desired position in [-max_leverage, +max_leverage]

    Output:
      - final_pos (float)
      - debug dict (what got applied)
    """

    def __init__(self, cfg: GuardrailConfig, state: Optional[GuardrailState] = None):
        self.cfg = cfg
        self.state = state or GuardrailState()

    # -------------------------
    # Persistence helpers
    # -------------------------
    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"cfg": asdict(self.cfg), "state": asdict(self.state)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load_state(path: str, default_cfg: Optional[GuardrailConfig] = None) -> "PortfolioGuardrails":
        cfg = default_cfg or GuardrailConfig()
        if not os.path.exists(path):
            return PortfolioGuardrails(cfg=cfg, state=GuardrailState())

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        cfg_payload = payload.get("cfg", {})
        state_payload = payload.get("state", {})

        cfg2 = GuardrailConfig(**{**asdict(cfg), **cfg_payload})
        st2 = GuardrailState(**{**asdict(GuardrailState()), **state_payload})
        return PortfolioGuardrails(cfg=cfg2, state=st2)

    # -------------------------
    # Core risk scalar logic
    # -------------------------
    def _update_equity_anchors(self, ts: pd.Timestamp, equity: float) -> None:
        dk = _date_key(ts)
        if self.state.day_key != dk:
            # new day rollover
            self.state.day_key = dk
            self.state.day_start_equity = float(equity)

        if equity > self.state.peak_equity:
            self.state.peak_equity = float(equity)

    def _drawdown(self, equity: float) -> float:
        peak = max(self.state.peak_equity, 1e-12)
        return max(0.0, 1.0 - float(equity) / peak)

    def _daily_loss(self, equity: float) -> float:
        start = max(self.state.day_start_equity, 1e-12)
        return max(0.0, 1.0 - float(equity) / start)

    def _risk_scalar_from_dd(self, dd: float) -> float:
        """
        Piecewise:
          dd < dd_half_risk: 1.0
          dd_half_risk <= dd < dd_stop: linearly ramps to 0.0
          dd >= dd_stop: 0.0 (halt)
        """
        if dd >= self.cfg.dd_stop:
            return 0.0
        if dd <= self.cfg.dd_half_risk:
            return 1.0
        # linear ramp down
        x0, x1 = self.cfg.dd_half_risk, self.cfg.dd_stop
        return float(1.0 - (dd - x0) / (x1 - x0))

    def _risk_scalar_from_daily(self, dl: float) -> float:
        if dl >= self.cfg.daily_loss_limit:
            return 0.0
        return 1.0

    def _maybe_resume(self, dd: float) -> None:
        # Optional resume logic if halted
        if not self.state.halted:
            return
        if self.cfg.resume_dd is None:
            return
        if dd < self.cfg.resume_dd:
            self.state.halted = False

    # -------------------------
    # Position hygiene
    # -------------------------
    def _clip_leverage(self, pos: float) -> float:
        m = float(self.cfg.max_leverage)
        return float(np.clip(pos, -m, m))

    def _apply_deadband(self, target: float, prev: float) -> float:
        # If change is tiny, keep previous position
        if abs(target - prev) < float(self.cfg.rebalance_deadband):
            return float(prev)
        return float(target)

    def _apply_smoothing(self, target: float, prev: float) -> float:
        a = float(self.cfg.smooth_alpha)
        # EWMA: new = a*target + (1-a)*prev
        return float(a * target + (1.0 - a) * prev)

    def _cap_pos_change(self, new_pos: float, prev: float) -> float:
        cap = float(self.cfg.max_pos_change_per_bar)
        delta = new_pos - prev
        if abs(delta) <= cap:
            return float(new_pos)
        return float(prev + np.sign(delta) * cap)

    # -------------------------
    # Public API
    # -------------------------
    def apply(self, ts: pd.Timestamp, equity: float, target_pos: float) -> Tuple[float, Dict[str, Any]]:
        """
        Returns final_pos, debug_info
        """
        self._update_equity_anchors(ts, equity)
        dd = self._drawdown(equity)
        dl = self._daily_loss(equity)

        # Resume logic (if configured)
        self._maybe_resume(dd)

        # Determine risk scalar and halt status
        rs_dd = self._risk_scalar_from_dd(dd)
        rs_dl = self._risk_scalar_from_daily(dl)
        risk_scalar = float(min(rs_dd, rs_dl))

        if risk_scalar <= 0.0:
            self.state.halted = True

        prev_pos = float(self.state.last_pos)

        # Apply risk scalar to target
        raw_target = float(target_pos)
        risk_target = raw_target * (0.0 if self.state.halted else risk_scalar)

        # Hygiene pipeline
        x = self._clip_leverage(risk_target)
        x = self._apply_deadband(x, prev_pos)
        x = self._apply_smoothing(x, prev_pos)
        x = self._cap_pos_change(x, prev_pos)
        x = self._clip_leverage(x)

        # Update state
        self.state.last_target = raw_target
        self.state.last_pos = float(x)

        dbg = {
            "timestamp": str(ts),
            "equity": float(equity),
            "peak_equity": float(self.state.peak_equity),
            "day_start_equity": float(self.state.day_start_equity),
            "drawdown": float(dd),
            "daily_loss": float(dl),
            "risk_scalar_dd": float(rs_dd),
            "risk_scalar_daily": float(rs_dl),
            "risk_scalar": float(0.0 if self.state.halted else risk_scalar),
            "halted": bool(self.state.halted),
            "target_pos_raw": float(raw_target),
            "target_pos_after_risk": float(risk_target),
            "final_pos": float(x),
            "prev_pos": float(prev_pos),
            "pos_change": float(x - prev_pos),
        }
        return float(x), dbg


# -------------------------
# Optional: simple vector helper for research/backtests
# -------------------------
def apply_deadband_smoothing_series(
    target_pos: pd.Series,
    deadband: float = 0.02,
    smooth_alpha: float = 0.35,
    max_pos_change_per_bar: float = 0.25,
    max_leverage: float = 2.0,
) -> pd.Series:
    """
    Stateless helper: applies deadband/smoothing/turnover cap to a target position series.
    Useful for research (not using equity-based halts).
    """
    idx = target_pos.index
    out = []
    prev = 0.0
    for t in idx:
        x = float(target_pos.loc[t])
        x = float(np.clip(x, -max_leverage, max_leverage))

        # deadband
        if abs(x - prev) < deadband:
            x = prev

        # smoothing
        x = smooth_alpha * x + (1.0 - smooth_alpha) * prev

        # cap per-bar change
        delta = x - prev
        if abs(delta) > max_pos_change_per_bar:
            x = prev + np.sign(delta) * max_pos_change_per_bar

        x = float(np.clip(x, -max_leverage, max_leverage))
        out.append(x)
        prev = x

    return pd.Series(out, index=idx, name="pos_guarded")
