import numpy as np
import pandas as pd

from engine.data import load_parquet
from engine.features_candles import CandleFeatureBuilder
from engine.regime import RegimeEngine

def main():
    df = load_parquet("data/raw/BTCUSDT_1h.parquet").dropna()

    fb = CandleFeatureBuilder()
    feats = fb.build(df).df

    re = RegimeEngine()
    reg = re.score(feats)

    X = pd.DataFrame(index=feats.index)
    X["rv_z"] = reg["rv_z"]
    X["trend"] = reg["trend"]
    X["chop"] = reg["chop"]
    X["risk_multiplier"] = reg["risk_multiplier"]
    X["regime"] = reg["regime"]

    X = X.dropna().copy()

    # Optional: subsample for speed/plot readability
    Xs = X.iloc[::2].copy()  # every 2nd bar

    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # Hard cap chop for visualization (extra safety)
    X = X[X["chop"] < X["chop"].quantile(0.999)]

    # --- Plotly interactive 3D ---
    import plotly.express as px

    fig = px.scatter_3d(
        Xs,
        x="rv_z",
        y="trend",
        z="chop",
        color="risk_multiplier",
        hover_data=["regime"],
        title="BTC Regime State Space (rv_z, trend, chop)"
    )
    fig.show()

    # --- Trajectory path (last N points) ---
    N = 500
    tail = X.tail(N).copy()

    import plotly.graph_objects as go
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter3d(
        x=tail["rv_z"], y=tail["trend"], z=tail["chop"],
        mode="lines",
        name="trajectory",
    ))
    fig2.add_trace(go.Scatter3d(
        x=[tail["rv_z"].iloc[-1]],
        y=[tail["trend"].iloc[-1]],
        z=[tail["chop"].iloc[-1]],
        mode="markers",
        marker=dict(size=6),
        name="current",
    ))

    fig2.update_layout(
        title=f"Last {N} bars trajectory in regime space",
        scene=dict(
            xaxis_title="rv_z",
            yaxis_title="trend",
            zaxis_title="chop"
        )
    )
    fig2.show()

if __name__ == "__main__":
    main()
