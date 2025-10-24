# payoff.py
import numpy as np
import plotly.graph_objects as go
from typing import List
from config import Leg, Strategy


def leg_payoff(leg: Leg, S: np.ndarray) -> np.ndarray:
    if leg.type == "stock":
        return S - leg.strike if leg.position == "long" else leg.strike - S
    intrinsic = np.maximum(0, S - leg.strike) if leg.type == "call" else np.maximum(0, leg.strike - S)
    sign = 1 if leg.position == "long" else -1
    return sign * (intrinsic - leg.premium)


def strategy_payoff(strategy: Strategy, S: np.ndarray) -> np.ndarray:
    return sum(leg_payoff(leg, S) for leg in strategy.legs)


def payoff_chart(strategy: Strategy, S0: float = None) -> go.Figure:
    strikes = [l.strike for l in strategy.legs if l.type != "stock"]
    min_s = min(strikes) if strikes else S0 * 0.7
    max_s = max(strikes) if strikes else S0 * 1.3
    S = np.linspace(max(1, min_s * 0.7), max_s * 1.3, 500)
    payoff = strategy_payoff(strategy, S)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S, y=payoff, mode="lines", name="P/L",
        line=dict(color="#3b82f6", width=3), fill="tozeroy",
        hovertemplate="Price: $%{x:.2f}<br>P/L: $%{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", annotation_text="Break-even")
    fig.update_layout(
        title=f"{strategy.name} – Payoff at Expiration",
        xaxis_title="Underlying Price ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    return fig
