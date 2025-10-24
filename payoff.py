# payoff.py
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from config import Leg, Strategy


def leg_payoff(leg: Leg, S: np.ndarray) -> np.ndarray:
    """
    Calculate payoff for a single leg at expiration.
    
    For long positions: you PAY the premium (cost)
    For short positions: you RECEIVE the premium (income)
    """
    if leg.type == "stock":
        # Stock position: profit/loss based on price movement
        if leg.position == "long":
            return S - leg.strike  # Buy at strike, current value S
        else:
            return leg.strike - S  # Sold at strike, buy back at S
    
    # Calculate intrinsic value at expiration
    if leg.type == "call":
        intrinsic = np.maximum(0, S - leg.strike)
    else:  # put
        intrinsic = np.maximum(0, leg.strike - S)
    
    # Apply position sign and premium
    if leg.position == "long":
        # Long: you pay premium, receive intrinsic value
        return intrinsic - leg.premium
    else:  # short
        # Short: you receive premium, pay intrinsic value
        return leg.premium - intrinsic


def strategy_payoff(strategy: Strategy, S: np.ndarray) -> np.ndarray:
    """Calculate total strategy payoff across all legs"""
    total = np.zeros_like(S, dtype=float)
    for leg in strategy.legs:
        total += leg_payoff(leg, S)
    return total


def calculate_breakeven_points(strategy: Strategy, S0: float = None) -> List[float]:
    """Calculate breakeven points for the strategy"""
    strikes = [l.strike for l in strategy.legs if l.type != "stock"]
    if not strikes and S0:
        strikes = [S0]
    elif not strikes:
        return []
    
    min_s = min(strikes) * 0.5
    max_s = max(strikes) * 1.5
    S = np.linspace(max(1, min_s), max_s, 10000)
    payoff = strategy_payoff(strategy, S)
    
    # Find zero crossings
    breakevens = []
    for i in range(len(payoff) - 1):
        if payoff[i] * payoff[i + 1] < 0:  # Sign change
            # Linear interpolation for more accuracy
            x1, x2 = S[i], S[i + 1]
            y1, y2 = payoff[i], payoff[i + 1]
            breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakevens.append(breakeven)
    
    return breakevens


def calculate_max_profit_loss(strategy: Strategy, S0: float = None) -> Tuple[float, float]:
    """Calculate maximum profit and loss for the strategy"""
    strikes = [l.strike for l in strategy.legs if l.type != "stock"]
    if not strikes and S0:
        strikes = [S0]
    elif not strikes:
        return (float('inf'), float('-inf'))
    
    min_s = min(strikes) * 0.5
    max_s = max(strikes) * 1.5
    S = np.linspace(max(1, min_s), max_s, 1000)
    payoff = strategy_payoff(strategy, S)
    
    # Also check extreme values
    extreme_low = strategy_payoff(strategy, np.array([0.01]))
    extreme_high = strategy_payoff(strategy, np.array([max_s * 10]))
    
    all_payoffs = np.concatenate([payoff, extreme_low, extreme_high])
    
    max_profit = np.max(all_payoffs)
    max_loss = np.min(all_payoffs)
    
    # Check if profit/loss is theoretically unlimited
    if max_profit > 1e6:
        max_profit = float('inf')
    if max_loss < -1e6:
        max_loss = float('-inf')
    
    return max_profit, max_loss


def payoff_chart(strategy: Strategy, S0: float = None) -> go.Figure:
    """Generate interactive payoff chart for strategy"""
    strikes = [l.strike for l in strategy.legs if l.type != "stock"]
    
    if strikes:
        min_s = min(strikes)
        max_s = max(strikes)
    elif S0:
        min_s = S0 * 0.7
        max_s = S0 * 1.3
    else:
        min_s = 50
        max_s = 150
    
    S = np.linspace(max(1, min_s * 0.7), max_s * 1.3, 500)
    payoff = strategy_payoff(strategy, S)
    
    # Calculate breakeven points
    breakevens = calculate_breakeven_points(strategy, S0)
    
    # Create figure
    fig = go.Figure()
    
    # Add payoff line with color gradient
    colors = ['#ef4444' if p < 0 else '#10b981' for p in payoff]
    
    fig.add_trace(go.Scatter(
        x=S, 
        y=payoff, 
        mode="lines", 
        name="P/L",
        line=dict(color="#3b82f6", width=3),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
        hovertemplate="Price: $%{x:.2f}<br>P/L: $%{y:.2f}<extra></extra>"
    ))
    
    # Add zero line
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="#94a3b8", 
        line_width=2,
        annotation_text="Break-even",
        annotation_position="right"
    )
    
    # Add current price line if provided
    if S0:
        fig.add_vline(
            x=S0,
            line_dash="dot",
            line_color="#f59e0b",
            line_width=2,
            annotation_text=f"Current: ${S0:.2f}",
            annotation_position="top"
        )
    
    # Mark breakeven points
    for be in breakevens:
        if min_s * 0.7 <= be <= max_s * 1.3:
            fig.add_trace(go.Scatter(
                x=[be],
                y=[0],
                mode="markers",
                marker=dict(size=10, color="#8b5cf6", symbol="circle"),
                name=f"BE: ${be:.2f}",
                hovertemplate=f"Breakeven: ${be:.2f}<extra></extra>"
            ))
    
    # Calculate max profit/loss
    max_profit, max_loss = calculate_max_profit_loss(strategy, S0)
    
    # Update layout
    title_text = f"{strategy.name} — Payoff at Expiration"
    if max_profit != float('inf') and max_loss != float('-inf'):
        title_text += f"<br><sub>Max Profit: ${max_profit:.2f} | Max Loss: ${max_loss:.2f}</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Underlying Price ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig
