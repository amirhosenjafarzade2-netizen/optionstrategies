# monte_carlo.py
import numpy as np
from typing import Dict
from config import Strategy
from payoff import strategy_payoff


def gbm_paths(S0: float, mu: float, sigma: float, T: float, steps: int, n_sims: int) -> np.ndarray:
    """
    Generate price paths using Geometric Brownian Motion.
    
    Args:
        S0: Initial stock price
        mu: Daily drift rate
        sigma: Annual volatility
        T: Time to expiration (years)
        steps: Number of time steps
        n_sims: Number of simulations
        
    Returns:
        Array of shape (n_sims, steps+1) containing price paths
    """
    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # Generate random returns
    rand = np.random.standard_normal((n_sims, steps))
    log_returns = drift + vol * rand
    
    # Prepend zero for initial price
    log_returns = np.column_stack([np.zeros(n_sims), log_returns])
    
    # Exponentiate cumulative sum to get price paths
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    return paths


def monte_carlo_metrics(
    strategy: Strategy, 
    S0: float, 
    vol: float, 
    mu: float, 
    T: float,
    sims: int = 5000, 
    steps: int = 30,
    tolerance: float = 0.10  # NEW: Breakeven tolerance in $
) -> Dict:
    """
    Calculate comprehensive Monte Carlo metrics for a strategy.
    
    Args:
        strategy: Options strategy to evaluate
        S0: Current stock price
        vol: Annual volatility
        mu: Daily drift
        T: Time to expiration (years)
        sims: Number of simulations
        steps: Number of time steps per simulation
        tolerance: Payoff within ±tolerance counts as breakeven (e.g., $0.10)
        
    Returns:
        Dictionary containing various risk/return metrics
    """
    # Generate price paths
    paths = gbm_paths(S0, mu, vol, T, steps, sims)
    terminal = paths[:, -1]
    
    # Calculate payoffs at expiration
    payoffs = strategy_payoff(strategy, terminal)
    
    # === ROBUST PROBABILITY METRICS WITH TOLERANCE ===
    profitable = payoffs > tolerance
    losing = payoffs < -tolerance
    breakeven = np.logical_and(payoffs >= -tolerance, payoffs <= tolerance)
    
    pop = 100 * np.mean(profitable)
    pol = 100 * np.mean(losing)
    poe = 100 * np.mean(breakeven)
    
    # Filter only meaningful wins/losses
    profitable_vals = payoffs[profitable]
    losing_vals = payoffs[losing]
    
    avg_win = np.mean(profitable_vals) if len(profitable_vals) > 0 else 0.0
    avg_loss = np.mean(losing_vals) if len(losing_vals) > 0 else 0.0
    
    # Loss/Profit ratio (how much average loss vs average win)
    lp_ratio = abs(avg_loss / avg_win) if avg_win > 0 else 0.0
    
    # Profit factor (gross profit / gross loss)
    gross_profit = np.sum(profitable_vals) if len(profitable_vals) > 0 else 0.0
    gross_loss = abs(np.sum(losing_vals)) if len(losing_vals) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expected value metrics
    mean_pl = np.mean(payoffs)
    median_pl = np.median(payoffs)
    std_pl = np.std(payoffs)
    
    # Risk metrics
    var95 = np.percentile(payoffs, 5)   # 95% Value at Risk
    var99 = np.percentile(payoffs, 1)   # 99% Value at Risk
    cvar95 = np.mean(payoffs[payoffs <= var95]) if np.any(payoffs <= var95) else mean_pl
    max_dd = np.min(payoffs)            # Worst case
    max_profit = np.max(payoffs)        # Best case
    
    # Risk-adjusted returns
    sharpe = mean_pl / std_pl if std_pl > 1e-8 else 0.0
    
    # Sortino ratio (downside deviation only)
    downside = payoffs[payoffs < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    sortino = mean_pl / downside_std if downside_std > 1e-8 else 0.0
    
    # Calmar ratio (return / max drawdown)
    calmar = mean_pl / abs(max_dd) if max_dd < 0 else 0.0
    
    # Win rate (only meaningful wins)
    win_rate = pop / 100 if (pop + pol) > 0 else 0.0
    
    # Return on Risk (Expected return / VaR)
    ror = mean_pl / abs(var95) if var95 < 0 else 0.0
    
    return {
        # Probability metrics
        "pop": round(pop, 2),
        "pol": round(pol, 2),
        "poe": round(poe, 2),
        "win_rate": round(win_rate, 3),
        
        # P/L metrics
        "expected_pl": round(mean_pl, 3),
        "median_pl": round(median_pl, 3),
        "std_pl": round(std_pl, 3),
        "avg_win": round(avg_win, 3) if avg_win > 0 else 0.0,
        "avg_loss": round(avg_loss, 3) if avg_loss < 0 else 0.0,
        "lp_ratio": round(lp_ratio, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor != float('inf') else 999.99,
        
        # Risk metrics
        "var95": round(var95, 3),
        "var99": round(var99, 3),
        "cvar95": round(cvar95, 3),
        "max_dd": round(max_dd, 3),
        "max_profit": round(max_profit, 3),
        
        # Risk-adjusted returns
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "ror": round(ror, 3),
        
        # Raw data for distributions
        "payoffs": payoffs,
        "terminal_prices": terminal
    }


def monte_carlo_distribution(
    strategy: Strategy,
    S0: float,
    vol: float,
    mu: float,
    T: float,
    sims: int = 5000,
    tolerance: float = 0.10
) -> Dict:
    """
    Generate distribution data for visualization.
    
    Returns:
        Dictionary with price and payoff distributions + stats
    """
    paths = gbm_paths(S0, mu, vol, T, 30, sims)
    terminal = paths[:, -1]
    payoffs = strategy_payoff(strategy, terminal)
    
    # Apply tolerance for visualization
    profitable = payoffs > tolerance
    losing = payoffs < -tolerance
    breakeven = np.abs(payoffs) <= tolerance
    
    return {
        "terminal_prices": terminal,
        "payoffs": payoffs,
        "price_mean": np.mean(terminal),
        "price_std": np.std(terminal),
        "payoff_mean": np.mean(payoffs),
        "payoff_std": np.std(payoffs),
        "pop": 100 * np.mean(profitable),
        "pol": 100 * np.mean(losing),
        "poe": 100 * np.mean(breakeven),
        "tolerance": tolerance
    }
