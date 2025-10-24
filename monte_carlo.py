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
        mu: Drift rate (daily)
        sigma: Volatility (annual)
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
    
    # Add initial zero return
    log_returns = np.column_stack([np.zeros(n_sims), log_returns])
    
    # Calculate prices
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    return paths


def monte_carlo_metrics(
    strategy: Strategy, 
    S0: float, 
    vol: float, 
    mu: float, 
    T: float,
    sims: int = 5000, 
    steps: int = 30
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
        
    Returns:
        Dictionary containing various risk/return metrics
    """
    # Generate price paths
    paths = gbm_paths(S0, mu, vol, T, steps, sims)
    terminal = paths[:, -1]
    
    # Calculate payoffs at expiration
    payoffs = strategy_payoff(strategy, terminal)
    
    # Basic probability metrics
    pop = 100 * np.mean(payoffs > 0)  # Probability of profit
    pol = 100 * np.mean(payoffs < 0)  # Probability of loss
    poe = 100 * np.mean(payoffs == 0)  # Probability of exactly breaking even
    
    # Profit/Loss analysis
    profitable = payoffs[payoffs > 0]
    losing = payoffs[payoffs < 0]
    
    avg_win = np.mean(profitable) if len(profitable) > 0 else 0
    avg_loss = np.mean(losing) if len(losing) > 0 else 0
    
    # Loss/Profit ratio (how much average loss vs average win)
    lp_ratio = abs(avg_loss / avg_win) if avg_win > 0 else 0
    
    # Profit factor (gross profit / gross loss)
    gross_profit = np.sum(profitable) if len(profitable) > 0 else 0
    gross_loss = abs(np.sum(losing)) if len(losing) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expected value metrics
    mean_pl = np.mean(payoffs)
    median_pl = np.median(payoffs)
    std_pl = np.std(payoffs)
    
    # Risk metrics
    var95 = np.percentile(payoffs, 5)  # 95% Value at Risk
    var99 = np.percentile(payoffs, 1)  # 99% Value at Risk
    cvar95 = np.mean(payoffs[payoffs <= var95])  # Conditional VaR (Expected Shortfall)
    max_dd = np.min(payoffs)  # Maximum drawdown (worst case)
    max_profit = np.max(payoffs)  # Best case
    
    # Risk-adjusted returns
    sharpe = mean_pl / std_pl if std_pl > 0 else 0
    
    # Sortino ratio (uses only downside deviation)
    downside = payoffs[payoffs < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0
    sortino = mean_pl / downside_std if downside_std > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    calmar = mean_pl / abs(max_dd) if max_dd < 0 else 0
    
    # Win rate
    win_rate = pop / 100 if (pop + pol) > 0 else 0
    
    # Return on Risk (Expected return / VaR)
    ror = mean_pl / abs(var95) if var95 < 0 else 0
    
    return {
        # Probability metrics
        "pop": pop,
        "pol": pol,
        "poe": poe,
        "win_rate": win_rate,
        
        # P/L metrics
        "expected_pl": mean_pl,
        "median_pl": median_pl,
        "std_pl": std_pl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "lp_ratio": lp_ratio,
        "profit_factor": profit_factor,
        
        # Risk metrics
        "var95": var95,
        "var99": var99,
        "cvar95": cvar95,
        "max_dd": max_dd,
        "max_profit": max_profit,
        
        # Risk-adjusted returns
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "ror": ror,
        
        # Raw data for distributions
        "payoffs": payoffs
    }


def monte_carlo_distribution(
    strategy: Strategy,
    S0: float,
    vol: float,
    mu: float,
    T: float,
    sims: int = 5000
) -> Dict:
    """
    Generate distribution data for visualization.
    
    Returns:
        Dictionary with price distribution and payoff distribution
    """
    paths = gbm_paths(S0, mu, vol, T, 30, sims)
    terminal = paths[:, -1]
    payoffs = strategy_payoff(strategy, terminal)
    
    return {
        "terminal_prices": terminal,
        "payoffs": payoffs,
        "price_mean": np.mean(terminal),
        "price_std": np.std(terminal),
        "payoff_mean": np.mean(payoffs),
        "payoff_std": np.std(payoffs)
    }
