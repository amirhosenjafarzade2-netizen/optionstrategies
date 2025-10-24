# monte_carlo.py
import numpy as np
from typing import Dict
from config import Strategy
from payoff import strategy_payoff

def gbm_paths(S0: float, mu: float, sigma: float, T: float, steps: int, n_sims: int) -> np.ndarray:
    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    rand = np.random.standard_normal((n_sims, steps))
    log_returns = drift + vol * rand
    log_returns = np.column_stack([np.zeros(n_sims), log_returns])
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths

def monte_carlo_metrics(
    strategy: Strategy, S0: float, vol: float, mu: float, T: float,
    sims: int = 5000, steps: int = 30
) -> Dict:
    paths = gbm_paths(S0, mu, vol, T, steps, sims)
    terminal = paths[:, -1]
    payoffs = strategy_payoff(strategy, terminal)

    pop = 100 * (payoffs > 0).mean()
    pol = 100 * (payoffs < 0).mean()
    lp_ratio = (payoffs[payoffs < 0].mean() / payoffs[payoffs > 0].mean()) if (payoffs > 0).any() else 0
    lp_ratio = abs(lp_ratio)

    mean_pl = payoffs.mean()
    var95 = np.percentile(payoffs, 5)
    max_dd = np.min(payoffs)
    std = payoffs.std()
    sharpe = mean_pl / std if std > 0 else 0
    downside = payoffs[payoffs < 0]
    sortino = mean_pl / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
    calmar = mean_pl / abs(max_dd) if max_dd < 0 else 0

    return {
        "pop": pop, "pol": pol, "lp_ratio": lp_ratio,
        "expected_pl": mean_pl, "var95": var95, "max_dd": max_dd,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar
    }
