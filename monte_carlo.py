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
    strategy: Strategy,
    S0: float, vol: float, mu: float, T: float,
    sims: int = 5000, steps: int = 30
) -> Dict:
    paths = gbm_paths(S0, mu, vol, T, steps, sims)
    terminal = paths[:, -1]
    payoffs = strategy_payoff(strategy, terminal)

    pop = 100 * (payoffs > 0).mean()
    expected_pl = payoffs.mean()
    var95 = np.percentile(payoffs, 5)

    return {"pop": pop, "expected_pl": expected_pl, "var95": var95}
