# genetic.py
import random
from typing import List, Literal, Dict
import numpy as np
from config import Strategy, Leg
from monte_carlo import monte_carlo_metrics
from payoff import strategy_payoff

FitnessKey = Literal["pop", "sharpe", "epl", "drawdown"]


def mutate_leg(leg: Leg, strike_noise: float = 10, prem_noise: float = 1) -> Leg:
    """Mutate a leg by adding noise to strike and premium"""
    new_strike = max(1, leg.strike + random.uniform(-strike_noise, strike_noise))
    new_premium = max(0.01, leg.premium + random.uniform(-prem_noise, prem_noise)) if leg.type != "stock" else 0
    
    return Leg(
        type=leg.type,
        position=leg.position,
        strike=new_strike,
        premium=new_premium
    )


def crossover(leg1: Leg, leg2: Leg) -> tuple[Leg, Leg]:
    """Crossover two legs by averaging their parameters"""
    avg_strike = (leg1.strike + leg2.strike) / 2
    avg_premium = (leg1.premium + leg2.premium) / 2
    
    child1 = Leg(leg1.type, leg1.position, avg_strike, avg_premium if leg1.type != "stock" else 0)
    child2 = Leg(leg2.type, leg2.position, avg_strike, avg_premium if leg2.type != "stock" else 0)
    
    return child1, child2


def evaluate(strategy: Strategy, asset: dict, fitness: FitnessKey, cache: dict) -> float:
    """Evaluate a strategy's fitness using Monte Carlo simulation"""
    # Create cache key from strategy legs
    key = tuple((l.type, l.position, round(l.strike, 3), round(l.premium, 3)) for l in strategy.legs)
    
    # Use cached result if available
    if key not in cache:
        cache[key] = monte_carlo_metrics(strategy, sims=500, **asset)
    
    res = cache[key]

    # Return fitness based on selected metric
    if fitness == "pop":
        return res["pop"]
    elif fitness == "epl":
        return res["expected_pl"]
    elif fitness == "drawdown":
        return -res["var95"]  # Minimize loss (maximize negative VaR)
    elif fitness == "sharpe":
        # Calculate Sharpe ratio using quick simulations
        try:
            quick_payoffs = []
            for _ in range(5):
                terminal_prices = np.random.lognormal(
                    np.log(asset["S0"]) + (asset["mu"] - 0.5 * asset["vol"]**2) * asset["T"],
                    asset["vol"] * np.sqrt(asset["T"]),
                    200
                )
                payoff = strategy_payoff(strategy, terminal_prices)
                quick_payoffs.extend(payoff)
            
            std = np.std(quick_payoffs)
            if std > 0:
                return res["expected_pl"] / std
            else:
                return 0
        except:
            return res["sharpe"]
    
    return 0


def create_optimized_strategy(base: Strategy, legs: List[Leg], suffix: str = "_opt") -> Strategy:
    """Create a new strategy from optimized legs"""
    return Strategy(
        name=base.name + suffix,
        short_desc=f"Optimized {base.short_desc}",
        description=f"Genetically optimized variant of {base.name}",
        legs=legs,
        max_profit="Optimized",
        max_loss="Optimized",
        good=base.good,
        bad=base.bad,
        is_custom=True
    )


def genetic_optimize(
    base: Strategy, 
    asset: dict, 
    fitness: FitnessKey,
    pop_size: int = 50, 
    gens: int = 100
) -> List[Strategy]:
    """
    Optimize a strategy using genetic algorithm
    
    Args:
        base: Base strategy to optimize
        asset: Asset parameters (S0, vol, mu, T)
        fitness: Fitness function to optimize
        pop_size: Population size
        gens: Number of generations
        
    Returns:
        List of top 3 optimized strategies
    """
    cache = {}

    def random_individual() -> Strategy:
        """Create a random individual by mutating the base strategy"""
        legs = [mutate_leg(l, strike_noise=20, prem_noise=2) for l in base.legs]
        return create_optimized_strategy(base, legs)

    # Initialize population
    population = [random_individual() for _ in range(pop_size)]

    # Evolution loop
    for gen in range(gens):
        # Evaluate all individuals
        scores = [evaluate(ind, asset, fitness, cache) for ind in population]
        
        # Keep top 2 elite individuals
        elite_indices = np.argsort(scores)[-2:]
        elite = [population[i] for i in elite_indices]
        new_pop = elite[:]

        # Generate new population through crossover and mutation
        while len(new_pop) < pop_size:
            # Select two random parents
            p1, p2 = random.sample(population, 2)
            
            # Create child through crossover
            child_legs = []
            for l1, l2 in zip(p1.legs, p2.legs):
                if random.random() < 0.7:  # 70% crossover rate
                    c, _ = crossover(l1, l2)
                    # 20% mutation rate
                    if random.random() < 0.2:
                        c = mutate_leg(c, strike_noise=5, prem_noise=0.5)
                    child_legs.append(c)
                else:
                    # Random selection from parents
                    child_legs.append(random.choice([l1, l2]))
            
            new_pop.append(create_optimized_strategy(base, child_legs))

        population = new_pop

    # Get top 3 individuals
    final_scores = [evaluate(s, asset, fitness, cache) for s in population]
    top_indices = np.argsort(final_scores)[-3:][::-1]  # Top 3 in descending order
    
    # Rename top strategies
    top_strategies = []
    for i, idx in enumerate(top_indices):
        strategy = population[idx]
        strategy.name = f"{base.name} — GA #{i+1}"
        top_strategies.append(strategy)
    
    return top_strategies
