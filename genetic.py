# genetic.py
import random
from typing import List, Literal, Dict, Tuple
import numpy as np
from config import Strategy, Leg
from monte_carlo import monte_carlo_metrics
from payoff import strategy_payoff

FitnessKey = Literal["pop", "sharpe", "epl", "drawdown", "sortino", "calmar", "profit_factor"]


def mutate_leg(leg: Leg, strike_noise: float = 10, prem_noise: float = 1) -> Leg:
    """
    Mutate a leg by adding noise to strike and premium.
    
    Args:
        leg: Leg to mutate
        strike_noise: Maximum absolute change in strike price
        prem_noise: Maximum absolute change in premium
        
    Returns:
        New mutated leg
    """
    new_strike = max(1, leg.strike + random.uniform(-strike_noise, strike_noise))
    
    if leg.type != "stock":
        new_premium = max(0.01, leg.premium + random.uniform(-prem_noise, prem_noise))
    else:
        new_premium = 0
    
    return Leg(
        type=leg.type,
        position=leg.position,
        strike=new_strike,
        premium=new_premium
    )


def crossover(leg1: Leg, leg2: Leg) -> Tuple[Leg, Leg]:
    """
    Crossover two legs by averaging their parameters.
    
    Args:
        leg1: First parent leg
        leg2: Second parent leg
        
    Returns:
        Tuple of two child legs
    """
    avg_strike = (leg1.strike + leg2.strike) / 2
    avg_premium = (leg1.premium + leg2.premium) / 2
    
    child1 = Leg(
        leg1.type, 
        leg1.position, 
        avg_strike, 
        avg_premium if leg1.type != "stock" else 0
    )
    child2 = Leg(
        leg2.type, 
        leg2.position, 
        avg_strike, 
        avg_premium if leg2.type != "stock" else 0
    )
    
    return child1, child2


def evaluate(strategy: Strategy, asset: dict, fitness: FitnessKey, cache: dict) -> float:
    """
    Evaluate a strategy's fitness using Monte Carlo simulation.
    
    Args:
        strategy: Strategy to evaluate
        asset: Asset parameters (S0, vol, mu, T)
        fitness: Fitness metric to optimize
        cache: Cache for storing results
        
    Returns:
        Fitness score (higher is better)
    """
    # Create cache key from strategy legs
    key = tuple((l.type, l.position, round(l.strike, 2), round(l.premium, 2)) for l in strategy.legs)
    
    # Use cached result if available
    if key not in cache:
        try:
            cache[key] = monte_carlo_metrics(strategy, sims=1000, steps=30, **asset)
        except Exception as e:
            # If simulation fails, return very bad score
            return -1e6
    
    res = cache[key]
    
    # Return fitness based on selected metric
    if fitness == "pop":
        return res["pop"]
    elif fitness == "epl":
        return res["expected_pl"]
    elif fitness == "drawdown":
        # Minimize loss (maximize negative of max loss)
        return -res["max_dd"] if res["max_dd"] < 0 else res["max_dd"]
    elif fitness == "sharpe":
        return res["sharpe"]
    elif fitness == "sortino":
        return res["sortino"]
    elif fitness == "calmar":
        return res["calmar"]
    elif fitness == "profit_factor":
        pf = res.get("profit_factor", 0)
        return pf if pf != float('inf') else 100
    
    return 0


def create_optimized_strategy(base: Strategy, legs: List[Leg], suffix: str = "_opt") -> Strategy:
    """
    Create a new strategy from optimized legs.
    
    Args:
        base: Base strategy
        legs: Optimized legs
        suffix: Suffix for strategy name
        
    Returns:
        New strategy with optimized parameters
    """
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


def validate_strategy(strategy: Strategy, asset: dict) -> bool:
    """
    Validate that a strategy is reasonable.
    
    Args:
        strategy: Strategy to validate
        asset: Asset parameters
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check that all strikes are positive
        for leg in strategy.legs:
            if leg.strike <= 0:
                return False
            if leg.premium < 0:
                return False
            # Strike should be somewhat reasonable relative to S0
            if leg.type != "stock" and (leg.strike < asset["S0"] * 0.1 or leg.strike > asset["S0"] * 10):
                return False
        
        # Try to calculate a sample payoff
        test_prices = np.array([asset["S0"] * 0.5, asset["S0"], asset["S0"] * 1.5])
        payoffs = strategy_payoff(strategy, test_prices)
        
        # Check for NaN or inf
        if np.any(np.isnan(payoffs)) or np.any(np.isinf(payoffs)):
            return False
        
        return True
    except:
        return False


def genetic_optimize(
    base: Strategy, 
    asset: dict, 
    fitness: FitnessKey,
    pop_size: int = 50, 
    gens: int = 100,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.7,
    elitism: int = 2
) -> List[Strategy]:
    """
    Optimize a strategy using genetic algorithm.
    
    Args:
        base: Base strategy to optimize
        asset: Asset parameters (S0, vol, mu, T)
        fitness: Fitness function to optimize
        pop_size: Population size
        gens: Number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        elitism: Number of elite individuals to keep
        
    Returns:
        List of top 3 optimized strategies
    """
    cache = {}
    
    def random_individual() -> Strategy:
        """Create a random individual by mutating the base strategy"""
        # Use varying noise levels
        strike_noise = asset["S0"] * 0.15  # 15% of current price
        prem_noise = 2.0
        
        legs = [mutate_leg(l, strike_noise=strike_noise, prem_noise=prem_noise) for l in base.legs]
        strategy = create_optimized_strategy(base, legs)
        
        # Validate strategy
        if not validate_strategy(strategy, asset):
            # If invalid, return base strategy
            return create_optimized_strategy(base, base.legs)
        
        return strategy
    
    # Initialize population
    population = [random_individual() for _ in range(pop_size)]
    
    best_score_history = []
    
    # Evolution loop
    for gen in range(gens):
        # Evaluate all individuals
        scores = [evaluate(ind, asset, fitness, cache) for ind in population]
        
        # Track best score
        best_score_history.append(max(scores))
        
        # Sort population by fitness
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Keep elite individuals
        elite = population[:elitism]
        new_pop = elite[:]
        
        # Generate new population through selection, crossover, and mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            tournament_size = 3
            parent1_idx = max(random.sample(range(len(population)), tournament_size), 
                            key=lambda i: scores[i])
            parent2_idx = max(random.sample(range(len(population)), tournament_size), 
                            key=lambda i: scores[i])
            
            p1 = population[parent1_idx]
            p2 = population[parent2_idx]
            
            # Create child through crossover
            child_legs = []
            for l1, l2 in zip(p1.legs, p2.legs):
                if random.random() < crossover_rate:
                    # Crossover
                    c, _ = crossover(l1, l2)
                else:
                    # Random selection from parents
                    c = random.choice([l1, l2])
                
                # Mutation
                if random.random() < mutation_rate:
                    strike_noise = asset["S0"] * 0.05  # Smaller mutation
                    prem_noise = 0.5
                    c = mutate_leg(c, strike_noise=strike_noise, prem_noise=prem_noise)
                
                child_legs.append(c)
            
            child = create_optimized_strategy(base, child_legs)
            
            # Validate child
            if validate_strategy(child, asset):
                new_pop.append(child)
            else:
                # If invalid, use parent instead
                new_pop.append(random.choice([p1, p2]))
        
        population = new_pop[:pop_size]
    
    # Final evaluation
    final_scores = [evaluate(s, asset, fitness, cache) for s in population]
    top_indices = np.argsort(final_scores)[-3:][::-1]  # Top 3 in descending order
    
    # Rename top strategies with their scores
    top_strategies = []
    for i, idx in enumerate(top_indices):
        strategy = population[idx]
        score = final_scores[idx]
        strategy.name = f"{base.name} — GA #{i+1} (Score: {score:.2f})"
        top_strategies.append(strategy)
    
    return top_strategies
