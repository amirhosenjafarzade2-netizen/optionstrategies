# genetic.py
import random
from typing import List, Literal, Dict
import numpy as np
from config import Strategy, Leg
from monte_carlo import monte_carlo_metrics
from payoff import strategy_payoff

FitnessKey = Literal["pop", "sharpe", "epl", "drawdown"]


def mutate_leg(leg: Leg, strike_noise: float = 10, prem_noise: float = 1) -> Leg:
    return Leg(
        type=leg.type,
        position=leg.position,
        strike=max(1, leg.strike + random.uniform(-strike_noise, strike_noise)),
        premium=max(0.01, leg.premium + random.uniform(-prem_noise, prem_noise))
    )


def crossover(leg1: Leg, leg2: Leg) -> tuple[Leg, Leg]:
    return (
        Leg(leg1.type, leg1.position, (leg1.strike + leg2.strike) / 2, (leg1.premium + leg2.premium) / 2),
        Leg(leg2.type, leg2.position, (leg1.strike + leg2.strike) / 2, (leg1.premium + leg2.premium) / 2)
    )


def evaluate(strategy: Strategy, asset: dict, fitness: FitnessKey, cache: dict) -> float:
    key = tuple((l.type, l.position, round(l.strike, 3), round(l.premium, 3)) for l in strategy.legs)
    if key not in cache:
        cache[key] = monte_carlo_metrics(strategy, sims=500, **asset)
    res = cache[key]

    if fitness == "pop":
        return res["pop"]
    if fitness == "epl":
        return res["expected_pl"]
    if fitness == "drawdown":
        return -res["var95"]

    # Sharpe: use quick 200 sims for std
    quick_payoffs = [strategy_payoff(strategy, np.random.lognormal(
        np.log(asset["S0"]), asset["vol"] * np.sqrt(asset["T"]), 200
    )) for _ in range(5)]
    std = np.std(np.concatenate(quick_payoffs))
    return res["expected_pl"] / std if std > 0 else 0


def genetic_optimize(base: Strategy, asset: dict, fitness: FitnessKey,
                     pop_size: int = 50, gens: int = 100) -> List[Strategy]:
    cache = {}

    def random_ind() -> Strategy:
        legs = [mutate_leg(l, 20, 2) for l in base.legs]
        return Strategy(name=base.name + "_opt", description="GA", legs=legs,
                        max_profit="?", max_loss="?", good="", bad="", is_custom=True)

    population = [random_ind() for _ in range(pop_size)]

    for gen in range(gens):
        scores = [evaluate(ind, asset, fitness, cache) for ind in population]
        elite = [population[i] for i in np.argsort(scores)[-2:]]
        new_pop = elite[:]

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            child_legs = []
            for l1, l2 in zip(p1.legs, p2.legs):
                if random.random() < 0.7:
                    c, _ = crossover(l1, l2)
                    child_legs.append(mutate_leg(c, 5, 0.5) if random.random() < 0.2 else c)
                else:
                    child_legs.append(random.choice([l1, l2]))
            new_pop.append(Strategy(name=base.name + "_opt", description="GA",
                                    legs=child_legs, max_profit="?", max_loss="?", good="", bad="", is_custom=True))

        population = new_pop

    final_scores = [evaluate(s, asset, fitness, cache) for s in population]
    top_idx = np.argsort(final_scores)[-3:][::-1]
    for i, idx in enumerate(top_idx):
        population[idx].name = f"{base.name} – GA #{i+1}"
    return [population[i] for i in top_idx]
