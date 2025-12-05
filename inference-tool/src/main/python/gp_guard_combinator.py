import random
from itertools import chain, combinations

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from gp_fitness import correct_dt, fitness_dt
from gp_pset import setup_pset, setup_simple_pset
from gp_reproduction import (
    genHalfAndHalf,
    mutateByCommute,
    mutateByFuzz,
    mutateByNegate,
    mutateByTerminal,
    mutInsert,
    random_seed,
)
from gp_simplification import simplify

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)


def subsets(individual):
    return chain.from_iterable(combinations(individual, r) for r in range(1, len(individual)))


def strip_unnecessary_clauses(individual, points, pset):
    """
    Try to remove surplus clauses from individuals.
    Return the original individual if no clauses can be removed.
    """
    for subset in subsets(individual):
        subset = creator.Individual(subset)
        if correct_dt(subset, points, pset):
            subset.fitness.values = fitness_dt(subset, points, pset)
            return subset
    return individual


def parsimony_select(individuals, k):
    """Select the *k* best individuals among the input *individuals*, breaking
    ties by the size of the individual, so that smaller individuals are selected
    over larger ones. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list containing the k best individuals.
    """
    return sorted(
        individuals,
        key=lambda i: i.fitness.values + ((1 / len(i)),) + ((1 / sum(c.height for c in i)),),
        reverse=True,
    )[:k]


def eaMuPlusLambda(population, toolbox, mu, lambda_, mutpb, ngen):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Begin the generational process
    best = max(population, key=lambda ind: ind.fitness.values)
    for _ in range(ngen):
        # Exit early as found optimal individual
        if toolbox.correct(best):
            return best
        # Vary the population
        offspring = algorithms.varAnd(population, toolbox, lambda_, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        best = max(population, key=lambda ind: ind.fitness.values)

    return best


def mutate(individual, pset, MAX_MUTATIONS=3):
    mutations = 0
    newNode = gp.PrimitiveTree(gp.PrimitiveTree.from_string(str(individual), pset))
    mutate = True
    operators = [
        # HVL SUB
        gp.mutNodeReplacement,
        # HLV DEL
        lambda individual, pset: gp.mutShrink(individual),
        # HVL INS
        mutInsert,
        # Reverse this.children if they have the same return type, e.g. (x - y) -> (y - x)
        mutateByCommute,
        # mutate by replacing a random node with a terminal
        mutateByTerminal,
        # fuzz a terminal
        mutateByFuzz,
    ]

    while mutate and mutations < MAX_MUTATIONS:
        mutations += 1
        mutate = random.choice([True, False])
        if len(individual) < 2:
            newNode = mutInsert(newNode, pset)[0]
            continue

        newNode = random.choice(operators)(newNode, pset)[0]
    return (newNode,)


def list_mutate(expression, pset):
    return (creator.Individual([mutate(x, pset)[0] if random.choice([True, False]) else x for x in expression]),)


def generate(pset, min_, max_, creator, type_=None, simp=None):
    return gp.PrimitiveTree(genHalfAndHalf(pset, min_, max_, creator, type_=None, simp=None))


def initRepeatUpTo(container, func, n):
    """Call the function *func* up to *n* times and return the results in a
    container type `container`

    :param container: The type to put in the data from func.
    :param func: The function that will be called n times to fill the
                 container.
    :param n: The maximum number of times to repeat func.
    :returns: An instance of the container filled with data from func.
    """
    return container(func() for _ in range(random.randint(1, n)))


def run_gp(
    points: pd.DataFrame,
    ngen: int,
    mu=10,
    lambda_=5,
    max_clauses: int = 4,
    max_clause_depth: int = 4,
    indpb=0.5,
    mutpb=0.5,
    seed=0,
    seeds=None,
):
    random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pset = setup_pset(points)
    simple_pset = setup_simple_pset(points)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", fitness_dt, pset=pset, points=points, random_state=seed)
    toolbox.register("correct", correct_dt, pset=pset, points=points)
    toolbox.register("clause", generate, pset=simple_pset, min_=1, max_=max_clause_depth, creator=creator)
    toolbox.register("complex_exp", initRepeatUpTo, list, toolbox.clause, n=max_clauses)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.complex_exp)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", parsimony_select)
    toolbox.register("mate", tools.cxUniform, indpb=indpb)
    toolbox.register("mutate", list_mutate, pset=simple_pset)

    pop = toolbox.population(mu)

    if seeds is not None:
        for s in seeds:
            individual = creator.Individual([gp.PrimitiveTree.from_string(clause, pset) for clause in s])
            individual.fitness.values = toolbox.evaluate(individual)
            if individual.fitness.values[0] == 0:
                return individual
            pop.append(individual)

    best = eaMuPlusLambda(pop, toolbox, mu, lambda_, mutpb, ngen)
    return strip_unnecessary_clauses(best, points, pset)


if __name__ == "__main__":
    random.seed(0)
    points = pd.read_csv("test-guard3.csv")[["r0", "r1", "r2", "r3", "i0", "expected"]]

    samples = list(range(100))
    points = pd.DataFrame({k: [random.randint(0, 100) for _ in samples] for k in ["r1", "r2", "r3", "i0"]})
    points["expected"] = (points["r2"] + points["i0"]) <= points["r3"]

    # best = run_gp(points, 100, seeds=[["ne(i0, r0)", "ge(r2, 2)"]])
    # best = run_gp(points, 100, seeds=[["le(add(r2, i0), r3)"]])
    best = run_gp(points, 100, seeds=[], seed=2)
    print([str(x) for x in best])
    print(len(points) - best.fitness.values[0])
