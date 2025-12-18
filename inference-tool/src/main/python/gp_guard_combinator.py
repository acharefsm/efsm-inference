import random
from itertools import chain, combinations

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from gp_fitness import correct_dt, fitness_dt, tree_to_guard
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

creator.create("Fitness_Guard", base.Fitness, weights=(1.0,))
creator.create("Individual_Guard", list, fitness=creator.Fitness_Guard)


def subsets(individual):
    return chain.from_iterable(combinations(individual, r) for r in range(1, len(individual)))


def strip_unnecessary_clauses(individual, points, pset):
    """
    Try to remove surplus clauses from individuals.
    Return the original individual if no clauses can be removed.
    """
    for subset in subsets(individual):
        subset = creator.Individual_Guard(subset)
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


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Begin the generational process
    best = max(population, key=lambda ind: ind.fitness.values)
    best_guard = toolbox.guard(best)
    for _ in range(ngen):
        print(_)
        # Exit early as found optimal individual
        if toolbox.correct(best):
            return (best, best_guard)
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
        best_guard = toolbox.guard(best)

    return (best, best_guard)


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
    return (creator.Individual_Guard([mutate(x, pset)[0] if random.choice([True, False]) else x for x in expression]),)


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
    return container(func() for _ in range(n))


def run_gp(
    points: pd.DataFrame,
    pset,
    simple_pset,
    ngen_guard = 10,
    mu_guard=10,
    lambda_guard=5,
    max_clauses: int = 4,
    max_clause_depth: int = 4,
    cxpb_guard=0.5,
    mutpb_guard=0.5,
    random_seed=0,
    seeds=None,
    **kwargs,
):
    print("random_seed:", random_seed, type(random_seed))
    random_seed = int(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", fitness_dt, pset=pset, points=points, random_state=random_seed)
    toolbox.register("correct", correct_dt, pset=pset, points=points, random_state=random_seed)
    toolbox.register("guard", tree_to_guard, pset=pset, points=points, random_state=random_seed)
    toolbox.register("clause", generate, pset=simple_pset, min_=1, max_=max_clause_depth, creator=creator)
    toolbox.register("complex_exp", initRepeatUpTo, list, toolbox.clause, n=max_clauses)
    toolbox.register("individual", tools.initIterate, creator.Individual_Guard, toolbox.complex_exp)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", parsimony_select)
    toolbox.register("mate", tools.cxUniform, indpb=cxpb_guard)
    toolbox.register("mutate", list_mutate, pset=simple_pset)

    pop = toolbox.population(mu_guard)

    if seeds is not None:
        for s in seeds:
            individual = creator.Individual_Guard([gp.PrimitiveTree.from_string(clause, pset) for clause in s])
            individual.fitness.values = toolbox.evaluate(individual)
            if individual.fitness.values[0] == 0:
                return individual
            pop.append(individual)

    best, best_guard = eaMuPlusLambda(pop, toolbox, mu_guard, lambda_guard, cxpb_guard, mutpb_guard, ngen_guard)
    return (best, best_guard)


if __name__ == "__main__":
    random.seed(0)
    points = pd.read_csv("test-guard3.csv")[["r0", "r1", "r2", "r3", "i0", "expected"]]
    data_s1_s0 = [
        [1234, 1000, 2, 2345, True],
        [2345, -500, 2, -9999, True],
        [2345, -500, 2, 1234, True],
        [1234, 1000, 0, -9999, False],
        [1234, 1000, 0, 2345, False],
        [1234, 1000, 1, -9999, False],
        [2345, -500, 0, 1234, False],
        [2345, -500, 1, 1234, False],
        [1234, 1000, 1, 2345, False],
        [2345, -500, 1, -9999, False],
        [2345, -500, 0, -9999, False],
        [1234, 1000, 1, 1234, False],
        [1234, 1000, 0, 1234, False],
        [2345, -500, 0, 2345, False],
        [2345, -500, 2, 2345, False],
        [2345, -500, 1, 2345, False],
        [1234, 1000, 2, 1234, False]
    ]
    df_s1_s0 = pd.DataFrame(data_s1_s0, columns=["r0", "r1", "r2", "i0", "guard"])
    df_s1_s0["target"] = "s0"

    data_s1_s2 = [
        [1234, 1000, 2, 2345, False],
        [2345, -500, 2, -9999, False],
        [2345, -500, 2, 1234, False],
        [1234, 1000, 0, -9999, False],
        [1234, 1000, 0, 2345, False],
        [1234, 1000, 1, -9999, False],
        [2345, -500, 0, 1234, False],
        [2345, -500, 1, 1234, False],
        [1234, 1000, 1, 2345, False],
        [2345, -500, 1, -9999, False],
        [2345, -500, 0, -9999, False],
        [1234, 1000, 1, 1234, True],
        [1234, 1000, 0, 1234, True],
        [2345, -500, 0, 2345, True],
        [2345, -500, 2, 2345, True],
        [2345, -500, 1, 2345, True],
        [1234, 1000, 2, 1234, True]
    ]
    df_s1_s2 = pd.DataFrame(data_s1_s2, columns=["r0", "r1", "r2", "i0", "guard"])
    df_s1_s2["target"] = "s2"

    data_s1_s1 = [
        [1234, 1000, 2, 2345, False],
        [2345, -500, 2, -9999, False],
        [2345, -500, 2, 1234, False],
        [1234, 1000, 0, -9999, True],
        [1234, 1000, 0, 2345, True],
        [1234, 1000, 1, -9999, True],
        [2345, -500, 0, 1234, True],
        [2345, -500, 1, 1234, True],
        [1234, 1000, 1, 2345, True],
        [2345, -500, 1, -9999, True],
        [2345, -500, 0, -9999, True],
        [1234, 1000, 1, 1234, False],
        [1234, 1000, 0, 1234, False],
        [2345, -500, 0, 2345, False],
        [2345, -500, 2, 2345, False],
        [2345, -500, 1, 2345, False],
        [1234, 1000, 2, 1234, False]
    ]
    df_s1_s1 = pd.DataFrame(data_s1_s1, columns=["r0", "r1", "r2", "i0", "guard"])
    df_s1_s1["target"] = "s1"

    df_all = pd.concat([df_s1_s0, df_s1_s1, df_s1_s2])

    train_data = df_all[df_all["guard"] == True].copy()

    pset = setup_pset(train_data)
    simple_pset = setup_simple_pset(train_data)

    # s = ["eq(i0, r0)", "ge(r2, 2)"]
    # individual = creator.Individual_Guard([gp.PrimitiveTree.from_string(clause, pset) for clause in s])
    # predict_dt(individual, train_data, pset)
    # print(len(train_data), fitness_dt(individual, train_data, pset))
    # print(tree_to_guard(individual, train_data, pset))

    train_data = train_data.drop("guard", axis=1, inplace=False)

    print(train_data)

    best, best_guards = run_gp(train_data, pset, simple_pset, 10, seeds=[], random_seed=0)


    # samples = list(range(100))
    # points = pd.DataFrame({k: [random.randint(0, 100) for _ in samples] for k in ["r1", "r2", "r3", "i0"]})
    # points["expected"] = (points["r2"] + points["i0"]) <= points["r3"]

    # # best = run_gp(points, 100, seeds=[["ne(i0, r0)", "ge(r2, 2)"]])
    # # best = run_gp(points, 100, seeds=[["le(add(r2, i0), r3)"]])
    # pset = setup_pset(points)
    # simple_pset = setup_simple_pset(points)
    # best, best_guard = run_gp(points, pset, simple_pset, 50, seeds=[], random_seed=10)
    print(best_guards)
    print(len(train_data) - best.fitness.values[0])
