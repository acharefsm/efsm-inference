#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 3 11:47:57 2025

This module implements the `run_gp` function, which is the main access point for running GP.

@author: Luca Devlin Luca0414
@author: Michael Foster Jmafoster1
"""

import logging
import operator
import random
import sys
import traceback
import warnings
from math import isclose

import numpy as np
import pandas as pd
import z3
from deap import algorithms, base, creator, gp, tools
from gp_fitness import fitness, latent_variables
from gp_pset import setup_pset
from gp_repair import repair
from gp_reproduction import genHalfAndHalf, mutate, new_mate
from gp_simplification import simplify
from patsy import EvalEnvironment
from pyrsistent import pset

warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__.*")
logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def is_distinct(pop):
    seen = []
    for p in pop:
        if p in seen:
            return False
        seen.append(p)
    return True


def parsimony_select(individuals, k):
    return sorted(individuals, key=operator.attrgetter("fitness", "height"), reverse=True)[:k]


def sort_height(individual, training_set):
    height = individual.height
    if height > 0:
        return height
    latent_vars = latent_variables(individual, training_set)
    if len(latent_vars) > 0:
        return float("inf")
    return height


def run_gp(
    mut_prob,
    points: pd.DataFrame,
    pset,
    latent_vars_rows=None,
    max_init=1,
    max_depth=5,
    type_="continuous",
    mu=500,
    lamb=10,
    ngen=100,
    random_seed=0,
    seeds=None,
    bad=None,
):
    seeds = [] if seeds is None else seeds
    bad = [] if bad is None else bad

    points = points.replace({np.nan: None})
    random.seed(random_seed)

    toolbox = base.Toolbox()

    if latent_vars_rows is None or pset.ret == bool:
        latent_vars_rows = [[] for _ in range(len(points))]
    assert len(points) == len(
        latent_vars_rows
    ), f"Must have latent variable information for each row. {len(points)} vs. {len(latent_vars_rows)}"

    toolbox.register(
        "evaluate",
        fitness,
        points=points,
        pset=pset,
        bad=bad,
        latent_vars_rows=latent_vars_rows,
        type_=type_,
        creator=creator,
    )
    toolbox.register("height", sort_height, training_set=points)

    generators = {
        np.dtype("float64"): z3.Real,
        np.dtype("int64"): z3.Int,
        np.dtype("int32"): z3.Int,
        np.dtype("bool"): z3.Bool,
        pd.Int64Dtype(): z3.Int,
        pd.StringDtype(): z3.String,
    }

    types = {k: generators.get(points.dtypes[k], generators[points.dtypes[points.columns[-1]]]) for k in points}
    toolbox.register("simplify", simplify, pset=pset, types=types, creator=creator)
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=1, max_=max_init, creator=creator)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", new_mate, pset=pset, creator=creator)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", mutate, pset=pset, creator=creator)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    toolbox.register("repair", repair, data_points=points, pset=pset, creator=creator)

    pop = toolbox.population(n=mu)

    if len(seeds) > 0:
        logger.debug("SEEDS!")
        for seed in seeds:
            logger.debug(f"Trying to add {seed}")
            try:
                individual = creator.Individual(gp.PrimitiveTree.from_string(seed, pset))
                logger.debug(f"Fitness of {individual} is {fitness(individual, points, pset, bad, latent_vars_rows)}")
                if fitness(individual, points, pset, bad, latent_vars_rows) == (0,):
                    logger.debug("Found perfect individual!")
                    return individual
                pop.append(individual)
            except TypeError:
                logger.debug(f"Failed to add seed {seed}")
                logger.debug(traceback.format_exc())

    # for terms in pset.terminals.values():
    #     terms = [creator.Individual([i]) for i in terms]
    #     for t in terms:
    #         if t not in pop:
    #             logger.debug(str(t))
    #             pop.append(t)
    pop = make_distinct(pop)

    assert is_distinct(pop), "Population contains duplicated individuals."
    pop += toolbox.population(n=mu - len(pop))
    # pop = [toolbox.simplify(i) for i in pop]
    # pop = [toolbox.repair(ind) for ind in pop]
    pop = sorted(pop, key=lambda x: x.fitness.values)
    print("pop", [str(x) for x in pop])

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    try:
        # print("Calling eaMuPlusLambda")
        pop, _ = eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lamb,
            1 - mut_prob,
            mut_prob,
            ngen,
            stats=mstats,
            halloffame=None,
            verbose=False,
        )

        best = toolbox.simplify(toolbox.repair(pop[0]))
        best.fitness.values = toolbox.evaluate(best)
        return best
    except:
        logger.debug(traceback.format_exc())


def graph(best) -> ([int], [(int, int)], {int: str}):
    (nodes, edges, labels) = gp.graph(best)
    return (nodes, edges, {k: str(v) for k, v in labels.items()})


def get_types(points: pd.DataFrame) -> {str: str}:
    type_strings = {
        np.dtype("float64"): "Real",
        np.dtype("int64"): "Int",
        pd.Int64Dtype(): "Int",
        pd.StringDtype(): "String",
    }
    output_type = type_strings[points.dtypes[points.columns[-1]]]
    type_strings[np.dtype("O")] = output_type
    return {v: type_strings[t] for v, t in points.dtypes.iteritems()}


def fill_pop(more, individual, avoid=[], TIMEOUT=3):
    fillers = []
    for _ in range(more):
        fillers.append(individual())
    return fillers


def make_distinct(pop):
    new_pop = []
    for ind in pop:
        # Skip constants
        if len(ind) == 1 and isinstance(ind[0].value, ind[0].ret):
            continue
        if ind not in new_pop:
            new_pop.append(ind)
    assert is_distinct(new_pop)
    return new_pop


def eaMuPlusLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    r"""This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    population = sorted(population, key=lambda i: i.fitness.values + (toolbox.height(i),))[:mu]
    assert all(ind.fitness.valid for ind in population), "Invalid fitnesses in population after setting fitnesses!"

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **(record.copy() if record else {}))
    if verbose:
        logger.debug(logbook.stream)

    # Begin the generational process
    # print("Entering main loop")
    for gen in range(0, ngen):
        print("pop", [(str(x), round(x.fitness.values[0], 2)) for x in population])
        # print("gen", gen, "best", toolbox.simplify(toolbox.repair(population[0], )), population[0].fitness.values)
        if population[0].fitness.values == (0,):
            return population, logbook
        assert all(ind.fitness.valid for ind in population), "Invalid fitnesses in population"
        # logger.debug("\ngen", gen, "best", str(halloffame[0]))
        # logger.debug([str(x) for x in population])
        seen = {}
        for p in population:
            p = str(p)
            if p not in seen:
                seen[p] = 0
            seen[p] += 1

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        # offspring = [toolbox.simplify(i) for i in offspring]

        population += offspring
        population = make_distinct(population)
        assert is_distinct(population), "Population contains duplicates"
        population += toolbox.population(n=mu - len(population))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        population = sorted(population, key=lambda i: i.fitness.values + (toolbox.height(i),))[:mu]
        assert len(population) == mu, f"Population should contain {mu} individuals but contains {len(population)}."
        # Update the statistics with the new population
        # record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **(record.copy() if record else {}))
        if verbose:
            logger.debug(logbook.stream)
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

    return population, logbook


def from_string(string, pset):
    return gp.PrimitiveTree.from_string(string, pset)


def need_latent(points: pd.DataFrame, latent_vars_rows: list) -> bool:
    try:
        return need_latent_aux(points, latent_vars_rows)
    except:
        logger.debug(traceback.format_exc())


def set_to_na(training_set, latent_registers):
    dtypes = training_set.dtypes
    to_update = [{k: "<NA>" for k in i} for i in latent_registers]
    training_set.update(to_update)
    training_set.replace("<NA>", None, inplace=True)
    for k in training_set.dtypes.index:
        training_set[k] = training_set[k].astype(dtypes[k])


def need_latent_aux(points: pd.DataFrame, latent_vars_rows: list) -> bool:
    points = points.copy()
    set_to_na(points, latent_vars_rows)
    print(points)

    inputs = list(points.columns)[:-1]
    if len(set(points.iloc[:, -1])) <= 1:
        return False
    if inputs == [] and len(set(points.iloc[:, -1])) > 1:
        return True
    elif points.dtypes[list(points.columns)[-1]] == float:
        for _, group in points.groupby(inputs):
            outputs = group.iloc[:, -1]
            for o1 in outputs:
                for o2 in outputs:
                    if not isclose(o1, o2, abs_tol=1e-10):
                        return True
        return False
    else:
        return any(len(set(group.iloc[:, -1])) > 1 for _, group in points.groupby(inputs))


def shortcut_latent(points: pd.DataFrame) -> bool:
    return len(points.columns[:-1]) == 0 and len(set(points[points.columns[-1]])) > 1


if __name__ == "__main__":
    train = "test-guard.csv" if not len(sys.argv) > 1 else sys.argv[1]

    points = pd.read_csv(train)

    for col in points:
        if points.dtypes[col] == object:
            points[col] = points[col].astype("string")
    pset = setup_pset(points)
    print(pset.mapping)

    best = run_gp(1, points, pset, random_seed=3, seeds=[], mu=10, lamb=5, ngen=20, max_init=2)
    logger.debug(f"\nbest is {best}:{round(best.fitness.values[0],2)}")
    logger.debug(best.height)
