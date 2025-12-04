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
from gp_fitness import fitness_dt
from gp_pset import setup_pset, setup_simple_pset
from gp_repair import repair
from gp_reproduction import genHalfAndHalf, mutateByCommute, mutateByFuzz, mutateByNegate, mutateByTerminal, mutInsert
from gp_simplification import simplify

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def fitness(individual, pset, points):
    """
    This is a stub. Wait for Luca's implementation.
    """
    return (1,)


def mutate(individual, pset, creator, MAX_MUTATIONS=3):
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


def list_mutate(expression, pset, creator):
    return (
        creator.Individual([mutate(x, pset, creator)[0] if random.choice([True, False]) else x for x in expression]),
    )


def generate(pset, min_, max_, creator, type_=None, simp=None):
    return gp.PrimitiveTree(genHalfAndHalf(pset, min_, max_, creator, type_=None, simp=None))


def run_gp(
    points: pd.DataFrame,
    ngen: int,
    mu=10,
    lambda_=5,
    max_clauses: int = 4,
    max_clause_depth: int = 4,
    cxpb=0.5,
    mutpb=0.5,
):
    pset = setup_pset(points)
    simple_pset = setup_simple_pset(points)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", fitness_dt, pset=pset, points=points)
    toolbox.register("clause", generate, pset=simple_pset, min_=1, max_=max_clause_depth, creator=creator)
    toolbox.register("complex_exp", tools.initRepeat, list, toolbox.clause, n=max_clauses)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.complex_exp)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", list_mutate, pset=simple_pset, creator=creator)

    ind = toolbox.individual()

    pop, _ = algorithms.eaMuPlusLambda(toolbox.population(mu), toolbox, mu, lambda_, cxpb, mutpb, ngen)
    return max(pop, key=lambda ind: ind.fitness.values)


if __name__ == "__main__":
    points = pd.read_csv("test-guard2.csv")
    best = run_gp(points, 100)
    print([str(x) for x in best])
    print(best.fitness.values)
