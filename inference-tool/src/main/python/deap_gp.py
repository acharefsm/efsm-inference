#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 3 11:47:57 2025

@author: Luca Devlin Luca0414
"""

import ast
import logging
import operator
import random
import re
import sys
import traceback
import warnings
from itertools import product
from math import isclose, sqrt
from numbers import Number

import networkx as nx
import numpy as np
import pandas as pd
import patsy
import statsmodels
import statsmodels.formula.api as smf
import z3
from deap import algorithms, base, creator, gp, tools
from enchant.utils import levenshtein
from patsy import EvalEnvironment
from pyrsistent import pset

warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__.*")
logging.basicConfig()

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def distance_between(expected, actual, type_="continuous"):
    if isinstance(expected, Number) and isinstance(actual, Number) and not is_null(actual):
        if type_ == "step":
            return float(expected != actual)
        return abs(expected - actual)
    if type(expected) == str and type(actual) == str:
        return levenshtein(expected, actual)
    if type(expected) == bool and type(actual) == bool:
        return float(expected != actual)
    # elif type(expected) != type(actual) or is_null(actual):
    #     print(f"BAD TYPES {type(expected)} and {type(actual)}")
    #     return float("inf")
    raise ValueError(
        f"Expected bool, int, float, or string type, not {expected}:{type(expected)} {actual}:{type(actual)}."
    )


def rmsd(errors: [float]) -> float:
    assert len(errors) > 0, "Cannot calculate RMSD of empty list."
    total = sum([float(d) ** 2 for d in errors])
    assert not is_null(total), f"sum of {errors} cannot be nan"
    mean = total / len(errors)
    return sqrt(mean)


def is_null(value):
    if isinstance(value, str):
        return value is None
    return value is None or value is pd.NA or np.isnan(value)


def find_smallest_distance(individual, pset, args, expected, latent_vars, verbose=False, type_="continuous"):
    if verbose:
        print(f"Looking for smallest distance between {individual} and {expected}")
    undefined_vars = [x for x in args if is_null(args[x])]
    consts = set()
    type_ = individual[0].ret

    # print("INDIVIDUAL", individual, "ARGS:", pset.arguments, "HEIGHT:", individual.height)
    try:
        height = individual.height
    except IndexError:
        # This individual is structurally invalid (e.g., a forest instead of a single tree).
        # It cannot be evaluated, so return an infinite distance.
        return float("inf")

    if height == 0 and individual.root.value not in pset.arguments:
        return distance_between(pset.ret(expected), individual.root.value, type_=type_)
    func = gp.compile(expr=individual, pset=pset)
    if not callable(func):
        return distance_between(pset.ret(expected), func, type_=type_)

    if len(undefined_vars) == 0:
        actual = func(**args)
        distance = distance_between(pset.ret(expected), actual, type_=type_)
        if isclose(distance, 0, abs_tol=1e-10):
            return 0
        if len(latent_vars) == 0:
            return distance

    consts = set([c.value for c in pset.terminals[type_] if type(c.value) == type_])
    assignments = [
        {k: v for k, v in zip(latent_vars, assignment)} for assignment in product(consts, repeat=len(latent_vars))
    ]

    min_distance = float("inf")
    for assignment in assignments:
        new_args = args.copy()
        new_args.update(assignment)
        try:
            actual = func(**new_args)
        except:
            logger.debug(f"Problem executing {individual} with {new_args}")
            logger.debug(traceback.format_exc())
        off_by = distance_between(expected, actual, type_=type_)
        if off_by == 0:
            return 0
        if off_by < min_distance:
            min_distance = off_by

    if isclose(min_distance, 0, abs_tol=1e-10):
        return 0
    assert not is_null(min_distance), "min_distance cannot be nan"
    return min_distance


def vars_in_tree(individual):
    _, _, labels = gp.graph(individual)
    return labels.values()


def latent_variables(individual, points, criterion=lambda points_c: any([is_null(v) for v in points_c])):
    undefined_at = [c for c in list(points) if criterion(points[c])]
    return list(set(undefined_at).intersection(vars_in_tree(individual)))


def all_vars_defined(individual, pset):
    return all([str(v) in pset.mapping for v in vars_in_tree(individual)])


def add_consts_to_pset(individual, pset):
    try:
        for n in individual:
            if hasattr(n, "value") and str(creator.Individual([n])) not in pset.mapping:
                logger.debug("Adding", str(creator.Individual([n])))
                pset.addTerminal(n.value, type(n.value))
    except:
        logger.debug("Problem with", individual)
        logger.debug(traceback.format_exc())


def process_row(args, type_="continuous"):
    (individual, (pset, ((inx, row), latent_vars))) = args
    try:
        return find_smallest_distance(individual, pset, row.iloc[:-1].to_dict(), row.iloc[-1], latent_vars, type_=type_)
    except:
        logger.debug(f"Problem executing {individual} with arguments\n{row}")
        logger.debug(traceback.format_exc())
        return float("inf")


def get_unused_vars(individual, points, latent_vars_rows, verbose=False):
    total_vars = list(points.columns)[:-1]
    undefined_vars = [item for items in latent_vars_rows for item in items]
    if verbose:
        print("Total vars:", total_vars)
        print("undefined_vars:", undefined_vars)
        print("vars in tree:", vars_in_tree(individual))
    return set(total_vars).difference(vars_in_tree(individual)).difference(undefined_vars)


def evaluate_candidate(
    individual, points: pd.DataFrame, pset: gp.PrimitiveSet, latent_vars_rows, verbose=False, type_="continuous"
) -> float:
    """
    Evaluate a candidate function for a set of function executions and aggregate the distances between the expected
    and actual values.

    :param individual: The candidate function to be evaluated.
    :type individual: TYPE
    :param points: The points with which to evaluate the individual.
    N.B. The expected output MUST be the last column in the table.
    :type points: pd.DataFrame
    :param pset: The set of primitives.
    :type pset: TYPE
    :return: The aggregated distance between expected and actual values.
    :rtype: float
    """
    if str(individual) in ["True", "False"]:
        return float("inf")
    assert len(points) == len(
        latent_vars_rows
    ), "Must have latent variable information for every row in the training set"
    if isinstance(individual, str):
        individual = creator.Individual(gp.PrimitiveTree.from_string(individual, pset))

    latent_vars_rows = [list(r) for r in latent_vars_rows]
    unused_vars = get_unused_vars(individual, points, latent_vars_rows)

    individual_rep = [individual for _ in range(len(points))]
    pset_rep = np.repeat(pset, len(points))
    data = zip(individual_rep, zip(pset_rep, zip(points.iterrows(), latent_vars_rows)))

    distances = [process_row(row, type_) for row in data]

    if verbose:
        print(f"Evaluating {individual}")
        print("  distances", distances)

    assert not any([is_null(x) for x in distances]), "no distance can be nan"

    copy = points.copy()
    copy["distances"] = distances

    mistakes = sum([x > 0 for x in distances])

    assert not is_null(rmsd(distances)), "rmsd(distances) cannot be nan (evaluate_candidate:145)"
    fitness = rmsd(distances) + mistakes

    assert not is_null(fitness), "fitness cannot be nan (evaluate_candidate:148)"

    return fitness + len(set(unused_vars).intersection(latent_variables(individual, points)))


def get_children(individual, index=0):
    node = individual[index]

    children = []
    pos = index + 1
    for _ in range(node.arity):
        child_slice = individual.searchSubtree(pos)
        child = individual[child_slice]
        children.append(gp.PrimitiveTree(child))

        pos = child_slice.stop
    return children


def fitness(
    individual, points: pd.DataFrame, pset: gp.PrimitiveSet, bad: list, latent_vars_rows: list, type_="continuous"
) -> float:
    """
    Determine the fitness of an individual based on its ability to account for a set of expected function executions.

    :param individual: The candidate function to be evaluated.
    :type individual: TYPE
    :param points: The points with which to evaluate the individual.
    N.B. The expected output MUST be the last column in the table.
    :type points: pd.DataFrame
    :param pset: The set of primitives.
    :type pset: TYPE
    :return: The fitness of the individnal.
    :rtype: float
    """
    if individual in bad:
        return (float("inf"),)
    try:
        ind = repair(individual, points, pset)

        if type_ == "step":
            score = score = evaluate_candidate(ind, points, pset, latent_vars_rows, type_=type_)
        elif type_ == "recursive" and len(ind) > 2 and ind[0].arity == 2 and ind[0].ret == bool:
            child1, child2 = get_children(ind)
            score1 = fitness(
                individual=child1,
                points=points,
                pset=pset,
                bad=[],
                latent_vars_rows=latent_vars_rows,
                type_="recursive",
            )
            score2 = fitness(
                individual=child2,
                points=points,
                pset=pset,
                bad=[],
                latent_vars_rows=latent_vars_rows,
                type_="recursive",
            )
            score = score1[0] + score2[0]
        else:
            score = evaluate_candidate(ind, points, pset, latent_vars_rows, type_="continuous")
        newline = "\n  "
        assert not is_null(score), f"Score cannot be nan\nPSET:\n  {newline.join(sorted(list(pset.mapping)))}"
        return (score,)
    except:
        # logger.debug(f"Problem evaluating candidate {individual}")
        logger.debug(traceback.format_exc())
        return (float("inf"),)


def correct(individual, points: pd.DataFrame, pset: gp.PrimitiveSet, latent_vars_rows: list) -> bool:
    """
    Does the candidate function perfectly reproduce the expected executions, assuming any latent variables hold the
    correct values upon evaluation?

    :param individual: The candidate function to be evaluated.
    :type individual: TYPE
    :param points: The points with which to evaluate the individual.
    N.B. The expected output MUST be the last column in the table.
    :type points: pd.DataFrame
    :param pset: The set of primitives.
    :type pset: TYPE
    :return: The fitness of the individnal.
    :rtype: bool
    """

    assert len(points) == len(
        latent_vars_rows
    ), "Must have latent variable information for every row in the training set"
    if isinstance(individual, str):
        individual = creator.Individual(gp.PrimitiveTree.from_string(individual, pset))

    for (inx, row), latent_vars in zip(points.iterrows(), latent_vars_rows):
        try:
            best = find_smallest_distance(individual, pset, row.iloc[:-1].to_dict(), row[-1], latent_vars)
            if best > 0:
                return False
        except:
            logger.debug(f"Problem executing {individual} with arguments\n{row}")
    return True


def setup_full_pset(points: pd.DataFrame) -> gp.PrimitiveSet:
    """
    Set up and return the primitive set containing all operators.

    :param points: The sample function executions with expected outputs.
    N.B. The expected output MUST be the last column in the dataframe.
    N.B. Strings will, by default, appear as objects, so will be indistinguishable from latent registers.
    They MUST be converted explicitly using `.astype('string')` before calling this method.
    :type points: pd.DataFrame
    :return: The primitive set.
    :rtype: gp.PrimitiveSet
    """
    generators = {
        np.dtype("float64"): float,
        np.dtype("int64"): int,
        np.dtype("bool"): bool,
        pd.Int64Dtype(): int,
        pd.StringDtype(): str,
    }
    output_type = generators[points.dtypes[points.columns[-1]]]
    generators[np.dtype("O")] = output_type

    assert output_type in {int, float, str, bool}, f"Bad output type {output_type}"

    datatypes = {}
    for col in points:
        datatypes[col] = type(points[col].tolist()[0])

    pset = gp.PrimitiveSet("MAIN", len(datatypes))

    rename = {f"ARG{i}": col for i, col in enumerate(datatypes)}
    pset.renameArguments(**rename)

    assert all([type(t.value) in {int, str, float} for t in pset.mapping.values() if hasattr(t, "value")]), "Bad type"

    # Add literal terminals
    # print("-" * 80)
    # print("TERMINALS")
    # print(names)
    # print("printing full pset")
    for v, typ in datatypes.items():
        # print("----------", v, typ)
        # assert typ in {int, str, float, bool}, f"Bad pset terminal type {typ}"
        term_set = set(points[v])
        for term in term_set:
            if not is_null(term):
                pset.addTerminal(typ(term))

    types = [(t.value, type(t.value)) for t in pset.mapping.values() if hasattr(t, "value")]
    assert all(
        [type(t.value) in {int, str, float, bool} for t in pset.mapping.values() if hasattr(t, "value")]
    ), f"Bad type: {types}"

    pset.addTerminal("", str)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.__le__, 2)
    pset.addPrimitive(operator.__ge__, 2)
    pset.addPrimitive(operator.__lt__, 2)
    pset.addPrimitive(operator.__gt__, 2)
    pset.addPrimitive(operator.__eq__, 2)
    pset.addPrimitive(operator.__and__, 2)
    pset.addPrimitive(operator.__or__, 2)
    pset.addPrimitive(operator.__not__, 2)
    return pset


def setup_pset(points: pd.DataFrame) -> gp.PrimitiveSet:
    try:
        return setup_pset_aux(points)
    except:
        logger.debug(traceback.format_exc())


class PrimitiveSetTyped(gp.PrimitiveSetTyped):
    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if type_ == ret_type:
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, gp.Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if prim.ret == type_:
                dict_[type_].append(prim)


def setup_pset_aux(points: pd.DataFrame) -> gp.PrimitiveSet:
    """
    Set up and return the primitive set. Currently supported operators are +, -, *, and /.

    :param points: The sample function executions with expected outputs.
    N.B. The expected output MUST be the last column in the dataframe.
    N.B. Strings will, by default, appear as objects, so will be indistinguishable from latent registers.
    They MUST be converted explicitly using `.astype('string')` before calling this method.
    :type points: pd.DataFrame
    :return: The primitive set.
    :rtype: gp.PrimitiveSet
    """

    generators = {
        np.dtype("float64"): float,
        np.dtype("int64"): int,
        np.dtype("int32"): int,
        np.dtype("bool"): bool,
        np.dtype("O"): str,
        pd.Int64Dtype(): int,
        pd.StringDtype(): str,
    }
    output_type = generators[points.dtypes[points.columns[-1]]]
    # generators[np.dtype("O")] = output_type

    assert output_type in {int, float, str, bool}, f"Bad output type {output_type}"

    types = points.dtypes.to_dict()
    names = list(types)
    datatypes = [generators[types[v]] for v in names]
    assert all([t in {int, float, str, bool} for t in datatypes]), f"Bad datatype {output_type}"

    pset = PrimitiveSetTyped("MAIN", datatypes[:-1], output_type)

    rename = {f"ARG{i}": col for i, col in enumerate(names)}
    pset.renameArguments(**rename)

    assert all([type(t.value) in {int, str, float} for t in pset.mapping.values() if hasattr(t, "value")]), "Bad type"

    # Add literal terminals
    for v, typ in zip(names, datatypes):
        assert typ in {int, str, float, bool}, "Bad pset terminal type {typ}"
        term_set = set(points[v])
        # print("----------", v, typ)
        for term in term_set:
            if not is_null(term):
                pset.addTerminal(typ(term), typ)
                # print(typ(term))

    types = [(t.value, type(t.value)) for t in pset.mapping.values() if hasattr(t, "value")]
    assert all(
        [type(t.value) in {int, str, float, bool} for t in pset.mapping.values() if hasattr(t, "value")]
    ), f"Bad type: {types}"

    if output_type == int:
        pset.addPrimitive(operator.add, [int, int], int)
        pset.addPrimitive(operator.sub, [int, int], int)
        pset.addPrimitive(operator.mul, [int, int], int)
    elif output_type == bool:
        pset.addPrimitive(operator.__le__, [int, int], bool)
        pset.addPrimitive(operator.__ge__, [int, int], bool)
        pset.addPrimitive(operator.__lt__, [int, int], bool)
        pset.addPrimitive(operator.__gt__, [int, int], bool)
        pset.addPrimitive(operator.__eq__, [int, int], bool)
        pset.addPrimitive(operator.__and__, [bool, bool], bool)
        pset.addPrimitive(operator.__or__, [bool, bool], bool)
        pset.addPrimitive(operator.__not__, [bool], bool)
        if int in datatypes:
            pset.addPrimitive(operator.add, [int, int], int)
            pset.addPrimitive(operator.sub, [int, int], int)
            pset.addPrimitive(operator.mul, [int, int], int)
    else:
        raise ValueError(f"Invalid output type {output_type}.")

    assert all(
        [type(t.value) in {int, str, float, bool} for t in pset.mapping.values() if hasattr(t, "value")]
    ), "Bad type"
    return pset


# def if_then_else(condition: bool, out1: float, out2: float) -> float:
#     return out1 if condition else out2


def split(individual):
    if len(individual) > 1:
        terms = []
        # Recurse over children if add/sub
        if individual[0].name in ["add", "sub"]:
            terms.extend(
                split(
                    creator.Individual(
                        gp.PrimitiveTree(
                            individual[individual.searchSubtree(1).start : individual.searchSubtree(1).stop]
                        )
                    )
                )
            )
            terms.extend(split(creator.Individual(gp.PrimitiveTree(individual[individual.searchSubtree(1).stop :]))))
        else:
            terms.append(individual)
        return terms
    return [individual]


op_mapp = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
}


def infix_to_prefix2(expr):
    """
    Convert an infix arithmetic expression like '(r0 + r2 * 10)'
    into DEAP prefix notation: add(r0, mul(r2, 10))
    """
    tree = ast.parse(expr, mode="eval")
    return recurse(tree.body)


def recurse(node):
    if isinstance(node, ast.BinOp):
        op = op_mapp[type(node.op)]
        return f"{op}({recurse(node.left)}, {recurse(node.right)})"
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"-{recurse(node.operand)}"
        return recurse(node.operand)
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        # Handle inner I(...) wrappers
        if isinstance(node.func, ast.Name) and node.func.id == "I":
            return recurse(node.args[0])
        elif isinstance(node.func, ast.Name) and node.func.id in {"add", "sub", "mul", "div", "pow"}:
            args = ", ".join(recurse(a) for a in node.args)
            return f"{node.func.id}({args})"
        raise NotImplementedError(node)
    else:
        raise NotImplementedError(node)


def repair(individual, data_points, pset):
    if data_points.iloc[:, -1].dtype == "int64":
        eq = f"y ~ {' + '.join(str(x) for x in split(individual))}"
        data_points.rename(columns={data_points.columns[-1]: "y"}, inplace=True)
        data_points = data_points.astype(float)

        pattern_mul = r"mul\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
        while re.search(pattern_mul, eq):
            eq = re.sub(pattern_mul, r"(\1 * \2)", eq)

        pattern_add = r"add\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
        while re.search(pattern_add, eq):
            eq = re.sub(pattern_add, r"(\1 + \2)", eq)

        pattern_sub = r"sub\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
        while re.search(pattern_sub, eq):
            eq = re.sub(pattern_sub, r"(\1 - \2)", eq)

        # If both sides constant e.g 1000 + 10 then evaluate and replace with constant e.g 1010
        match = re.search(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\*/\+\-])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", eq)
        if match:
            left, op, right = match.groups()
            result = eval(f"{left} {op} {right}")
            # print(result)
            eq = re.sub(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\*/\+\-])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", str(result), eq)
            if re.match(r"^\s*y\s*~\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*$", eq):
                return individual

        # Add I() to right/left side constant and left/right side variable e.g 1000 + r2
        eq = re.sub(r"(\b[A-Za-z_]+\d*\b)\s*([\*/\+\-])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", r"I(\1 \2 \3)", eq)
        eq = re.sub(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\*/\+\-])\s*(\b[A-Za-z_]+\d*\b)", r"I(\1 \2 \3)", eq)
        # constant in parentheses
        eq = re.sub(r"(\b[A-Za-z_]+\d*\b)\s*([\*/\+\-])\s*\((?<![A-Za-z_])(-?\d+\.?\d*)\)", r"I(\1 \2 \3)", eq)
        eq = re.sub(r"\((?<![A-Za-z_])(-?\d+\.?\d*)\)\s*([\*/\+\-])\s*(\b[A-Za-z_]+\d*\b)", r"I(\1 \2 \3)", eq)

        # Add I() to right/left side constant and left/right side expression I() e.g 1000 * I(10 + r2)
        eq = re.sub(r"\(?\s*I\(([^()]*)\)\s*\)?\s*([\+\-\*/])\s*(?<![A-Za-z_])(\d+\.?\d*)", r"I(\1 \2 \3)", eq)
        eq = re.sub(r"(?<![A-Za-z_])(\d+\.?\d*)\s*([\+\-\*/])\s*\(?\s*I\(([^()]*)\)\s*\)?", r"I(\1 \2 \3)", eq)

        # Add I() to right/left side constant and left/right side expression () e.g 1000 * (r0 + r2)
        eq = re.sub(r"(\([^()]+\))\s*([\+\-\*/])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", r"I((\1) \2 \3)", eq)
        eq = re.sub(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\+\-\*/])\s*(\([^()]+\))", r"I(\1 \2 (\3))", eq)

        env = EvalEnvironment.capture()

        try:
            # Create model, fit (run) it, give estimates from it]
            model = smf.ols(eq, data_points, eval_env=env)
            res = model.fit()

            if "Intercept" in res.params:
                eqn = f"{int(round(res.params['Intercept']))}"
            else:
                eqn = "0"
            for term, coefficient in res.params.items():
                if term != "Intercept":
                    if ":" in term:
                        parts = term.split(":")
                        term = "mul(" + ", ".join(parts) + ")"
                    term = re.sub(r"I\((.*?)\)", r"(\1)", term)

                    term = infix_to_prefix2(term)

                    eqn = f"add({eqn}, mul({int(round(coefficient))}, {term}))"
            repaired = type(individual)(gp.PrimitiveTree.from_string(eqn, pset))
            return repaired
        except (
            UnboundLocalError,
            SyntaxError,
            TypeError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
            statsmodels.tools.sm_exceptions.MissingDataError,
            patsy.PatsyError,
            np.core._exceptions._UFuncOutputCastingError,
            np.linalg.LinAlgError,
        ) as e:
            return individual
    else:
        return individual


def choose_terminal(pset, type_, prob=0.7):
    try:
        variables = [t for t in pset.terminals[type_] if t.name.startswith("ARG")]
        return random.choice(variables)
    except IndexError:
        constants = [t for t in pset.terminals[type_] if t not in variables]
        return random.choice(constants)


def mutateByTerminal(individual, pset):
    if len(individual) < 2:
        return (individual,)

    index = random.randrange(1, len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    term = choose_terminal(pset, node.ret)

    if gp.isclass(term):
        term = term()
    individual[slice_] = [term]

    return (individual,)


def mutateByCommute(individual, pset):
    if len(individual) < 2:
        return (individual,)

    possible_nodes = [(i, node) for i, node in enumerate(individual) if node.arity > 1]
    if len(possible_nodes) == 0:
        return (individual,)
    i, node = random.choice(possible_nodes)
    individual[i].args.reverse()
    return (individual,)


def mutateByFuzz(individual, pset):
    terminals = [(i, node) for i, node in enumerate(individual) if node.arity == 0]

    index, node = random.choice(terminals)

    term = random.choice(pset.terminals[node.ret])
    if gp.isclass(term):
        term = term()
    individual[index] = term

    return (individual,)


def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return (individual,)

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choose_terminal(pset, arg_type)
            if gp.isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position : position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return (individual,)


def mutate(individual, pset, MAX_MUTATIONS=3):
    mutations = 0
    newNode = creator.Individual(gp.PrimitiveTree.from_string(str(individual), pset))
    mutate = True
    while mutate and mutations < MAX_MUTATIONS:
        mutations += 1
        mutate = random.choice([True, False])
        if len(individual) < 2:
            newNode = mutInsert(newNode, pset)[0]
            continue

        op = random.choice(range(6))
        if op == 0:
            # HVL SUB
            newNode = gp.mutNodeReplacement(newNode, pset)[0]
            # logger.debug("Mutating", individual, "by substitution", newNode)
        if op == 1:
            # HLV DEL
            newNode = gp.mutShrink(newNode)[0]
            # logger.debug("Mutating", individual, "by deletion", newNode)
        if op == 2:
            # HVL INS
            newNode = mutInsert(newNode, pset)[0]
            # logger.debug("Mutating", individual, "by insertion", newNode)
        if op == 3:
            # Reverse this.children if they have the same return type, e.g. (x - y) -> (y - x)
            newNode = mutateByCommute(newNode, pset)[0]
            # logger.debug("Mutating", individual, "by commutation", newNode)
        if op == 4:
            # mutate by replacing a random node with a terminal
            newNode = mutateByTerminal(newNode, pset)[0]
            # logger.debug("Mutating", individual, "by terminal swap", newNode)
        if op == 5:
            # fuzz a terminal
            newNode = mutateByFuzz(newNode, pset)[0]
            # logger.debug("Mutating", individual, "by fuzzing", newNode)
    return (newNode,)


def gen_terminal(expr, pset, type_):
    try:
        term = choose_terminal(pset, type_)
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError(
            "The gp.generate function tried to add "
            "a terminal of type '%s', but there is "
            "none available." % (type_,)
        ).with_traceback(traceback)
    if gp.isclass(term):
        term = term()
    expr.append((term))


def gen_primitive(expr, pset, type_, stack, depth):
    try:
        prim = random.choice(pset.primitives[type_])
        expr.append(prim)
        for arg in reversed(prim.args):
            stack.append((depth + 1, arg))
    except IndexError:
        gen_terminal(expr, pset, type_)


def generate(pset, min_, max_, condition, type_=None, simp=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        d, t = stack.pop()
        if condition(height, d):
            gen_terminal(expr, pset, t)
        else:
            gen_primitive(expr, pset, t, stack, d)
    if simp is not None:
        nodes, edges, labels = gp.graph(expr)
        types = [type(v) for v in labels.values()]
        assert all(
            [t in {int, str, float, bool} for t in types]
        ), f"Bad type {[(v, type(v)) for v in labels.values()]} in {str(creator.Individual(expr))}\n Type was {type_}"
        return simp(expr)
    return expr


def genHalfAndHalf(pset, min_, max_, type_=None, simp=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """

    def genGrow(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)

    def genFull(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(
        pset,
        min_,
        max_,
        condition=random.choice([genGrow, genFull]),
        type_=type_,
        simp=simp,
    )


def is_distinct(pop):
    seen = []
    for p in pop:
        if p in seen:
            return False
        else:
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


def new_mate(ind1, ind2, pset):
    def new_mate_and(ind1, ind2):
        try:
            return creator.Individual.from_string("and_(" + str(ind1) + ", " + str(ind2) + ")", pset)
        except Exception as e:
            print(e)
            return ind1

    def new_mate_or(ind1, ind2):
        try:
            return creator.Individual.from_string("or_(" + str(ind1) + ", " + str(ind2) + ")", pset)
        except Exception as e:
            for name, primitive in pset.primitives.items():
                print(f"Type: {name}")
                for prim in primitive:
                    print(prim.name, prim.args, prim.ret, prim.arity)
            for name, terminal in pset.terminals.items():
                print(f"  Type: {name}")
                for term in terminal:
                    print(f"    {term.value}")
            print(e)
            print(pset.ret)
            return ind1

    if pset.ret != bool:
        return gp.cxOnePoint(ind1, ind2)
    else:
        offspring1 = random.choice([new_mate_and(ind1, ind2), new_mate_or(ind1, ind2)])
        offspring2 = random.choice([new_mate_and(ind1, ind2), new_mate_or(ind1, ind2)])

        return offspring1, offspring2


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
    seeds=[],
    bad=[],
):
    # print("Running GP")
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
    toolbox.register("simplify", simplify, pset=pset, types=types)
    toolbox.register(
        "expr",
        genHalfAndHalf,
        pset=pset,
        min_=1,
        max_=max_init,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", new_mate, pset=pset)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", mutate, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    toolbox.register("repair", repair, data_points=points, pset=pset)
    print(points)

    # print("Generating initial population")

    pop = toolbox.population(n=mu)
    # print(f"Fitness of pop[0] {pop[0]} is {pop[0].fitness.values}")

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
                # logger.debug("Type error.")
                logger.debug(traceback.format_exc())
                # logger.debug(pset.mapping)
                # assert False
                # pass

    # for terms in pset.terminals.values():
    #     terms = [creator.Individual([i]) for i in terms]
    #     for t in terms:
    #         if t not in pop:
    #             logger.debug(str(t))
    #             pop.append(t)
    # logger.debug("Initial population:", len(pop), [str(p) for p in pop])
    pop = make_distinct(pop)
    # logger.debug("\nDistinct Initial population:", len(pop), [str(p) for p in pop])

    assert is_distinct(pop), "Population contains duplicated individuals."
    pop += toolbox.population(n=mu - len(pop))
    # pop = [toolbox.simplify(i) for i in pop]
    # pop = [toolbox.repair(ind) for ind in pop]
    pop = sorted(pop, key=lambda x: x.fitness.values)
    print("pop", [str(x) for x in pop])

    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    try:
        # print("Calling eaMuPlusLambda")
        pop, log = eaMuPlusLambda(
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
        # print("Finished GP")

        # for p in pop:
        #     logger.debug(str(p), p.fitness.values, toolbox.height(p))

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


def to_z3(tree, labels, types):
    def _make_tuple(tree, root, _parent):
        # Get the neighbors of `root` that are not the parent node. We
        # are guaranteed that `root` is always in `tree` by construction.
        children = set(tree[root]) - {_parent}
        if len(children) == 0:
            if labels[root] in types:
                return types[labels[root]](labels[root])
            else:
                return labels[root]

        nested = tuple(_make_tuple(tree, v, root) for v in children)
        if labels[root] == "lt":
            c1, c2 = nested
            return c1 < c2
        if labels[root] == "ge":
            c1, c2 = nested
            return c2 <= c1
        if labels[root] == "gt":
            c1, c2 = nested
            return c2 < c1
        if labels[root] == "le":
            c1, c2 = nested
            return c1 <= c2
        if labels[root] == "eq":
            c1, c2 = nested
            return c1 == c2
        if labels[root] == "and_":
            c1, c2 = nested
            return z3.And(c1, c2)
        if labels[root] == "or_":
            c1, c2 = nested
            return z3.Or(c1, c2)
        if labels[root] == "not_":
            (c1,) = nested
            return z3.Not(c1)
        if labels[root] == "add":
            c1, c2 = nested
            return c1 + c2
        elif labels[root] == "sub":
            c1, c2 = nested
            return c1 - c2
        elif labels[root] == "mul":
            c1, c2 = nested
            return c1 * c2
        elif labels[root] == "div":
            c1, c2 = nested
            try:
                return c1 / c2
            except ZeroDivisionError:
                return float("inf")
        elif labels[root] == "id":
            return nested[0]
        else:
            raise ValueError(f"Invalid operator {root}")

    # Do some sanity checks on the input.
    root = 0  # By construction of GP individuals
    if not nx.is_tree(tree):
        raise nx.NotATree("provided graph is not a tree")
    if root not in tree:
        raise nx.NodeNotFound(f"Graph {tree} contains no node {root}")

    return _make_tuple(tree, root, None)


def make_binary(fun, children):
    if len(children) == 1:
        c1 = to_exp_string(children[0])
        return f"{fun}({c1})"
    if len(children) == 2:
        c1 = to_exp_string(children[0])
        c2 = to_exp_string(children[1])
        return f"{fun}({c1}, {c2})"
    else:
        c1 = to_exp_string(children[0])
        c2 = make_binary(fun, children[1:])
        return f"{fun}({c1}, {c2})"


def to_exp_string(exp):
    if str(exp.decl()) == "And":
        return make_binary("and_", exp.children())
    if str(exp.decl()) == "Or":
        return make_binary("or_", exp.children())
    if str(exp.decl()) == "Not":
        return make_binary("not_", exp.children())
    if str(exp.decl()) == ">=":
        return make_binary("ge", exp.children())
    if str(exp.decl()) == "<":
        return make_binary("lt", exp.children())
    if str(exp.decl()) == "<=":
        return make_binary("le", exp.children())
    if str(exp.decl()) == ">":
        return make_binary("gt", exp.children())
    if str(exp.decl()) == "==":
        return make_binary("eq", exp.children())
    if str(exp.decl()) == "+":
        return make_binary("add", exp.children())
    elif str(exp.decl()) == "-":
        return make_binary("sub", exp.children())
    elif str(exp.decl()) == "*":
        return make_binary("mul", exp.children())
    elif str(exp.decl()) == "/":
        return make_binary("div", exp.children())
    elif type(exp) == z3.RatNumRef:
        x2 = exp.as_fraction()
        return str(float(x2.numerator) / float(x2.denominator))
    else:
        return str(exp)


def from_z3(exp, pset):
    return gp.PrimitiveTree.from_string(to_exp_string(exp), pset)


def to_z3_string(individual, dtypes):
    generators = {
        np.dtype("float64"): z3.Real,
        np.dtype("int64"): z3.Int,
        np.dtype("bool"): z3.Bool,
        pd.Int64Dtype(): z3.Int,
        pd.StringDtype(): z3.String,
    }

    types = {k: generators[d] for k, d in dtypes.to_dict().items()}

    nodes, edges, labels = gp.graph(individual)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    z3_exp = to_z3(g, labels, types)
    try:
        return z3_exp.sexpr()
    except AttributeError:
        return str(z3_exp)


op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div"}  # use operator.truediv in pset


def infix_to_prefix(expr):
    """
    Convert a simple infix expression (like "r1 + i0") to DEAP prefix notation.
    Only works for single-level binary operations (can be extended for more).
    """
    # Remove spaces
    expr = expr.replace(" ", "")

    # Match simple binary operation: operand1 operator operand2
    match = re.match(r"(\w+)([+\-*/])(\w+)", expr)
    if not match:
        raise ValueError(f"Expression '{expr}' not recognized")

    op1, operator_symbol, op2 = match.groups()

    prefix_op = op_map[operator_symbol]
    return f"{prefix_op}({op1},{op2})"


def to_nodes_edges_labels(exp, pset, rename={}):
    # print("=" * 80)
    # print(exp)
    # for k, v in pset.mapping.items():
    #     print(f"{k}: {v}")
    # print("=" * 80)
    try:
        exp = creator.Individual(gp.PrimitiveTree.from_string(exp, pset))
    except:
        exp = creator.Individual(gp.PrimitiveTree.from_string(infix_to_prefix(exp), pset))
    rename = {k: gp.Terminal(v, None, object) for k, v in rename.items()}
    for inx, element in enumerate(exp):
        if isinstance(element, gp.Terminal) and element.format() in rename:
            exp[inx] = rename[element.format()]
    assert "r_b" not in str(exp), f"{exp}: {rename}"
    return gp.graph(exp)


def simplify(individual, pset, types):
    # Duplicate the individual
    individual = creator.Individual(individual.copy())
    nodes, edges, labels = gp.graph(individual)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    try:
        z3_exp = to_z3(g, labels, types)
        if type(z3_exp) in {int, float, str, bool}:
            return creator.Individual(gp.PrimitiveTree([gp.Terminal(z3_exp, z3_exp, type(z3_exp))]))
        elif type(z3_exp) == np.dtype("int64"):
            return creator.Individual(gp.PrimitiveTree([gp.Terminal(int(z3_exp), int(z3_exp), int)]))
        return creator.Individual(from_z3(z3.simplify(z3_exp), pset))
    except:
        logger.debug("Problem when simplifying", individual, type(individual))
        # logger.debug("z3_exp", z3_exp, type(z3_exp))
        logger.debug(
            "PSET",
            [(v.value, type(v.value)) for v in pset.mapping.values() if hasattr(v, "value")],
        )
        logger.debug("types", types)
        logger.debug("labels", [(v, type(v)) for v in labels.values()])
        logger.debug(traceback.format_exc())


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
    assert all([ind.fitness.valid for ind in population]), "Invalid fitnesses in population after setting fitnesses!"

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
        assert all([ind.fitness.valid for ind in population]), "Invalid fitnesses in population"
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
        return any([len(set(group.iloc[:, -1])) > 1 for _, group in points.groupby(inputs)])


def shortcut_latent(points: pd.DataFrame) -> bool:
    return len(points.columns[:-1]) == 0 and len(set(points[points.columns[-1]])) > 1


if __name__ == "__main__":
    train = "test-guard.csv" if not len(sys.argv) > 1 else sys.argv[1]

    points = pd.read_csv(train)

    for col in points:
        if points.dtypes[col] == object:
            points[col] = points[col].astype("string")
    pset = setup_pset(points)

    best = run_gp(
        1,
        points,
        pset,
        random_seed=3,
        seeds=[],
        mu=10,
        lamb=5,
        ngen=10,
        latent_vars_rows=[() for i in range(len(points))],
    )
    logger.debug(f"\nbest is {best}:{round(best.fitness.values[0],2)}")
    logger.debug(best.height)
