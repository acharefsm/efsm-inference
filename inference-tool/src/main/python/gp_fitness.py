"""
This module implements the GP fitness function and auxilliary functions.
"""

import logging
import traceback
from itertools import product
from math import isclose, sqrt
from numbers import Number

import numpy as np
import pandas as pd
from deap import gp
from enchant.utils import levenshtein
from gp_pset import is_null
from gp_repair import repair
from sklearn.tree import DecisionTreeClassifier, export_text

logger = logging.getLogger("main")


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
        logger.debug(f"Literal {individual}")
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


def fitness_bool(individual, points, pset):
    expected = points["expected"]
    args = points[[c for c in points.columns if c != "expected"]]
    match individual[0].name:
        case "True":
            return float("inf")
        case "False":
            return float("inf")
        case "and_":
            # Sum of the two child fitnesses, since both have to be true to satisfy the and
            c1, c2 = get_children(individual)
            # return fitness_bool(c1, points, pset) + fitness_bool(c2, points, pset)
            return max(fitness_bool(c1, points, pset), fitness_bool(c2, points, pset))
        case "or_":
            # Return the minimal child fitness, since only one must be true to satisfy the or
            c1, c2 = get_children(individual)
            return min(fitness_bool(c1, points, pset), fitness_bool(c2, points, pset))
        case "not_":
            # Return the fitness of the child, since negation doesn't really change the utility of the guard
            (c1,) = get_children(individual)
            return fitness_bool(c1, points, pset)
        case "eq":
            # Return the sum of the differences between individuals that are supposed to be equal but are not
            # plus one for every pair that were equal but shouldn't have been
            c1, c2 = [gp.compile(c, pset) for c in get_children(individual)]
            distances = args.apply(lambda row: c1(**row), axis=1) - args.apply(lambda row: c2(**row), axis=1)
            return distances.loc[expected].abs().sum() + (~distances.loc[~expected].astype(bool)).astype(int).sum()
        case "ne":
            # Inverse of eq
            c1, c2 = [gp.compile(c, pset) for c in get_children(individual)]
            distances = args.apply(lambda row: c1(**row), axis=1) - args.apply(lambda row: c2(**row), axis=1)
            return distances.loc[~expected].abs().sum() + (~distances.loc[expected].astype(bool)).astype(int).sum()
        case "gt":
            # Return the sum of the differences between individuals that are supposed to be > but are not
            # plus the sum of the differences between individuals that are not supposed to be > but are
            c1, c2 = [gp.compile(c, pset) for c in get_children(individual)]
            e1 = args.apply(lambda row: c1(**row), axis=1)
            e2 = args.apply(lambda row: c2(**row), axis=1)
            return ((e2 - e1) + 1).loc[expected & (e1 <= e2)].sum() + ((e1 - e2) + 1).loc[~expected & (e1 > e2)].sum()
        case "ge":
            # Return the sum of the differences between individuals that are supposed to be >= but are not
            # plus the sum of the differences between individuals that are not supposed to be >= but are
            c1, c2 = [gp.compile(c, pset) for c in get_children(individual)]
            e1 = args.apply(lambda row: c1(**row), axis=1)
            e2 = args.apply(lambda row: c2(**row), axis=1)
            return ((e2 - e1)).loc[expected & (e1 < e2)].sum() + ((e1 - e2) + 1).loc[~expected & (e1 >= e2)].sum()
        case "lt":
            # Return the sum of the differences between individuals that are supposed to be < but are not
            # plus the sum of the differences between individuals that are not supposed to be < but are
            c1, c2 = [gp.compile(c, pset) for c in get_children(individual)]
            e1 = args.apply(lambda row: c1(**row), axis=1)
            e2 = args.apply(lambda row: c2(**row), axis=1)
            return ((e1 - e2) + 1).loc[expected & (e1 >= e2)].sum() + ((e2 - e1) + 1).loc[~expected & (e1 < e2)].sum()
        case "le":
            # Return the sum of the differences between individuals that are supposed to be <= but are not
            # plus the sum of the differences between individuals that are not supposed to be <= but are
            c1, c2 = [gp.compile(c, pset) for c in get_children(individual)]
            e1 = args.apply(lambda row: c1(**row), axis=1)
            e2 = args.apply(lambda row: c2(**row), axis=1)
            return ((e1 - e2)).loc[expected & (e1 > e2)].sum() + ((e2 - e1) + 1).loc[~expected & (e1 <= e2)].sum()
        case _:
            raise ValueError(f"Could not evaluate {individual}")


def fitness(
    individual,
    points: pd.DataFrame,
    pset: gp.PrimitiveSet,
    bad: list,
    latent_vars_rows: list,
    creator,
    type_="continuous",
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
    if pset.ret == bool:
        if "guard" in points.columns and "expected" not in points.columns:
            points.rename(columns={"guard": "expected"}, inplace=True)
        return (fitness_bool(individual, points, pset),)
    try:
        ind = repair(individual, points, pset, creator)

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


reverse_op = {
    "eq": "ne",
    "ne": "eq",
    "lt": "ge",
    "gt": "le",
    "le": "gt",
    "ge": "lt",
}


def tree_to_guard(tree, feature_names):
    def reverse(operation):
        return reverse_op[operation[:2]] + operation[2:]

    def make_list(node):
        if tree.feature[node] == -2:  # leaf
            outcome = int(tree.value[node][0].argmax())
            return outcome
        else:
            return [
                [reverse(feature_names[tree.feature[node]]), make_list(tree.children_left[node])],
                [feature_names[tree.feature[node]], make_list(tree.children_right[node])],
            ]

    print(make_list(0))

    def find_paths(mylist):
        child1, child2 = mylist
        paths = []

        if isinstance(child1, list) and isinstance(child2, list):
            child1paths = find_paths(child1)
            for path in child1paths:
                paths.append(path)
            child2paths = find_paths(child2)
            for path in child2paths:
                paths.append(path)

        if isinstance(child1, str) and isinstance(child2, list):
            child2paths = find_paths(child2)
            for path in child2paths:
                paths.append(child1 + path)

        if isinstance(child1, str) and isinstance(child2, int) and child2 == 1:
            paths.append(child1)

        return paths

    if tree.feature[0] == -2:
        return []

    print(find_paths(make_list(0)))

    return make_list(0)


def predict_dt(individual, points: pd.DataFrame, pset, random_state=0):
    # reaslised that need all transitions from a state on a input all guards essentially I think actually no tho, we shall see

    expressions = {}
    for simple_expr in individual:
        expressions[str(simple_expr)] = points.drop("expected", axis=1).apply(
            lambda row: gp.compile(simple_expr, pset)(**row), axis=1
        )
        print(simple_expr)
        f = gp.compile(simple_expr, pset)
        expressions[str(simple_expr)] = points.drop("expected", axis=1).apply(lambda row: f(**row), axis=1)
    expressions = pd.DataFrame(expressions)

    X = expressions
    y = points["expected"]

    clf = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    clf.fit(X, y)

    print("\nDecision Tree Rules:")
    print(export_text(clf, feature_names=list(X.columns)))

    print(clf.classes_)

    tree_to_guard(clf.tree_, list(X.columns))
    return clf.predict(expressions)


def fitness_dt(individual, points: pd.DataFrame, pset, random_state=0):
    # reaslised that need all transitions from a state on a input all guards essentially I think actually no tho, we shall see

    predicted_outcome = predict_dt(individual, points, pset, random_state)

    print("\nPredicted transitions:")

    diff = points["expected"] == predicted_outcome

    print(diff)

    return (diff.sum(),)


def correct_dt(individual, points: pd.DataFrame, pset, random_state=0):
    predicted_outcome = predict_dt(individual, points, pset, random_state)

    return (points["expected"] == predicted_outcome).all()
