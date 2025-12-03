"""
This module contains the z3 simplification code.
Long term, it would be nice to switch to sympy.
"""

import logging

import networkx as nx
import numpy as np
import z3
from deap import gp

logger = logging.getLogger(__name__)


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
        if labels[root] == "ne":
            c1, c2 = nested
            return c1 != c2
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


def simplify(individual, pset, types, creator):
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
