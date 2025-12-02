"""
This module implements all the functions necessary to produce and reproduce individuals,
including mating, mutation, and random initial generation.
"""

import random

from deap import gp


def new_mate(ind1, ind2, pset, creator):
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

    offspring1 = random.choice([new_mate_and(ind1, ind2), new_mate_or(ind1, ind2)])
    offspring2 = random.choice([new_mate_and(ind1, ind2), new_mate_or(ind1, ind2)])
    return offspring1, offspring2


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


def mutate(individual, pset, creator, MAX_MUTATIONS=3):
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


def generate(pset, min_, max_, condition, creator, type_=None, simp=None):
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


def genHalfAndHalf(pset, min_, max_, creator, type_=None, simp=None):
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
        creator=creator,
        type_=type_,
        simp=simp,
    )
