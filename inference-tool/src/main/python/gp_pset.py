import logging
import operator
import traceback

import numpy as np
import pandas as pd
from deap import gp
from gp_fitness import is_null

logger = logging.getLogger(__name__)


def add_consts_to_pset(individual, pset):
    try:
        for n in individual:
            if hasattr(n, "value") and str(creator.Individual([n])) not in pset.mapping:
                logger.debug("Adding", str(creator.Individual([n])))
                pset.addTerminal(n.value, type(n.value))
    except:
        logger.debug("Problem with", individual)
        logger.debug(traceback.format_exc())


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


def setup_pset(points: pd.DataFrame) -> gp.PrimitiveSet:
    try:
        return setup_pset_aux(points)
    except:
        logger.debug(traceback.format_exc())
