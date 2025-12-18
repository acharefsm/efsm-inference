"""
This module contains the functions necessary to set up the pset.
"""

import logging
import operator
import traceback

import numpy as np
import pandas as pd
from deap import gp

logger = logging.getLogger(__name__)


def is_null(value):
    if isinstance(value, str):
        return value is None
    return value is None or value is pd.NA or np.isnan(value)


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
    for v, typ in datatypes.items():
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
    pset.addPrimitive(operator.__ne__, 2)
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

    def __init__(self, name, in_types, ret_type, prefix="ARG", weights=None):
        super().__init__(name, in_types, ret_type, prefix)
        if weights is None:
            weights = [1] * len(in_types)
        elif len(weights) != len(in_types):
            raise ValueError(f"Please specify a weight for each of the {len(in_types)} arguments.")
        self.weights = {f"{prefix}{index}": weight for index, weight in enumerate(weights)}

    def renameArguments(self, **kargs):
        """
        Rename function arguments with new names from *kargs*.
        """
        super().renameArguments(**kargs)
        for old_name in kargs:
            if old_name in self.weights:
                new_name = kargs[old_name]
                self.weights[new_name] = self.weights[old_name]
                del self.weights[old_name]

    def addPrimitive(self, primitive, in_types, ret_type, name=None, weight=1):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :param in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        :param weight: The weight of the parameter for random choice.
                       Higher weights indicate chosen more often.
        """
        super().addPrimitive(primitive, in_types, ret_type, name)
        self.weights[primitive] = weight

    def addTerminal(self, terminal, ret_type, name=None, weight=1):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        :param weight: The weight of the parameter for random choice.
                       Higher weights indicate chosen more often.
        """
        super().addTerminal(terminal, ret_type, name)
        self.weights[terminal] = weight

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


def setup_simple_pset(points: pd.DataFrame) -> gp.PrimitiveSet:
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
    local_points = points.copy()
    if local_points.columns[-1] == "target":
        output_type = bool
        local_points.drop("target", axis=1, inplace=True)
    else:
        output_type = generators[local_points.dtypes[local_points.columns[-1]]]
    # generators[np.dtype("O")] = output_type

    assert output_type in {int, float, str, bool}, f"Bad output type {output_type}"

    types = local_points.dtypes.to_dict()
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
        term_set = set(local_points[v])
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
        pset.addPrimitive(operator.__le__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__ge__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__ne__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__lt__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__gt__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__eq__, [int, int], bool, weight=1)
        if int in datatypes:
            pset.addPrimitive(operator.add, [int, int], int)
            pset.addPrimitive(operator.sub, [int, int], int)
            pset.addPrimitive(operator.mul, [int, int], int)
    elif output_type == str:
        pass
    else:
        raise ValueError(f"Invalid output type {output_type}.")

    assert all(
        [type(t.value) in {int, str, float, bool} for t in pset.mapping.values() if hasattr(t, "value")]
    ), "Bad type"
    return pset


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
    local_points = points.copy()
    if local_points.columns[-1] == "target":
        output_type = bool
        local_points.drop("target", axis=1, inplace=True)
    else:
        output_type = generators[local_points.dtypes[local_points.columns[-1]]]
    # generators[np.dtype("O")] = output_type

    assert output_type in {int, float, str, bool}, f"Bad output type {output_type}"

    types = local_points.dtypes.to_dict()
    names = list(types)
    datatypes = [generators[types[v]] for v in names]
    assert all([t in {int, float, str, bool} for t in datatypes]), f"Bad datatype {output_type}"

    if output_type == bool:
        pset = PrimitiveSetTyped("MAIN", datatypes, output_type)
    else:
        pset = PrimitiveSetTyped("MAIN", datatypes[:-1], output_type)

    rename = {f"ARG{i}": col for i, col in enumerate(names)}
    pset.renameArguments(**rename)

    assert all([type(t.value) in {int, str, float} for t in pset.mapping.values() if hasattr(t, "value")]), "Bad type"

    # Add literal terminals
    for v, typ in zip(names, datatypes):
        assert typ in {int, str, float, bool}, "Bad pset terminal type {typ}"
        term_set = set(local_points[v])
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
        pset.addPrimitive(operator.__le__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__ge__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__ne__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__lt__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__gt__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__eq__, [int, int], bool, weight=1)
        pset.addPrimitive(operator.__and__, [bool, bool], bool, weight=1)
        pset.addPrimitive(operator.__or__, [bool, bool], bool, weight=1)
        pset.addPrimitive(operator.__not__, [bool], bool, weight=1)
        if int in datatypes:
            pset.addPrimitive(operator.add, [int, int], int)
            pset.addPrimitive(operator.sub, [int, int], int)
            pset.addPrimitive(operator.mul, [int, int], int)
    elif output_type == str:
        pass
    else:
        raise ValueError(f"Invalid output type {output_type}.")

    assert all(
        [type(t.value) in {int, str, float, bool} for t in pset.mapping.values() if hasattr(t, "value")]
    ), "Bad type"
    return pset


def setup_pset(points: pd.DataFrame) -> gp.PrimitiveSet:
    # try:
    #     return setup_pset_aux(points)
    # except:
    #     logger.debug(traceback.format_exc())
    return setup_pset_aux(points)
