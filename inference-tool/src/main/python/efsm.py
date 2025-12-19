"""
Created on Sept 11 2025

utility methods to manipulate EFSM and sampled NFSM

@author: German Vega
"""

import re

import networkx as nx
import pandas as pd


class EFSM(dict):

    __slots__ = ("initialisation", "arity")

    def __init__(self, initialisation, *args, **kwargs):
        self.initialisation = initialisation
        self.arity = len(initialisation["registers"])
        dict.__init__(self, *args, **kwargs)


class State(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)


class Transition:

    __slots__ = ("input_arity", "output_arity", "samples", "guard", "output", "update")

    def __init__(self, input_arity: int, output_arity: int, samples: pd.DataFrame, update: list[str]):
        self.input_arity = input_arity
        self.output_arity = output_arity
        self.samples = samples
        self.guard = None
        self.output = []
        self.update = update


def efsm(nfsm: nx.Graph) -> EFSM:
    """
    Builds an EFSM from a sampled NFSM (represented in dot format)

    Corresponding samples are associated with each edge.
    """

    initial = __initialisation(nfsm)
    observations = __samples(nfsm)

    assert len(initial) == 1, "a single initial configuration must be specified"
    initial = initial.iloc[0]

    states = {state for state in observations["origin"]} | {state for state in observations["dest"]}
    efsm = EFSM(initial, **{state: State() for state in states})

    for (origin, dest, ip_sig, op_sig), group_samples in observations.groupby(["origin", "dest", "ip_sig", "op_sig"]):

        ip_arity = group_samples["ip_arity"].max()
        op_arity = group_samples["op_arity"].max()

        registers = expand_list(group_samples, "registers", "r")
        ip_args = expand_list(group_samples, "ip_args", "i")
        op_args = expand_list(group_samples, "op_args", "o")

        update = group_samples["update"].iloc[0]

        efsm[origin][(ip_sig, op_sig, dest)] = Transition(
            ip_arity, op_arity, pd.concat([registers, ip_args, op_args], axis=1), update
        )

    return efsm


def to_dot(efsm, filepath, translation=None):

    def translate(expression):

        if expression is None:
            return expression

        if translation is None:
            return expression

        return translation(ip_sig, op_sig, dest, transition, expression)

    def guard(guard):
        return f"[{guard}]" if guard is not None else ""

    def tuple(elements):
        return f"({','.join(elements)})" if len(elements) != 0 else ""

    def label(ip_sig, op_sig, dest, transition):

        t_guard = translate(transition.guard)
        t_output = [translate(output) for output in transition.output]
        t_update = [translate(update) for update in transition.update]

        return f"{ip_sig}{guard(t_guard)}/{op_sig}{tuple(t_output)} {tuple(t_update)}"

    G = nx.MultiDiGraph(name="EFSM")
    for origin in efsm:
        for ip_sig, op_sig, dest in efsm[origin]:

            transition = efsm[origin][(ip_sig, op_sig, dest)]
            G.add_edge(origin, dest, label=label(ip_sig, op_sig, dest, transition))

    nx.nx_pydot.write_dot(G, filepath)
    return G


def generalise(
    efsm: EFSM,
    infer_function,
    guard_infer_function,
    *args,
    **kwargs,
) -> EFSM:
    """
    Generalises an EFSM by inferring output functions from samples associated with each edge.
    """

    def infer_output(samples: pd.DataFrame):
        return infer_function(
            samples,
            *args,
            **kwargs,
        )
    
    def infer_guard(samples: pd.DataFrame):
        return guard_infer_function(
            samples,
            *args,
            **kwargs,
        )

    for origin in efsm:

        # compute output functions for each transition
        for ip_sig, op_sig, dest in efsm[origin]:

            transition = efsm[origin][(ip_sig, op_sig, dest)]

            columns = transition.samples.columns.to_list()
            output_columns = columns[-transition.output_arity :] if transition.output_arity != 0 else []
            args_columns = columns[: -transition.output_arity] if transition.output_arity != 0 else columns

            # for output in output_columns:
            #     print(ip_sig, op_sig, dest, output)
            #     print("===================================================================================================")

            transition.output = [infer_output(transition.samples[args_columns + [output]]) for output in output_columns]

        # compute guards when required
        transitions = pd.DataFrame(efsm[origin].keys(), columns=["ip_sig", "op_sig", "dest"])
        for ip_sig, branches in transitions.groupby("ip_sig"):

            if len(branches) > 1:

                # concatenate the samples of all the branches
                branching = pd.DataFrame()
                for _, (ip_sig, op_sig, dest) in branches.iterrows():

                    transition = efsm[origin][(ip_sig, op_sig, dest)]

                    columns = transition.samples.columns.to_list()
                    args_columns = columns[: -transition.output_arity] if transition.output_arity != 0 else columns

                    branch = pd.concat(
                        [
                            transition.samples[args_columns],
                            pd.Series(op_sig, index=transition.samples.index, name="op_sig"),
                            pd.Series(dest, index=transition.samples.index, name="dest"),
                        ],
                        axis=1,
                    )

                    branching = pd.concat([branching, branch])

                # for each branch compute a branching expression (must evaluate to True for the given branch
                # and False for any other branch)

                columns = branching.columns.to_list()
                args_columns = columns[:-2]

                # branching["guard"] = False
                # branching.loc[branch.index, "guard"] = True

                branching["target"] = branching["dest"].astype(str) + branching["op_sig"].astype(str)

                guards = infer_guard(branching[args_columns + ["target"]])
                for (op_sig, dest), branch in branching.groupby(["op_sig", "dest"]):

                    # print(infer_guard(branching[args_columns + ["target"]]))

                    transition = efsm[origin][(ip_sig, op_sig, dest)]
                    # transition.guard = 

    return efsm


__quoted_attribute_pattern = re.compile(r"[\<\"](?P<unquoted>.*)[\"\>]")


def __unquote(value):
    global __quoted_attribute_pattern

    return match.group("unquoted") if (match := __quoted_attribute_pattern.match(value)) else value


def __initialisation(nfsm: nx.Graph) -> pd.DataFrame:
    """
    Extract the initial configurations (state and register values) recorded in
    a sampled NFSM (represented in dot format)
    """
    global nfsm_configuration_pattern

    markers = [node for node in nfsm.nodes if node.startswith("__start")]
    initial = [dest for _, dest in nfsm.edges(markers)]

    nfsm.remove_nodes_from(markers)

    initial = pd.DataFrame(
        [{"state": state, "configuration": __unquote(nfsm.nodes[state]["configuration"])} for state in initial]
    )
    initial.set_index("state", inplace=True, drop=False)

    initial["registers"] = expand_configuration(initial, "configuration")

    return initial[["state", "registers"]]


__nfsm_label_pattern = re.compile(r"(?P<input>.*)/(?P<output>.*)")


def __samples(nfsm: nx.Graph) -> pd.DataFrame:
    """
    Extract a dataframe from the samples recorded in a sampled NFSM (represented in dot format)
    """
    global __nfsm_label_pattern

    transitions = pd.DataFrame(
        [
            {
                "tid": tid,
                "origin": origin,
                "dest": dest,
                "label": __unquote(data["label"]),
                "configuration": __unquote(data["configuration"]),
                "update": __unquote(data["update"]) if "update" in data else None,
            }
            for tid, (origin, dest, data) in enumerate(nfsm.edges(data=True), 1)
        ]
    )

    transitions.set_index("tid", inplace=True, drop=False)

    labels = transitions["label"].apply(__nfsm_label_pattern.match)
    assert labels.map(lambda match: match != None).all(), "transition label does not match pattern "

    transitions["inputs"] = labels.apply(lambda match: match.group("input"))
    transitions["outputs"] = labels.apply(lambda match: match.group("output"))

    inputs = expand_parametrised(transitions, "inputs")
    outputs = expand_parametrised(transitions, "outputs")
    registers = expand_configuration(transitions, "configuration")
    updates = expand_configuration(transitions, "update").apply(
        lambda update: [exp if exp is not None else f"r{index}" for index, exp in enumerate(update)]
    )

    inputs.rename(columns={"signature": "ip_sig", "arity": "ip_arity", "args": "ip_args"}, inplace=True)
    outputs.rename(columns={"signature": "op_sig", "arity": "op_arity", "args": "op_args"}, inplace=True)
    registers.name = "registers"
    updates.name = "update"

    samples = pd.concat([transitions[["origin", "dest"]], registers, inputs, outputs, updates], axis=1)
    samples.reset_index(inplace=True)
    samples.index = samples["tid"]

    return samples


__parametrised_pattern = re.compile(r"(?P<signature>[\w-]+)(\((?P<args>.*?)\))?")


def expand_parametrised(frame, column):
    """
    Expands a column with parametrised values into separate columns for the signature, arity and arguments.

    Arguments are represented as a list of values (converting to numeric values if possible)
    """

    global __parametrised_pattern

    matches = frame[column].apply(__parametrised_pattern.match)
    assert matches.map(lambda match: match != None).all(), "values do not match parameterized pattern "

    signatures = matches.apply(lambda match: match.group("signature"))
    args = matches.apply(lambda match: match.group("args"))

    expanded = pd.DataFrame({"signature": signatures, "arity": 0, "args": args}, index=frame.index)

    for _, group in expanded[~expanded["args"].isna()].groupby("signature"):

        group = group["args"].str.split(",", expand=True).apply(pd.to_numeric, errors="ignore")

        arity = group.columns.size
        group = group.apply(pd.Series.to_list, axis=1)

        expanded.loc[group.index, "arity"] = arity
        expanded.loc[group.index, "args"] = group

    expanded["args"] = expanded["args"].apply(lambda args: args if args is not None else [])
    return expanded


def expand_configuration(frame, column):
    """
    Expands a column with a textual configuration of registers (represented as the text of a list
    or tuple) into a series composed of list of values (converting numeric values if possible)

    """

    registers = (
        frame[column]
        .str.extract("[\[\(](.*?)[\)\]]", expand=False)
        .str.replace(" ", "")
        .str.split(",", expand=True)
        .apply(pd.to_numeric, errors="ignore")
        .apply(pd.Series.to_list, axis=1)
    )

    return registers


def expand_list(frame, column, prefix):
    """
    Expands a column that contains a list of values into multiple columns.

    """
    expanded = pd.DataFrame(frame[column].to_list(), index=frame.index)

    textual = expanded.select_dtypes(exclude=["number", "bool"]).astype("string")
    expanded[textual.columns] = textual

    names = {column: prefix + str(column) for column in expanded.columns}
    expanded.rename(columns=names, inplace=True)

    return expanded
