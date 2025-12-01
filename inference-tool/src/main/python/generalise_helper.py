import json
import re
from collections import OrderedDict

import deap_gp
import efsm
import pandas as pd


class AdditiveDict:
    def __init__(self, generator):
        self.generator = generator
        self.data = {}

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.generator(self.data)
        return self.data[key]


def __transition_pset(ip_sig, op_sig, dest, transition):
    """
    Computes the set of all terminal/variables/functions that can appear in guards, output and
    update functions of the transition.
    """
    samples = pd.concat(
        [
            pd.Series(ip_sig, index=transition.samples.index, name="ip_sig").astype("string"),
            pd.Series(op_sig, index=transition.samples.index, name="op_sig").astype("string"),
            transition.samples,
            pd.Series(True, index=transition.samples.index, name="guard"),
        ],
        axis=1,
    )

    return deap_gp.setup_full_pset(samples)


def __formula_to_tree(formula, pset):
    """
    Convert inferred function from string to tree representation.
    """
    return deap_gp.to_nodes_edges_labels(formula, pset)


def efsm_to_json(_efsm: efsm.EFSM, filepath):
    """
    Convert the provided EFSM to a JSON representation which is parsable by the scala-gp inference tool.
    """

    __output_update_pattern = re.compile(r"^[oO](?P<index>[\d]+)$")

    def update_to_tree(formula, pset, outputs):
        match_output = __output_update_pattern.match(formula)
        return (
            __formula_to_tree(formula, pset) if match_output is None else outputs[int(match_output.group("index")) + 1]
        )

    state_ids = AdditiveDict(lambda x: len(x))
    edge_id = 0
    G = []

    samples = efsm.expand_list(pd.DataFrame([_efsm.initialisation], index=[0]), "registers", "r")
    samples = pd.concat([samples, pd.Series("epsilon", index=samples.index, name="op_sig").astype("string")], axis=1)
    pset = deap_gp.setup_full_pset(samples)

    initial = _efsm.initialisation["state"]
    configuration = _efsm.initialisation["registers"]

    G.append(
        {
            "tid": [edge_id := edge_id + 1],
            "origin": state_ids[initial],
            "dest": state_ids[initial],
            "label": "init",
            "arity": 0,
            "guards": [],
            "outputs": [__formula_to_tree("epsilon", pset)],
            "updates": [(f"r{i}", __formula_to_tree(str(value), pset)) for (i, value) in enumerate(configuration)],
            "drop_guards": False,
        }
    )

    for origin in _efsm:
        for ip_sig, op_sig, dest in _efsm[origin]:

            transition = _efsm[origin][(ip_sig, op_sig, dest)]
            pset = __transition_pset(ip_sig, op_sig, dest, transition)

            input_arity = transition.input_arity
            output_arity = transition.output_arity

            guards = [__formula_to_tree(transition.guard, pset)] if transition.guard is not None else []
            outputs = [__formula_to_tree(op_sig, pset)] + (
                [__formula_to_tree(output, pset) for output in transition.output] if output_arity > 0 else []
            )
            updates = [(f"r{i}", update_to_tree(update, pset, outputs)) for (i, update) in enumerate(transition.update)]

            G.append(
                {
                    "tid": [edge_id := edge_id + 1],
                    "origin": state_ids[origin],
                    "dest": state_ids[dest],
                    "label": ip_sig,
                    "arity": input_arity,
                    "guards": guards,
                    "outputs": outputs,
                    "updates": updates,
                    "drop_guards": True,
                }
            )

    with open(filepath, "w") as f:
        json.dump(G, f, indent=2)

    return G, state_ids


def efsm_to_dot(_efsm: efsm.EFSM, filepath):

    import operator

    __renamings = {
        operator.add.__name__: "+",
        operator.sub.__name__: "-",
        operator.mul.__name__: "*",
        operator.truediv.__name__: "/",
        "div": "/",
        operator.__le__.__name__: "<=",
        operator.__ge__.__name__: ">=",
        operator.__lt__.__name__: "<",
        operator.__gt__.__name__: ">",
        operator.__eq__.__name__: "=",
        operator.__and__.__name__: "and",
        operator.__or__.__name__: "or",
        operator.__not__.__name__: "not",
    }

    def rename(graph):
        nodes, _, labels = graph
        for node in nodes:
            if labels[node] in __renamings:
                labels[node] = __renamings[labels[node]]
        return graph

    def infix(graph, node):

        nodes, edges, labels = graph

        outgoing = [dest for (source, dest) in edges if source == node]
        arity = len(outgoing)

        if arity == 0:
            return str(labels[node])

        if arity == 2:
            return f"({infix(graph,outgoing[0])} {labels[node]} {infix(graph,outgoing[1])})"

        return f"{labels[node]} ({','.join([infix(graph,dest) for dest in outgoing])})"

    psets = {}

    def pset(ip_sig, op_sig, dest, transition):

        if (ip_sig, op_sig, dest) not in psets:
            psets[(ip_sig, op_sig, dest)] = __transition_pset(ip_sig, op_sig, dest, transition)

        return psets[(ip_sig, op_sig, dest)]

    def translate(ip_sig, op_sig, dest, transition, expression):
        graph = __formula_to_tree(expression, pset(ip_sig, op_sig, dest, transition))
        return infix(rename(graph), 0)

    return efsm.to_dot(_efsm, filepath, translation=translate)
