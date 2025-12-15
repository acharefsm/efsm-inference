"""
Created on Fri Oct 20 13:14:41 2023

@author: michael, luca + German

Used to genrelise a specific EFSM utilising the efsm class too
"""

import argparse
import json
import re

import gp_pset
import gp_simplification
import gp_fitness
import efsm
import deap_gp

import pandas as pd
import networkx as nx

from deap import gp

from gp_repair import infix_to_prefix2 as infix_to_prefix
from gp_guard_combinator import run_gp
from gp_pset import setup_pset, setup_simple_pset
from gp_fitness import correct_dt


from collections import OrderedDict

class Counter:
    def __init__(self):
        self.total_incorrect = 0

    def increment(self):
        self.total_incorrect += 1

    def get_total_incorrect(self):
        return self.total_incorrect

class AdditiveDict:
    def __init__(self, generator):
        self.generator = generator
        self.data = {}

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.generator(self.data)
        return self.data[key]
    
def infer_guard(samples, counter=None, **kwargs):
    pset = setup_pset(samples)
    simple_pset = setup_simple_pset(samples)
    best, best_guard = run_gp(samples, pset, simple_pset, **kwargs)

    correct = correct_dt(best, samples, pset)

    if not correct:
        counter.increment()

    print(best_guard)

    return best_guard
    
def infer_output(samples, counter=None, **kwargs):
    pset = deap_gp.setup_pset(samples)
    best = deap_gp.run_gp(
        samples,
        pset,
        **kwargs,
    )

    args = samples[samples.columns[:-1]]
    outputs = samples[samples.columns[-1]]

    correct = gp_fitness.correct(best, samples, pset, [() for i in range(len(samples))])

    if not correct:
        counter.increment()
        bf = deap_gp.gp.compile(expr=best, pset=pset)
        predicted = args.apply(lambda args: bf(**(args.to_dict())), axis=1)
        correct = outputs == predicted
        # print("guard inferred ",str(best))
        # print("samples",samples)
        # print("correct",correct)

    return str(best)

def trace_to_json(trace: pd.DataFrame, filepath):

    with open(filepath, "w") as f:

        inputs = efsm.expand_parametrised(trace, "Input")
        outputs = efsm.expand_parametrised(trace, "Output")

        json.dump(
            [
                [{"label": "init", "inputs": [], "outputs": ["epsilon"]}]
                + [
                    {
                        "label": inputs.loc[step]["signature"],
                        "inputs": inputs.loc[step]["args"],
                        "outputs": [outputs.loc[step]["signature"]] + outputs.loc[step]["args"],
                    }
                    for step in trace.index
                ]
            ],
            f,
            indent=2,
        )


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

    return gp_pset.setup_full_pset(samples)


def __formula_to_tree(exp, pset, rename={}):
    """
    Convert inferred function from string to tree representation.
    """
    try:
        exp = gp.PrimitiveTree.from_string(exp, pset)
    except:
        print(exp)
        exp = gp.PrimitiveTree.from_string(infix_to_prefix(exp), pset)
    rename = {k: gp.Terminal(v, None, object) for k, v in rename.items()}
    for inx, element in enumerate(exp):
        if isinstance(element, gp.Terminal) and element.format() in rename:
            exp[inx] = rename[element.format()]
    assert "r_b" not in str(exp), f"{exp}: {rename}"
    return gp.graph(exp)


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
    pset = gp_pset.setup_full_pset(samples)

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
                    "arity": int(input_arity),
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
        operator.__ne__.__name__: "not =",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="get_groups.py",
        description="Determines the transition grouping and runs GP to generalise the conjecture model.",
    )
    parser.add_argument("-c", "--conjecture", help="Path to the DOT file containing the conjecture model.", required=True)
    parser.add_argument("-t", "--trace", help="Path to the CSV containing the trace.", required=False)
    parser.add_argument("-s", "--seed", help="The random seed.", required=False, default=0)

    args = parser.parse_args()

    # trace = pd.read_csv(args.trace)
    # trace.set_index("Step", inplace=True, drop=False)
    # trace_to_json(trace, args.trace.replace(".csv", "_new_trace.json"))

    conjecture = nx.nx_pydot.read_dot(args.conjecture)

    counter = Counter()

    generalised = efsm.generalise(efsm.efsm(conjecture),infer_output, infer_guard, random_seed=args.seed, counter=counter)

    print(counter.get_total_incorrect())

    efsm_to_dot(generalised, args.conjecture.replace(".dot", f"_generalised.dot"))
    efsm_to_json(generalised, args.conjecture.replace(".dot", "_generalised.json"))
