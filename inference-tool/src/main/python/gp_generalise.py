"""
Created on Fri Oct 20 13:14:41 2023

@author: michael
"""

import argparse
import json
import re

import html
import csv
import efsm

import pandas as pd
import deap_gp

import networkx as nx

from collections import OrderedDict
from enchant.utils import levenshtein


class AdditiveDict:
    def __init__(self, generator):
        self.generator = generator
        self.data = {}

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.generator(self.data)
        return self.data[key]

def infer_output(samples, mu_size, lambda_size, generation_size, mut_proba, max_init_depth, max_depth, fitness_type, **kwargs):
    global total_correct
    
    pset = deap_gp.setup_pset(samples)
    best = deap_gp.run_gp(mut_proba, samples,pset, mu=mu_size, lamb=lambda_size, ngen=generation_size, type_=fitness_type, **kwargs)

    args        = samples[samples.columns[:-1]]
    outputs     = samples[samples.columns[-1]]

    correct = deap_gp.correct(best, samples, pset, [() for i in range(len(samples))])

    if not correct:
        total_correct += 1
        print(total_correct)
        bf = deap_gp.gp.compile(expr=best, pset=pset)
        predicted = args.apply(lambda args: bf(**(args.to_dict())),axis=1)
        correct   = outputs == predicted
        # print("guard inferred ",str(best))
        # print("samples",samples)
        # print("correct",correct)

    return str(best)

def trace_to_json(trace : pd.DataFrame, filepath):

    with open(filepath, "w") as f:
        
        inputs      = efsm.expand_parametrised(trace,"Input")
        outputs     = efsm.expand_parametrised(trace,"Output")

        json.dump(
            [
                [
                    {
                        "label" : "init",
                        "inputs" : [],
                        "outputs" : ["epsilon"]
                    }
                ]
                +
                [
                    {
                        "label":    inputs.loc[step]["signature"],
                        "inputs":   inputs.loc[step]["args"],
                        "outputs":  [outputs.loc[step]["signature"]]+outputs.loc[step]["args"],
                    }

                    for step in trace.index
                ]
            ],
            f,
            indent=2,
        )

def __transition_pset(ip_sig,op_sig,dest,transition) :
    """
    Computes the set of all terminal/variables/functions that can appear in guards, output and
    update functions of the transition.
    """
    samples = pd.concat([
                pd.Series(ip_sig,index=transition.samples.index,name="ip_sig").astype("string"),
                pd.Series(op_sig,index=transition.samples.index,name="op_sig").astype("string"),
                transition.samples,
                pd.Series(True,index=transition.samples.index,name="guard")
            ],axis=1)
    
    return deap_gp.setup_full_pset(samples)

def __formula_to_tree(formula,pset) :
    """
    Convert inferred function from string to tree representation.
    """
    return deap_gp.to_nodes_edges_labels(formula,pset)

def efsm_to_json(_efsm : efsm.EFSM, filepath):

    """
    Convert the provided EFSM to a JSON representation which is parsable by the scala-gp inference tool.
    """

    __output_update_pattern = re.compile(r"^[oO](?P<index>[\d]+)$")
    def update_to_tree(formula,pset,outputs) :
        match_output = __output_update_pattern.match(formula)
        return __formula_to_tree(formula,pset) if match_output is None else outputs[int(match_output.group("index"))+1]

  
    state_ids   = AdditiveDict(lambda x: len(x))
    edge_id     = 0
    G = []


    samples = efsm.expand_list(pd.DataFrame([_efsm.initialisation],index=[0]),"registers","r")
    samples = pd.concat([samples,pd.Series("epsilon",index=samples.index,name="op_sig").astype("string")],axis=1)
    pset    = deap_gp.setup_full_pset(samples)

    initial         = _efsm.initialisation["state"]
    configuration   = _efsm.initialisation["registers"]

    G.append(
        {
            "tid": [edge_id := edge_id+1],
            "origin": state_ids[initial],
            "dest": state_ids[initial],
            "label": "init",
            "arity": 0,
            "guards": [],
            "outputs": [__formula_to_tree("epsilon",pset)],
            "updates": [(f"r{i}",__formula_to_tree(str(value),pset)) for (i,value) in enumerate(configuration)],
            "drop_guards": False,
        }
    )

    for origin in _efsm:
        for (ip_sig,op_sig,dest) in _efsm[origin]:

            transition  = _efsm[origin][(ip_sig,op_sig,dest)]
            pset        = __transition_pset(ip_sig,op_sig,dest,transition)

            input_arity     = transition.input_arity
            output_arity    = transition.output_arity

            guards          = [__formula_to_tree(transition.guard,pset)] if transition.guard is not None else []
            outputs         = [__formula_to_tree(op_sig,pset)]+([__formula_to_tree(output,pset) for output in transition.output] if output_arity > 0 else [])
            updates         = [(f"r{i}", update_to_tree(update,pset,outputs)) for (i,update) in enumerate(transition.update)]
            
            G.append( 
                {
                    "tid": [edge_id := edge_id+1],
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

def efsm_to_dot(_efsm : efsm.EFSM, filepath):
    
    import operator
    __renamings = {

        operator.add.__name__       : "+",
        operator.sub.__name__       : "-",
        operator.mul.__name__       : "*",
        operator.truediv.__name__   : "/",
        "div"                       : "/",
        operator.__le__.__name__    : "<=",
        operator.__ge__.__name__    : ">=",
        operator.__lt__.__name__    : "<",
        operator.__gt__.__name__    : ">",
        
        operator.__eq__.__name__    : "=",
        
        operator.__and__.__name__   : "and",
        operator.__or__.__name__    : "or",
        operator.__not__.__name__   : "not",
    }

    def rename(graph) :
        nodes,_,labels = graph
        for node in nodes:
            if (labels[node] in __renamings) :
                labels[node] = __renamings[labels[node]]
        return graph

    def infix(graph,node) :
        
        nodes,edges,labels = graph

        outgoing    = [dest for (source,dest) in edges if source == node]
        arity       = len(outgoing)
        
        if arity == 0 :
            return str(labels[node])

        if arity == 2 :
            return f"({infix(graph,outgoing[0])} {labels[node]} {infix(graph,outgoing[1])})"

        return f"{labels[node]} ({','.join([infix(graph,dest) for dest in outgoing])})"
    
    psets = {}
    def pset(ip_sig,op_sig,dest,transition) :
        
        if (ip_sig,op_sig,dest) not in psets :
            psets[(ip_sig,op_sig,dest)] = __transition_pset(ip_sig,op_sig,dest,transition)

        return psets[(ip_sig,op_sig,dest)]

    def translate(ip_sig,op_sig,dest,transition,expression) : 
        graph = __formula_to_tree(expression,pset(ip_sig,op_sig,dest,transition))
        return infix(rename(graph),0)
    
    return efsm.to_dot(_efsm,filepath, translation=translate)

def compare_efsm_graphs(original : nx.MultiDiGraph, estimated : nx.MultiDiGraph) -> float:
    for u in estimated:
        for v in original[u]:
            estimate_labels = []
            original_labels = []
            for label in original[u][v]:
                estimate_labels.append(estimated[u][v][label].get('label', '').strip('<>'))
                original_labels.append(original[u][v][label].get('label', '').strip('<>'))

            for estimated_label in estimate_labels:
                estimated_parts = estimated_label.split('/')
                for original_label in original_labels:
                    original_parts = original_label.split('/')
                    if estimated_parts[0][:4] == original_parts[0][:4] and estimated_parts[1][:4] == original_parts[1][:4]:
                        original_label = html.unescape(original_label)
                        original_label = re.sub(r"([a-zA-Z_]\w*|-?\d+)\s*>=\s*([a-zA-Z_]\w*|-?\d+)", r"\2 <= \1", original_label)
                        original_label = re.sub(r"([a-zA-Z_]\w*|-?\d+)\s*>\s*([a-zA-Z_]\w*|-?\d+)", r"\2 < \1", original_label)
                        return levenshtein(estimated_label, original_label)

parser = argparse.ArgumentParser(
    prog="get_groups.py",
    description="Determines the transition grouping and runs GP to generalise the conjecture model.",
)
parser.add_argument("-c", "--conjecture", help="Path to the DOT file containing the conjecture model.", required=True)
parser.add_argument("-t", "--trace", help="Path to the CSV containing the trace.", required=True)
parser.add_argument("-s", "--seed", help="The random seed.", required=False, default=0)
parser.add_argument("-e", "--efsm", help="The original EFSM", required=True)

args = parser.parse_args()

original = nx.MultiDiGraph(nx.nx_pydot.read_dot(args.efsm))

mu_sizes = [20]
lambda_sizes = [5]
generation_sizes = [50]
mutation_probs = [0.5]
max_depths = [(7, 15)]
fitness_types = ["step", "continuous"]


trace = pd.read_csv(args.trace)
trace.set_index("Step",inplace=True,drop=False)
trace_to_json(trace,args.trace.replace(".csv", "_new_trace.json"))

headers = ["mu_size", "lambda_size", "generation_size", "mutation_prob", "max_init_depth", "max_depth", "fitness_type", "levenshtein_distance", "total_wrong"]

# with open("experiment_results.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(headers)

conjecture = nx.nx_pydot.read_dot(args.conjecture)
efsmm = efsm.efsm(conjecture)
for mu_size in mu_sizes:
    for lambda_size in lambda_sizes:
        for generation_size in generation_sizes:
            for mutation_prob in mutation_probs:
                for max_init_depth, max_depth in max_depths:
                    for fitness_type in fitness_types:
                        total_correct = 0

                        generalised = efsm.generalise(mu_size, lambda_size, generation_size, mutation_prob, max_init_depth, max_depth, fitness_type, efsmm, infer_output, random_seed=args.seed)

                        print(mu_size, lambda_size, generation_size, mutation_prob, max_init_depth, max_depth, fitness_type)

                        estimated = efsm_to_dot(generalised, args.conjecture.replace(".dot", f"_generalised_{mu_size}_{lambda_size}_{generation_size}_{mutation_prob}_{max_init_depth}_{max_depth}_{fitness_type}.dot"))
                        efsm_to_json(generalised, args.conjecture.replace(".dot", f"_generalised_{mu_size}_{lambda_size}_{generation_size}_{mutation_prob}_{max_init_depth}_{max_depth}_{fitness_type}.json"))

                        print(compare_efsm_graphs(original, estimated), total_correct)

                        # with open("experiment_results.csv", "a", newline="") as f:
                        #     writer = csv.writer(f)
                        #     writer.writerow([mu_size, lambda_size, generation_size, mutation_prob, max_init_depth, max_depth, fitness_type, compare_efsm_graphs(original, estimated), total_correct])
