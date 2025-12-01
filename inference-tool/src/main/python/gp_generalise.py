"""
Created on Fri Oct 20 13:14:41 2023

@author: michael
"""

import argparse
import json

import efsm
import pandas as pd
from gp_experiment import run_experiment


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


parser = argparse.ArgumentParser(
    prog="get_groups.py",
    description="Determines the transition grouping and runs GP to generalise the conjecture model.",
)
parser.add_argument("-c", "--conjecture", help="Path to the DOT file containing the conjecture model.", required=True)
parser.add_argument("-t", "--trace", help="Path to the CSV containing the trace.", required=True)
parser.add_argument("-s", "--seed", help="The random seed.", required=False, default=0)
parser.add_argument("--n_jobs", required=False, type=int, default=1)

args = parser.parse_args()

trace = pd.read_csv(args.trace)
trace.set_index("Step", inplace=True, drop=False)
trace_to_json(trace, args.trace.replace(".csv", "_new_trace.json"))

run_experiment(trace, args.conjecture, args.seed, args.n_jobs)
