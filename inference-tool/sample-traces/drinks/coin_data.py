"""
Quick script to get the data from the coin transitions to infer output and update
functions.
"""

import json
import pandas as pd

with open("drinks_register_guard_bounded_inferred.simulation_new_trace.json") as f:
    trace = json.load(f)[0]

r = {r: None for r in range(4)}

new_trace = []

for e in trace:
    i = e["inputs"]
    o = e["outputs"][1:]
    e = e | {f"r{r}": val for r, val in r.items()}
    if e["outputs"] in [["Omega"], ["omega"], ["epsilon"]]:
        updates = (r[0], r[1], r[2], r[3])
    elif e["label"] == "select":
        updates = (i[0], r[1], o[1], o[2])
    elif e["label"] == "coin" and e["outputs"][0] == "Display":
        updates = (r[0], i[0], o[0], r[3])
    elif e["label"] == "coin" and e["outputs"][0] == "Reject":
        updates = (r[0], i[0], r[2], r[3])
    elif e["label"] == "vend" and e["outputs"][0] == "Serv":
        updates = (r[0], r[1], o[1], o[2])
    else:
        raise ValueError(f"Invalid event {e}")
    r = {r: val for r, val in enumerate(updates)}
    new_trace.append(e)

trace = pd.DataFrame(new_trace).query("label == 'coin'")
trace["i0"] = [i[0] for i in trace["inputs"]]
trace = trace.drop("inputs", axis=1)
trace["o0"] = [o[0] for o in trace["outputs"]]
trace = trace.query("o0 != 'Omega'")
trace["o1"] = [o[1] for o in trace["outputs"]]
trace = trace.drop("outputs", axis=1)

trace["expected"] = (trace["r2"] + trace["i0"]) <= trace["r3"]

trace["o0"] = [x == "Display" for x in trace["o0"]]
print(trace)
print((trace["o0"] == trace["expected"]).all())
trace.to_csv("/tmp/test-guard3.csv", index=False)
