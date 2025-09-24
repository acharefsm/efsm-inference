# Generalising an EFSM

## Inputs:
1. A CSV file with the following format representing the trace, where the relevant registers for each event signature s are named r_s. You do not need to include every register.
|Step|Input    |Output|r_Select|r_Coin|r_Display|r_Serve|
|----|---------|------|--------|------|---------|-------|
|1   |Coin(100)|Omega |        |      |         |       |
|2   |Vend(0)  |Omega |        |      |         |       |
|3   |Select(2)|Pay(0)|2       |0     |0        |0      |
2. A DOT file representing the sampled EFSM. This DOT file must accept the trace in the CSV file.

## Inferring output functions
Execute `get_groups.py` using

```
get_groups.py -c CONJECTURE -t TRACE -i INITIAL_STATE [-s SEED]

required arguments:
  -c CONJECTURE, --conjecture CONJECTURE
                        Path to the DOT file containing the conjecture model.
  -t TRACE, --trace TRACE
                        Path to the CSV containing the trace.
  -i INITIAL_STATE, --initial_state INITIAL_STATE
                        The initial state of the conjecture.

optional arguments:
  -s SEED, --seed SEED  The random seed.
```

This will run for a while, and then produce several files:

`$CONJECTURE_generalised.dot` is a visual representation of the generalised EFSM WITHOUT GUARDS.

`$CONJECTURE_generalised.json` is a JSON representation of the same thing (to be fed into my tool in the next step).

`$CONJECTURE_ungeneralised.json` is a JSON representation of the input `$CONJECTURE` also compatible with my tool in the next step.

`$CONJECTURE_new_trace.json` is a JSON file containing my tool's representation of the `TRACE` for the next step.

## Inferring guards
Call my EFSM inference tool (from the `ehW` branch) using

```
java -jar target/scala-2.12/inference-tool-assembly-0.1.0-SNAPSHOT.jar \
--pta $CONJECTURE_generalised.json \
-h distinguish  \
$CONJECTURE_new_trace.json \
$CONJECTURE_new_trace.json
```

N.B. Before doing this, you may need to `export LD_LIBRARY_PATH=lib` to make sure Java has access to Z3.

This command will create a directory `dotfiles` with the various output files. `$CONJECTURE_new_trace.dot` will contain a dot representation of the final inferred EFSM with guards.

docker run -it --entrypoint /bin/bash -v C:\projects\efsm-inference\inference-tool\sample-traces:/home/sbtuser/efsm-inference/sample-traces -v C:\projects\efsm-inference\inference-tool\dotfiles:/home/sbtuser/efsm-inference/dotfiles inference

java -jar target/scala-2.12/inference-tool-assembly-0.1.0-SNAPSHOT.jar --pta sample-traces/sampled_generalised.json -h distinguish sample-traces/ades2bb_4thrun.simulation_new_trace.json sample-traces/ades2bb_4thrun.simulation_new_trace.json