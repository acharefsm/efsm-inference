# Generalisation

There are two files you can execute to generalise from a sampled EFSM:
  * gp_experiment.py - runs a scikit grid parameter optimisation experiment according to your 'param_grid' (you can also use this to use a different set of parameters if you create a vector instead of a matrix)
  * gp_generalise.py - uses the default parameters to generalise an EFSM

## Parameters

* -c = the sampled efsm you are generalising
* -s = seed
* -n_jobs (gp_experiment only) = number of grid iterations

## Example Usage

python gp_generalise.py -c H:\VASCO\ehW\drinks_register_guard_bounded_inferred.sampled.bis.dot