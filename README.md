# Overview
This repository contains code for the paper:

Bryan Wilder, Bistra Dilkina, Milind Tambe. Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. AAAI Conference on Artificial Intelligence. 2019.
```
@inproceedings{wilder2019melding,
 author = {Wilder, Bryan},
 title = {Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization},
 booktitle = {Proceedings of the 33rd AAAI Conference on Artificial Intelligence},
 year = {2019}
}
```

Included are differentiable solvers for LPs and submodular maximization, along with code to run the experiments in the paper. You can [download](http://teamcore.usc.edu/people/bryanwilder/files/data_decisions_benchmarks.zip) the datasets from my website.

# Dependencies
* The linear programming experiments use the [Gurobi](http://www.gurobi.com/) solver.
* All code in the directory qpthlocal is derived from the [qpth](https://github.com/locuslab/qpth) library. It has been modified to support use of the Gurobi solver in the forward pass.
