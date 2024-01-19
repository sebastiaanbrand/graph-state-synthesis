# Graph-state synthesis under LC+VD using BMC
Using bounded model checking (BMC), given a source graph and target graph, find a transformation from source to target using only local complementations (LC) (corresponding to single-qubit Clifford gates), vertex deletions (VD) (corresponding to single-qubit Pauli measurements), and optionally edge flips on a selection of pairs of nodes (corresponding to two-qubit CZ gates).

## Prerequisites
This project requires `Python 3` to run, and `Make` and a `C/C++` compiler to build [Z3](https://github.com/Z3Prover/z3) and [Kissat](https://github.com/arminbiere/kissat).


## Installation on Linux
1. Clone this repo (including submodules):
```shell
git clone --recursive <this repo's url>
```

2. We recommend running the code and installing dependencies within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). From the root of the repo run:
```shell
python -m venv .venv
source .venv/bin/activate
```

3. Build Kissat and test build:
```shell
cd extern/kissat
./configure && make test
cd ../..
```


4. Install Z3 and Python bindings:
```shell
cd extern/z3/
python scripts/mk_make.py --python
cd build
make
make install
cd ../../..
```

5. Install other Python packages:
```shell
pip install -r requirements.txt
```

6. The installation can be tested with
```shell
pytest
```

## Running on individual graphs

For individual source and target graphs, determining reachability using BMC 
```shell
python run_gs_bmc.py <source_graph.cnf> <target_graph.cnf>
```

The folder [`examples/`](examples/) contains the four graphs given in Figure 2 in the [paper](https://arxiv.org/pdf/2309.03593.pdf). These graphs are encoded in [DIMACS CNF format](https://jix.github.io/varisat/manual/0.2.0/formats/dimacs.html), where each edge is associated with a unique variable (these need to be sequential odd integers), and a positive (negative) occurrence of this variable indicates the presence (absence) of this edge in the graph.
For example:

```shell
python run_gs_bcm.py examples/graph2.cnf examples/graph4.cnf
```

By default the `z3` solver is used. The specific solver can be chosen by adding `--solver {z3|kissat|glucose4}`. If the target is found to be reachable, the actual sequence of LCs+VDs can be extracted from the satisfying assignment. However, this is currently only implemented for when the `z3` solver is used.


### Including edge flips

Edge flips can be included by specifying a set of pairs of nodes between which these are allowed in a JSON file. 
For example:

```shell
python run_gs_bmc.py examples/graph4.cnf examples/graph2.cnf
```
yields `Target is unreachable`. However, if we allow the edge flips specified in [`examples/allowed_flips.json`](examples/allowed_flips.json)
```shell
python run_gs_bmc.py examples/graph4.cnf examples/graph2.cnf --info examples/allowed_flips.json
```
we are able to find a transformation `['LC(0)', 'EF(0,2)']`.




## Reproducing experiments
1. To **generate the benchmarks** used in the paper, run the following. The source and target graphs are stored in DIMACS `.cnf` files. A CNF formula for the transition relation is also stored as as `.cnf` file, but is currently unused.
```shell
python generate_benchmarks.py --ghz_k 4 --max_qubits 20 --timeout 30m
python generate_benchmarks.py --ghz_k 4 --max_qubits 20 --cz_frac 0.5 --timeout 30m
python generate_benchmarks.py --rabbie --timeout 30m
```

2. To **run bounded model checking** on these benchmarks run the following on each of the three benchmark folders generated in the previous step. The results are written to `.csv` files in the corresponding benchmark folder. Depending on hardware, for the first two sets this will take ~30-40h each, while for the third set this will take ~10h. Terminating the script early yields partial results (starting at the lowest number of qubits) which can still be plotted.
```shell
bash benchmarks/<benchmark_folder_name>/run_all_bmc.sh
```

3. To **generate the plots** from the benchmark data, run the following on each of the three benchmark folders. This generates a number of plots inside the selected benchmark folder.
```
python generate_plots.py benchmarks/<benchmark_folder_name>
```


## Acknowledgements
This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.
