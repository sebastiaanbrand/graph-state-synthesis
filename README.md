# Graph-state synthesis under LC+VD using BMC
Given a source graph, synthesize a desired target graph using only local complementations (LC), corresponding to single-qubit Clifford gates, vertex deletions (VD), corresponding to single-qubit Pauli measurements, and optionally a small amount of edge flips, corresponding the two-qubit CZ gate.

## Prerequisites
This project requires `Python 3` to run, and `Make` and a `C/C++` compiler to build [Z3](https://github.com/Z3Prover/z3).


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

3. Install Z3 and Python bindings:
```shell
cd extern/z3/
python scripts/mk_make.py --python
cd build
make
make install
```

4. Install other Python packages:
```shell
pip install -r requirements.txt
```

5. The installation can be tested with
```shell
pytest
```


## Reproducing experiments
1. To **generate the benchmarks** used in the paper, run the following. The source and target graphs are stored in DIMACS `.cnf` files. A CNF formula for the transition relation is also stored as as `.cnf` file, but is currently unused.
```shell
python generate_benchmarks.py --ghz_k 4 --max_qubits 20 --timeout 30m
python generate_benchmarks.py --ghz_k 4 --max_qubits 20 --cz_frac 0.5 --timeout 30m
python generate_benchmarks.py --rabbie --timeout 30m
```

2. To **run bounded model checking** on these benchmarks run the following on each of the three benchmark folders generated in the previous step. The results are written to `.csv` files in the corresponding benchmark folder.
```shell
bash benchmarks/<benchmark_folder_name>/run_all_bmc.sh
```

3. To **generate the plots** from the benchmark data, run the following on each of the three benchmark folders.
```
python generate_plots.py benchmarks/<benchmark_folder_name>
```


## Acknowledgements
This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.
