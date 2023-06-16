# Graph-state synthesis under LC+VD using BMC
Given a source graph, synthesize a desired target graph using only local complementations (LC), corresponding to single-qubit Clifford gates, vertex deletions (VD), corresponding to single-qubit Pauli measurements, and optionally a small amount of edge flips, corresponding the two-qubit CZ gate.

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
