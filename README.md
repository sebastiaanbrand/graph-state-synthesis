# Graph-state synthesis under LC+VD using BMC
Using bounded model checking (BMC), given a source graph and target graph, find a transformation from source to target using only local complementations (LC) (corresponding to single-qubit Clifford gates), vertex deletions (VD) (corresponding to single-qubit Pauli measurements), and optionally edge flips on a selection of pairs of nodes (corresponding to two-qubit CZ gates).

## Prerequisites
This project requires `Python 3` to run, and Make and a C/C++ compiler to build [Kissat](https://github.com/arminbiere/kissat), and [Rust](https://www.rust-lang.org/tools/install) to build the BMC encoder.


## Installation on Linux
1. Clone this repo (including submodules):
```shell
git clone --recurse-submodules <this repo's url>
```

2. We recommend running the code and installing dependencies within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). From the root of the repo run:
```shell
python -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:
```shell
pip install -r requirements.txt
```

4. Build encoder
```shell
cd bmc_encoder
maturin develop -r
cd ..
```

5. Build Kissat and test build:
```shell
cd extern/kissat
./configure && make test
cd ../..
```

6. The installation can be tested with
```shell
pytest
```


## Python example
The file [`example.py`](example.py) contains an example where a source and target graph are constructed, and bounded model checking is used to find a transformation between them.


## Command line interface

For individual source and target graphs, determining reachability using BMC 
```shell
python run_gs_bmc.py <source_graph.tgf> <target_graph.tgf>
```
for graphs given in the [Trivial Graph Format](https://en.wikipedia.org/wiki/Trivial_Graph_Format). The directory [`examples/`](examples/) contains the four graphs given in Figure 2 in the [paper](https://arxiv.org/pdf/2309.03593.pdf), in TGF.
For example:

```shell
python run_gs_bcm.py examples/graph2.tgf examples/graph4.tgf
```

By default the kissat solver is used. The specific solver can be chosen by adding `--solver {kissat|glucose4}`. Additionally, it is possible to only search over solutions where all the vertex deletions happen at the end by adding `--force_vds_end`. This can yield significant improvents in performance.


### Including edge flips

Edge flips can be included by specifying a set of pairs of nodes between which these are allowed in a JSON file. 
For example:

```shell
python run_gs_bmc.py examples/graph4.cnf examples/graph2.cnf
```
yields `Target is unreachable`. However, if we allow the edge flips specified in [`examples/allowed_flips.json`](examples/allowed_flips.json)
```shell
python run_gs_bmc.py examples/graph4.tgf examples/graph2.tgf --info examples/allowed_flips.json
```
we are able to find a transformation `['LC(0)', 'EF(0,2)']`.




## Paper
Most of the ideas in this repository are described in the paper "Quantum Graph-State Synthesis with SAT", Brand, S., Coopmans, T., Laarman, A. (2023) [[arXiv](https://doi.org/10.48550/arXiv.2309.03593)], presented at the Pragmatics of SAT 2023 workshop. To reproduce the plots from this paper, please refer to the [sat23-version](https://github.com/sebastiaanbrand/graph-state-synthesis/tree/sat23-version) of the repository.


## Acknowledgements
This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.
