"""
Code to generate graph state reachability benchmarks which can be run with both
bounded model checking and with BDDs.
"""
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from graph_states import Graph, GraphFactory

parser = argparse.ArgumentParser()
parser.add_argument('--dahlberg', action='store_true', default=False, help="Generate graphs from Dahlberg2020")
parser.add_argument('--rabbie', action='store_true', default=False, help="Generate network from Rabbie2022")
parser.add_argument('--ghz', action='store_true', default=False, help="Generate bench with source ER(n) and target GHZ(n)")
parser.add_argument('--ghz_k', metavar='k', action='store', default=0, help="Generate bench with source ER(n) and target GHZ(k)")
parser.add_argument('--cz_frac', metavar='p', action='store', default=0, type=float, help="Allow for CZ gates on a random n * p selection of edges")
parser.add_argument('--min_qubits', metavar='n', action='store', default=2, help="Minimum number of qubits (default 2)")
parser.add_argument('--max_qubits', metavar='n', action='store', default=30, help="Maximum number of qubits (default 30)")
parser.add_argument('--encodings', nargs='+', choices=['sat23','vds_end'], default=['sat23'], help="Which encoding(s) to benchmark (can be multiple) (default sat23)")
parser.add_argument('--solvers', nargs='+', choices=['kissat','glucose4'], default=['kissat'], help="Which solver(s) to benchmark (can be multiple) (default kissat)")
parser.add_argument('--rseed', metavar='r', action='store', default=42, help="Random seed for generating benchmarks (default 42, 0 sets no random seed)")
parser.add_argument('--timeout', metavar='t', action='store', default='30m', help="String indicating the timeout per BMC run (inc binary search).")

bmc_cl = "timeout {} python run_gs_bmc.py {} {} --solver {} --info {} --statsfile {}\n"

def generate_benchmarks(nqubits, p_source, source_f, target_f, cz_f, bench_name, args):
    """
    source_f(nqubits, p_source) -> Graph
    target_f(nqubits) -> Graph
    """

    if bench_name is None:
        folder = f"benchmarks/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        folder = f"benchmarks/{bench_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(folder).mkdir(parents=True, exist_ok=False)

    # prep results file
    bmc_csv = f"{folder}/bmc_results.csv"
    with open(bmc_csv, 'w', encoding='utf-8') as f:
        f.write("name, nqubits, enc_time, solve_time, encoding, solver, reachable, nsteps\n")

    bmc_cls = []

    _id = 0
    for n in nqubits:
        for p in p_source:
            _id += 1

            # 1. Encode BMC problem
            source = source_f(n, p)
            target = target_f(n)
            cz_gates = cz_f(n)
            assert source.num_nodes == target.num_nodes

            # 2. Write graphs as TGF files
            src_tgf = f"{folder}/{_id}_source.tgf"
            trg_tgf = f"{folder}/{_id}_target.tgf"
            with open(src_tgf, 'w', encoding='utf-8') as f:
                f.write(source.to_tgf())
            with open(trg_tgf, 'w', encoding='utf-8') as f:
                f.write(target.to_tgf())

            # 3. Write experiment info
            info = f"{folder}/{_id}_info.json"
            with open(info, 'w', encoding='utf-8') as f:
                setup = {'source' : source.name,
                         'target' : target.name,
                         'nqubits' : n,
                         'cz_gates' : cz_gates}
                json.dump(setup, f)

            # 4. Add CL command to run this experiment
            for solver in args.solvers:
                if 'pos23' in args.encodings:
                    bmc_cls.append(bmc_cl.format(args.timeout, src_tgf, trg_tgf, solver, info, bmc_csv))
                if 'vds_end' in args.encodings:
                    _solver = solver + ' --force_vds_end'
                    bmc_cls.append(bmc_cl.format(args.timeout, src_tgf, trg_tgf, _solver, info, bmc_csv))

    # Write bash script to run bmc experiments
    with open(f"{folder}/run_all_bmc.sh", 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n\n")
        f.writelines(bmc_cls)

    # return next free bench id
    return _id


def main():
    """
    Generate benchmarks
    """
    args = parser.parse_args()

    if args.rseed != 0:
        random.seed(args.rseed)

    if args.dahlberg:
        generate_benchmarks(nqubits = range(4, 51),
                            p_source = [0.5],
                            source_f = lambda n, p : GraphFactory.get_dist_hereditary_graph(n, p),
                            target_f = lambda n : GraphFactory.get_star_graph(n, 4),
                            cz_f = lambda n : [],
                            bench_name = 'dahlberg2020',
                            args=args)

    if args.rabbie:
        generate_benchmarks(nqubits = [14],
                            p_source = [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
                            source_f = lambda n, p : GraphFactory.get_rabbie2022_network('random', p),
                            target_f = lambda n : GraphFactory.get_rabbie2022_network('ghz'),
                            bench_name = 'rabbie2022',
                            cz_f = lambda n : [],
                            args=args)

    if args.ghz:
        generate_benchmarks(nqubits = range(int(args.min_qubits), int(args.max_qubits)+1),
                            p_source = [0.9, 0.95, 1.0],
                            source_f = lambda n, p : GraphFactory.get_erdos_renyi_graph(n, p),
                            target_f = lambda n : GraphFactory.get_star_graph(n),
                            cz_f = lambda n : Graph.random_edges(n, int(n*args.cz_frac)),
                            bench_name = 'ghz',
                            args=args)

    if int(args.ghz_k) > 0:
        generate_benchmarks(nqubits = range(int(args.ghz_k), int(args.max_qubits)+1),
                            p_source = [0.8, 0.8, 0.8, 0.8, 0.8],
                            source_f = lambda n, p : GraphFactory.get_erdos_renyi_graph(n, p),
                            target_f = lambda n : GraphFactory.get_star_graph(n, int(args.ghz_k)),
                            cz_f = lambda n : Graph.random_edges(n, int(n*args.cz_frac)),
                            bench_name = f'ghz_{args.ghz_k}{"_ef" if args.cz_frac != 0 else ""}',
                            args=args)


if __name__ == '__main__':
    main()
