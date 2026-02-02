"""
CL script which run bounded model checking on the given input graph states.
"""
import time
import json
import argparse
from pysat.solvers import Glucose4
from pysat.formula import CNF
from graph_states import Graph
from kissat_wrapper import Kissat
import gs_bmc_encoder as encoder

t_enc = 0
t_solve = 0

parser = argparse.ArgumentParser()
parser.add_argument('source_file', metavar='source.cnf')
parser.add_argument('target_file', metavar='target.cnf')
parser.add_argument('--solver', default='kissat', choices=['glucose4','kissat'], action='store')
parser.add_argument('--force_vds_end', default=False, action='store_true')
parser.add_argument('--info', default=None, metavar='info.json', action='store')
parser.add_argument('--statsfile', metavar='out.csv', action='store')


def get_default_args():
    """
    Return an argparse Namespace with the default arguments.
    """
    args = parser.parse_args(['',''])
    return args


def _get_max_lc(n: int):
    """
    Get the maximum required number of LCs for a graph of n nodes.
    """
    if (n % 2) == 0:
        return int(n/2) * 3
    else:
        return int((n-1)/2) * 3 + 1

def search_depth(source: Graph, target: Graph):
    """
    Compute maximum required search depth depending on source and target graph.
    """
    n_source = len(source.get_non_isolated_nodes())
    n_target = len(target.get_non_isolated_nodes())

    depth = _get_max_lc(n_source) + (n_source - n_target)# + _get_max_lc(n_target)

    return depth


def run_bmc(source: Graph, target: Graph, allowed_efs: list, steps: int, args):
    """
    Do BMC for given number of steps.
    """
    # TODO: instead of running new BMC instance every time, modify previous instance.
    print(f"Running BMC on {source.name} for k={steps}")

    # 1. Encode BMC problem
    print("\tEncoding...")
    global t_enc
    t_start = time.time()
    num_nodes = max(source.num_nodes, target.num_nodes) # add isolated nodes to make num nodes equal
    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(),
                                num_nodes, steps,
                                allowed_efs=allowed_efs, force_vds_end=args.force_vds_end)
    if args.solver == 'glucose4':
        solver = Glucose4(bootstrap_with=CNF(from_string=dimacs))
    elif args.solver == 'kissat':
        solver = Kissat(dimacs)
    else:
        raise ValueError(f"Invalid solver '{args.solver}'")
    t_enc += time.time() - t_start


    # 2. Solve formula
    print("\tSolving...")
    global t_solve
    t_start = time.time()
    if args.solver == 'glucose4':
        is_sat = solver.solve()
        t_solve += time.time() - t_start
    elif args.solver == 'kissat':
        is_sat = solver.solve()
        t_solve += solver.solve_time

    # 3. Write results
    encoding = 'sat23' # TODO: don't hardcode
    if args.force_vds_end:
        encoding += '-vds_end'
    info = f"{source.name}, {source.num_nodes}, {round(t_enc,3)}, {round(t_solve,3)}, {encoding}, {args.solver}, {is_sat}, {steps}\n"
    if not args.statsfile is None:
        with open(args.statsfile, 'a', encoding='utf-8') as f:
            f.write(info)

    # 4. Output solution?
    if is_sat:
        sequence = encoder.decode_model(solver.get_model(), source.num_nodes, steps, allowed_efs)
        print(sequence)
    return is_sat


def binary_search(source: Graph, target: Graph, cz_gates: list, args):
    """
    Binary seach over the number of operations.
    """
    max_depth = search_depth(source, target)
    if max_depth <= 0:
        print(f"Source contains too many isolated nodes, target is unreachable under LC+VD.")
        return -1
    print(f"Max search depth: {max_depth}")

    # 1. Search up for a SAT instance
    k = 1
    while k <= max_depth:
        if run_bmc(source, target, cz_gates, k, args):
            break
        k = k * 2

    # 2. Stop if no SAT instance was found
    if k > max_depth:
        # if max_depth was a power of two we've already checked k == max_depth
        if k == max_depth * 2:
            return -1
        # otherwise, check k=max_depth
        elif run_bmc(source, target, cz_gates, max_depth, args):
            k = max_depth
        else:
            return -1

    # NOTE: no need to search for smallest k because any solution can be turned
    # into O(1) gates + measurements per qubit.
    return k


def get_cz_from_file(file: str):
        """
        Get allowed CZ gates from info.json file.
        """
        if file is None:
            return []

        with open(file, 'r', encoding='utf-8') as f:
            info = json.load(f)
            cz_gates = []
            for node_pair in info['cz_gates']:
                assert len(node_pair) == 2
                cz_gates.append(tuple(node_pair))
            return cz_gates

def main():
    """
    Parses args and runs binary search BMC.
    """
    args = parser.parse_args()
    source = Graph.from_tgf(args.source_file)
    target = Graph.from_tgf(args.target_file)
    cz_gates = get_cz_from_file(args.info)

    steps = binary_search(source, target, cz_gates, args)
    if steps == -1:
        print("Target is unreachable\n")
    else:
        print(f"Target is reachable in {steps} steps\n")


if __name__ == '__main__':
    main()
