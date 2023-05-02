"""
CL script which run bounded model checking on the given input graph states.
"""
import time
import argparse
import z3
from pysat.solvers import Glucose4
from graph_states import Graph
from gsreachability_using_bmc import GraphEncoding, GraphStateBMC

t_enc = 0
t_solve = 0

parser = argparse.ArgumentParser()
parser.add_argument('source_file', metavar='source.cnf')
parser.add_argument('target_file', metavar='target.cnf')
parser.add_argument('--solver', default='z3', choices=['z3','glucose4'], action='store')
parser.add_argument('--info', default=None, metavar='info.json', action='store')
parser.add_argument('--statsfile', metavar='out.csv', action='store')


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


def run_bmc(source: Graph, target: Graph, cz_gates: list, steps: int, _solver: str, statsfile: str | None):
    """
    Do BMC for given number of steps.
    """
    # TODO: instead of running new BMC instance every time, modify previous instance.
    print(f"Running BMC on {source.name} for k={steps}")

    # 1. Encode BMC problem
    print("\tEncoding...")
    global t_enc
    t_start = time.time()
    gs_bmc = GraphStateBMC(source, target, steps, cz_gates)
    bmccnf = gs_bmc.generate_bmc_cnf()
    # TODO: make SolverWrapper class instead of these if statements
    if _solver == 'z3':
        solver = z3.Solver()
        for clause in bmccnf.clauses:
            solver.add(clause.to_formula())
    elif _solver == 'glucose4':
        solver = Glucose4(bootstrap_with=bmccnf.to_pysat_clauses())
    t_enc += time.time() - t_start


    # 2. Solve formula
    print("\tSolving...")
    global t_solve
    t_start = time.time()
    if _solver == 'z3':
        check = solver.check()
        check = check == z3.sat # have check contain True/False instead of sat/unsat
    elif _solver == 'glucose4':
        check = solver.solve()
    t_solve += time.time() - t_start

    # 3. Write results
    info = f"{source.name}, {source.num_nodes}, {round(t_enc,3)}, {round(t_solve,3)}, {_solver}, {check}, {steps}\n"
    if not statsfile is None:
        with open(statsfile, 'a', encoding='utf-8') as f:
            f.write(info)

    # 4. Check solution
    if _solver == 'z3':
        if check:
            print(gs_bmc.retrieve_operations(solver.model(), steps, source.num_nodes))
        return check
    elif _solver == 'glucose4':
        if check:
            print(solver.get_model())
        return check


def binary_search(source: Graph, target: Graph, cz_gates: list, solver: str, statsfile: str | None):
    """
    Binary seach over the number of operations.
    """
    max_depth = search_depth(source, target)
    print(f"Max search depth: {max_depth}")

    # 1. Search up for a SAT instance
    k = 1
    while k <= max_depth:
        if run_bmc(source, target, cz_gates, k, solver, statsfile):
            break
        k = k * 2

    # 2. Stop if no SAT instance was found
    if k > max_depth:
        # if max_depth was a power of two we've already checked k == max_depth
        if k == max_depth * 2:
            return -1
        # otherwise, check k=max_depth
        elif run_bmc(source, target, cz_gates, max_depth, solver, statsfile):
            k = max_depth
        else:
            return -1

    # 3. Search for smallest k which is still SAT
    lowest_k = k
    diff = k / 4
    k = k - round(diff)
    while round(diff) > 0:
        if run_bmc(source, target, cz_gates, k, solver, statsfile):
            lowest_k = k
            diff = diff / 2
            k = k - round(diff)
        else:
            diff = diff / 2
            k = k + round(diff)

    return lowest_k


def main():
    """
    Parses args and runs binary search BMC.
    """
    args = parser.parse_args()
    source = Graph.from_cnf(args.source_file)
    target = Graph.from_cnf(args.target_file)
    cz_gates = GraphEncoding.get_cz_from_file(args.info)
    assert source.num_nodes == target.num_nodes

    steps = binary_search(source, target, cz_gates, args.solver, args.statsfile)
    if steps == -1:
        print("Target is unreachable\n")
    else:
        print(f"Target is reachable in {steps} steps\n")


if __name__ == '__main__':
    main()
