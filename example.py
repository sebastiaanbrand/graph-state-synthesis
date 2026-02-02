"""
Example that runs bounded model checking on the two specified graphs.
"""
from graph_states import GraphFactory
from run_gs_bmc import binary_search, get_default_args

def main():
    """
    For two given graphs, find transformation using BMC.
    """

    # Set source and target graphs
    source = GraphFactory.get_star_graph(n=5)
    target = GraphFactory.get_complete_graph(n=5)
    target.set_edge(0, 1, False)
    target.set_edge(0, 2, False)
    target.set_edge(0, 3, False)
    target.set_edge(0, 4, False)
    assert source.num_nodes == target.num_nodes

    # CZ-gates can be allowed between pairs of nodes by
    # setting e.g. cz_gates = [(0,1),(3,4)]
    cz_gates = []

    # Run binary search over transformation depth (runs new BMC query at each depth)
    args = get_default_args()
    steps = binary_search(source, target, cz_gates, args)
    if steps == -1:
        print("Target is unreachable\n")
    else:
        print(f"Target is reachable in {steps} steps\n")


if __name__ == '__main__':
    main()
