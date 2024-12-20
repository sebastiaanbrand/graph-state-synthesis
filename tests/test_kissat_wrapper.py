"""
Some test cases to check if the Kissat wrapper is working correctly.
"""
from kissat_wrapper import Kissat

from graph_states import GraphFactory
from gsreachability_using_bmc import GraphStateBMC


def test_lc1():
    """
    Simple local complementation test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)
   
    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert s.solve()


def test_lc2():
    """
    Simple local complementation test.
    """
    steps = 1
    nqubits = 4

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    source.set_edge(0, 1, True)
    source.set_edge(0, 2, True)
    target.set_edge(0, 1, True)
    target.set_edge(0, 2, True)
    target.set_edge(1, 2, True)
   
    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert s.solve()


def test_vd1():
    """
    Simple vertex deletion test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)
    target.set_edge(1, 2, False)

    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert s.solve()


def test_vd2():
    """
    Simple vertex deletion test.
    """
    steps = 2
    nqubits = 4

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)

    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert s.solve()


def test_unsat1():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)

    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert not s.solve()


def test_unsat2():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 4

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)

    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert not s.solve()


def test_unsat3():
    """
    Simple unreachability test.
    """
    steps = 3
    nqubits = 4

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)
    target.set_edge(1, 3, False)

    gs_bmc = GraphStateBMC(source, target, steps)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    s = Kissat(bmc_cnf)
    assert not s.solve()