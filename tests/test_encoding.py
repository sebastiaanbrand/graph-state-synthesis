"""
Some test cases to check if the Kissat wrapper is working correctly.
"""
import gs_bmc_encoder as encoder
from kissat_wrapper import Kissat
from graph_states import GraphFactory


def test_lc1():
    """
    Simple local complementation test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)
   
    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.get_operations(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'LC(2)'


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
   
    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.get_operations(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'LC(0)'


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

    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.get_operations(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'VD(1)'


def test_vd2():
    """
    Simple vertex deletion test.
    """
    steps = 2
    nqubits = 4

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)

    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.get_operations(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert 'VD(0)' in sequence
    assert 'VD(3)' in sequence


def test_unsat1():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)

    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()

    assert not is_sat


def test_unsat2():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 4

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)

    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()

    assert not is_sat


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

    dimacs = encoder.encode_from_strings(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()

    assert not is_sat
