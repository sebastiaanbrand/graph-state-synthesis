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

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

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

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'LC(0)'


def test_lc3():
    """
    star_graph(0) --> star_graph(3)
    """
    steps = 1
    nqubits = 5

    source = GraphFactory.get_star_graph(nqubits, center=0)
    target = GraphFactory.get_star_graph(nqubits, center=3)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()

    assert not is_sat


def test_lc4():
    """
    star_graph(0) --> star_graph(3)
    """
    steps = 2
    nqubits = 5

    source = GraphFactory.get_star_graph(nqubits, center=0)
    target = GraphFactory.get_star_graph(nqubits, center=3)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'LC(0)'
    assert sequence[1] == 'LC(3)'


def test_lc5():
    """
    star_graph(0) --> star_graph(3)
    """
    steps = 3
    nqubits = 5

    source = GraphFactory.get_star_graph(nqubits, center=0)
    target = GraphFactory.get_star_graph(nqubits, center=3)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert 'LC(0)' in sequence
    assert 'LC(3)' in sequence


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

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

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

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert 'VD(0)' in sequence
    assert 'VD(3)' in sequence


def test_id1():
    """
    Test identity operation.
    """
    steps = 1
    nqubits = 5

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'Id'

def test_ef1():
    """
    Simple EF test.
    """
    steps = 1
    nqubits = 5
    allowed_efs = [(1,2),(2,3)]

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(2, 3, True)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, allowed_efs)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps, allowed_efs)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'EF(2,3)'


def test_ef1():
    """
    Simple EF test.
    """
    steps = 2
    nqubits = 5
    allowed_efs = [(1,2),(2,3),(0,4),(1,4),(1,3)]

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)
    target.set_edge(1, 4, True)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, allowed_efs)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps, allowed_efs)

    assert is_sat
    assert len(sequence) == steps
    assert 'EF(1,2)' in sequence
    assert 'EF(1,4)' in sequence



def test_ef2():
    """
    Test only two nodes.
    """
    steps = 1
    nqubits = 2
    allowed_efs = [(0,1)]
    print(allowed_efs)

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, allowed_efs)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps, allowed_efs)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0] == 'EF(0,1)'


def test_ef_lc1():
    """
    Simple EF + LC test
    """
    steps = 8
    nqubits = 7
    allowed_efs = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, allowed_efs)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps, allowed_efs)

    assert is_sat
    assert len(sequence) == steps
    assert 'EF(0,1)' in sequence
    assert 'EF(0,2)' in sequence
    assert 'EF(0,3)' in sequence
    assert 'EF(0,4)' in sequence
    assert 'EF(0,5)' in sequence
    assert 'EF(0,6)' in sequence
    assert sequence[-1] == 'LC(0)'


def test_ef_lc2():
    """
    Simple EF + LC test
    """
    steps = 8
    nqubits = 7
    allowed_efs = [(0,3),(1,3),(2,3),(3,4),(3,5),(3,6)]

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, allowed_efs)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps, allowed_efs)

    assert is_sat
    assert len(sequence) == steps
    assert 'EF(0,3)' in sequence
    assert 'EF(1,3)' in sequence
    assert 'EF(2,3)' in sequence
    assert 'EF(3,4)' in sequence
    assert 'EF(3,5)' in sequence
    assert 'EF(3,6)' in sequence
    assert sequence[-1] == 'LC(3)'


def test_lc_vd1():
    """
    Simple tests of LCs + VDs
    """
    steps = 2
    nqubits = 5

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    assert sequence[0].startswith('LC')
    assert sequence[1].startswith('VD')


def test_lc_vd2():
    """
    Simple tests of LCs + VDs, with force_vds_end
    """
    steps = 5
    nqubits = 5

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, force_vds_end=True)
    s = Kissat(dimacs)
    is_sat = s.solve()
    sequence = encoder.decode_model(s.model, nqubits, steps)

    assert is_sat
    assert len(sequence) == steps
    for op in sequence:
        assert op.startswith('VD')


def test_unsat1():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
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

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
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

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps)
    s = Kissat(dimacs)
    is_sat = s.solve()

    assert not is_sat


def test_unsat4():
    """
    Should be unreachable with 6 steps.
    """
    steps = 6
    nqubits = 7
    allowed_efs = [(0,3),(1,3),(2,3),(3,4),(3,5),(3,6)]

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_star_graph(nqubits, center=0)

    dimacs = encoder.encode_bmc(source.to_tgf(), target.to_tgf(), nqubits, steps, allowed_efs)
    s = Kissat(dimacs)
    is_sat = s.solve()

    assert not is_sat
