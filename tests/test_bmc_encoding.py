"""
Some test cases for the graph state reachability BMC encoding.
"""
import z3
from graph_states import GraphFactory
from gsreachability_using_bmc import GraphStateBMC, GraphEncoding, GSOps


USE_LEQ_OPTIONS = [False, True]


def test_leq():
    """
    Test y <= b encoding for 4 bits.
    """
    op_encoder = GSOps(15, 0)
    assert len(op_encoder.y_vars) == 4

    # exhaustively check for all values of b and assignments to y
    for b in range(0, 16):
        for k in range(0, 16):
            s = z3.Solver()
            # add contraint vec(y) <= b
            for clause in op_encoder.clauses_vec_leq(b):
                s.add(clause.to_formula())
            # add constraint vec(y) == k
            for clause in op_encoder.clauses_y_eq(k):
                s.add(clause.to_formula())
            # should be sat iff k <= b
            if k <= b:
                assert s.check() == z3.sat
            else:
                assert s.check() == z3.unsat


def test_lc1():
    """
    Simple local complementation test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 1
        assert ops[0] == 'LC(2)'


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

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 1
        assert ops[0] == 'LC(0)'


def test_lc3():
    """
    Simple local complementation test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 1
        assert ops[0] == 'LC(2)'


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

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 1
        assert ops[0] == 'VD(1)'


def test_vd2():
    """
    Simple vertex deletion test.
    """
    steps = 2
    nqubits = 4

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 2
        assert 'VD(0)' in ops
        assert not 'VD(1)' in ops
        assert not 'VD(2)' in ops
        assert 'VD(3)' in ops


def test_lc_vd():
    """
    Test LC + VD.
    """
    steps = 2
    nqubits = 4

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    source.set_edge(0, 1, True)
    source.set_edge(0, 2, True)
    target.set_edge(1, 2, True)

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 2
        assert ops[0] == 'LC(0)'
        assert ops[1] == 'VD(0)'


def test_unsat1():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.unsat


def test_unsat2():
    """
    Simple unreachability test.
    """
    steps = 1
    nqubits = 4

    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.unsat


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

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.unsat


def test_ef1():
    """
    Simple edge flip test.
    """
    steps = 1
    nqubits = 3

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(0, 1, True)
    allowed_ef = [(0, 1)]

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps, allowed_ef)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        assert len(ops) == 1
        assert ops[0] == 'EF(0,1)'


def test_ef2():
    """
    Simple edge flip test.
    """
    steps = 2
    nqubits = 4

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(0, 1, True)
    target.set_edge(1, 3, True)
    allowed_ef = [(0, 1), (1, 2), (1, 3)]

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps, allowed_ef)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        print(ops)
        assert len(ops) == 2
        assert 'EF(0,1)' in ops
        assert 'EF(1,3)' in ops


def test_ef3():
    """
    (0)---(1)   EF(1,2)   (0)---(1)   LC(1)   (0)---(1)   VD(1)   (0)   (1)
                  -->          /       -->      |  /      -->      |
      (2)                   (2)                 (2)                (2)
    """
    steps = 3
    nqubits = 3

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    source.set_edge(0, 1, True)
    target.set_edge(0, 2, True)
    allowed_ef = [(1, 2)]

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps, allowed_ef)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.sat

        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        print(ops)
        assert len(ops) == 3
        assert ops[0] == 'EF(1,2)'
        assert ops[1] == 'LC(1)'
        assert ops[2] == 'VD(1)'


def test_ef_unsat1():
    """
    Edge flip test where the target is unreachable.
    """
    steps = 2
    nqubits = 5

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(2, 4, True)
    allowed_ef = [(0, 1), (1, 4), (2, 3)]

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps, allowed_ef)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.unsat


def test_ef_unsat2():
    """
    Edge flip test where the target is unreachable.
    """
    steps = 1
    nqubits = 5

    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(2, 4, True)
    target.set_edge(1, 2, True)
    allowed_ef = [(1, 2), (2, 4)]

    for use_leq in USE_LEQ_OPTIONS:
        GraphEncoding.use_leq = use_leq

        s = z3.Solver()
        gs_bmc = GraphStateBMC(source, target, steps, allowed_ef)
        bmc_cnf = gs_bmc.generate_bmc_cnf()
        for clause in bmc_cnf.clauses:
            s.add(clause.to_formula())

        assert s.check() == z3.unsat
