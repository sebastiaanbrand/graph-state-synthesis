"""
Compute reachability (and corresponding path) for quantum graph states using SAT.
"""
import math
import json
import z3
from cnf import CNF, Clause, Variable, Literal
from graph_states import Graph, GraphFactory


def bits_to_int(map_a):
    """
    Given a map_a, e.g. {2 : 1, 1 : 1, 0 : 0}, return the integer interpretation
    of this map, in the case of the example 110_bin = 5_dec.
    """
    res = 0
    for k, v in map_a.items():
        res += v * 2**k
    return res



class GSOps:
    """
    Hold some functions for encoding/decoding the operation identifiers.
    (y and z vars in the paper)
    """

    # NOTE: changes made here also require changes in num_opvars() and unused_opids()
    LC = 0 # local complementation (Cliff gate)
    VD = 1 # vertex deletion (measurement)
    EF = 2 # edge flip (CZ gate)


    def __init__(self, num_nodes, tstep, allow_cz=False):
        self.num_nodes = num_nodes
        self.allow_ef = allow_cz
        self.allowed_ef_edges = []
        self.y_vars = [Variable(variable_name=f'y_{j}_{tstep}') for j in range(self.num_nodevars)]
        self.y_vars.reverse()
        self.z_vars = [Variable(variable_name=f'z_{j}_{tstep}') for j in range(self.num_opvars)]
        self.z_vars.reverse()


    @property
    def num_nodevars(self):
        """
        Get the number of variables needed to label a node.
        """
        nbits = math.ceil(math.log2(self.num_nodes + 1))
        return nbits


    @property
    def num_opvars(self):
        """
        Get the number of vars needed to encode an op
        """
        if self.allow_ef is True:
            return 2
        else:
            return 1


    @property
    def opvars(self):
        """
        y and z vars.
        """
        return self.y_vars + self.z_vars


    @property
    def unused_opids(self):
        """
        Get the unused op IDs.
        """
        if self.allow_ef is True:
            return [3]
        else:
            return []


    def allow_ef_for_edges(self, edges):
        """
        Allow edge flip (CZ gates) of the given edges.
        """
        if self.allow_ef is False:
            raise ValueError("Class should be initialized with allow_cz")
        if len(edges) > self.num_nodes:
            raise ValueError("Currently don't allow more than num_node CZs")
        self.allowed_ef_edges = edges


    def get_name(self, int_y, int_z):
        """
        Returns the name of the operation identified by the given assignment to
        y and z variables.
        """
        if int_y >= self.num_nodes:
            return 'I'
        if int_z == GSOps.LC:
            return f'LC({int_y})'
        elif int_z == GSOps.VD:
            return f'VD({int_y})'
        elif int_z == GSOps.EF:
            if self.allow_ef is False:
                raise ValueError("CZ gates should be disabled")
            if int_y < len(self.allowed_ef_edges):
                e = self.allowed_ef_edges[int_y]
                return f'EF({e[0]},{e[1]})'
            else:
                return 'I'
        else:
            return 'I'


    def clause_vec_neq(self, y_val=None, z_val=None):
        """
        Returns a clause which encodes y_vars != y_val  v  z_vars != z_val
        """
        res = Clause()

        # TODO: clean code duplication between y and z
        # constraint on y_vars
        if not y_val is None:
            binary_string = f'{y_val:b}'.zfill(len(self.y_vars))
            assert len(binary_string) == len(self.y_vars)
            for i, b_i in enumerate(binary_string):
                if b_i == '0':
                    res.add_literal(Literal(variable=self.y_vars[i], negated=False))
                else:
                    res.add_literal(Literal(variable=self.y_vars[i], negated=True))

        # constraint on z_vars
        if not z_val is None:
            binary_string = f'{z_val:b}'.zfill(len(self.z_vars))
            assert len(binary_string) == len(self.z_vars)
            for i, b_i in enumerate(binary_string):
                if b_i == '0':
                    res.add_literal(Literal(variable=self.z_vars[i], negated=False))
                else:
                    res.add_literal(Literal(variable=self.z_vars[i], negated=True))

        return res


    def clauses_vec_eq(self, y_val):
        """
        Returns len(y_vars) clauses which encodes vec(y_vars) == bin(y_val).
        """
        clauses = []
        binary_string = f'{y_val:b}'.zfill(len(self.y_vars))
        assert len(binary_string) == len(self.y_vars)
        for i, b_i in enumerate(binary_string):
            clause = Clause()
            if b_i == '0':
                clause.add_literal(Literal(variable=self.y_vars[i], negated=True))
            else:
                clause.add_literal(Literal(variable=self.y_vars[i], negated=False))
            clauses.append(clause)
        return clauses


    def clauses_vec_leq(self, y_val, z_val=None):
        """
        Returns len(y_vars) clauses which encode vec(y) <= bin(y_val)

        Note that in the math we order [n, ..., 1, 0], however here the indices
        are reversed:
        - y_vars are ordered [y_n, ..., y_1, y_0] so y_vars[0] gives y_n
        - binary_string =    [b_n, ..., b_1, b_0] so binary_string[0] gives b_n
        """
        clauses = []
        bin_y = f'{y_val:b}'.zfill(len(self.y_vars))
        assert len(bin_y) == len(self.y_vars)
        for i, b_i in enumerate(bin_y):
            if b_i == '0':
                clause = Clause()
                # 1. add part of constraint vec(y) <= bin(y_val)
                clause.add_literal(Literal(variable=self.y_vars[i], negated=True))
                for j in range(0, i): # indices are reversed so j > i becomes j < i
                    if bin_y[j] == '1':
                        clause.add_literal(Literal(variable=self.y_vars[j], negated=True))

                # 2. add constraint on z_vars #TODO: code dup w/ clause_vec_neq
                if not z_val is None:
                    bin_z = f'{z_val:b}'.zfill(len(self.z_vars))
                    assert len(bin_z) == len(self.z_vars)
                    for i, b_i in enumerate(bin_z):
                        if b_i == '0':
                            clause.add_literal(Literal(variable=self.z_vars[i], negated=False))
                        else:
                            clause.add_literal(Literal(variable=self.z_vars[i], negated=True))

                clauses.append(clause)

        return clauses




class GraphEncoding(Graph):
    """
    Boolean encoding of a graph state and its transitions under LC and VD.
    """

    # Use encoding for [y <= b], instead of [\forall b > y : y != b]
    use_leq = True

    def __init__(self, tstep : str, graph, allowed_ef_edges = None):
        super().__init__(graph.num_nodes, from_graph=graph)
        self.tstep = tstep

        self._graph = graph # TODO: either store graph here or use inheritance, not both
        self._state_cnf = CNF()
        self._create_state_cnf()

        if allowed_ef_edges is None or len(allowed_ef_edges) == 0:
            self.op_encoder = GSOps(graph.num_nodes, tstep, allow_cz=False)
        else:
            self.op_encoder = GSOps(graph.num_nodes, tstep, allow_cz=True)
            self.op_encoder.allow_ef_for_edges(allowed_ef_edges)


    @staticmethod
    def get_cz_from_file(file: str):
        """
        Get allowed CZ gates from info.json file.
        """
        if file is None:
            return []
        
        with open(file, 'r', encoding='utf-8') as f:
            info = json.load(f)
            return info['cz_gates']


    def _create_state_cnf(self):
        """
        Initialize a CNF encoding of the given graph given to constructor.
        """
        for (v, w) in Graph.all_possible_edges(self.num_nodes):
            var = Variable(variable_name=f'x_{v}_{w}_{self.tstep}')
            literal = Literal(variable=var, negated=False)
            self.set_edge(v, w, var)
            if self._graph.get_edge(v, w):
                clause = Clause(literals=[literal])
            else:
                literal.negated = True
                clause = Clause(literals=[literal])
            self._state_cnf.add_clause(clause)


    @property
    def cnf(self):
        """
        CNF encoding of the graph state given to constructor.
        """
        return self._state_cnf


    def get_state_formula(self):
        """
        Get the graph state as a z3 formula.
        """
        return self._state_cnf.to_formula()


    def get_update_formula_with(self, other, dummy_clauses=False):
        """
        Returns a CNF formula which encodes the 1-step transitions from the
        current GraphEncoding (self) to the given GraphStateEncoding (other),
        allowing for local complementations (LC) and vertex deletion (VD).

        Args:
            other: Graph encoding containing variables at time t+1
            dummy_clauses: Add dummy clauses (a v ~a) for helper variables.
        """

        if self.num_nodes != other.num_nodes:
            raise ValueError

        res = CNF()

        # formula part 1a (LC)
        for m in range(self.num_nodes):
            for (v, w) in Graph.all_possible_edges(self.num_nodes):
                if v != m and w != m:
                    clause_a = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_a.add_literal(Literal(self.get_edge(v, m)).Not())
                    clause_a.add_literal(Literal(self.get_edge(w, m)).Not())
                    clause_a.add_literal(Literal(other.get_edge(v, w)))
                    clause_a.add_literal(Literal(self.get_edge(v, w)))
                    res.add_clause(clause_a)

                    clause_b = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_b.add_literal(Literal(self.get_edge(v, m)).Not())
                    clause_b.add_literal(Literal(self.get_edge(w, m)).Not())
                    clause_b.add_literal(Literal(other.get_edge(v, w)).Not())
                    clause_b.add_literal(Literal(self.get_edge(v, w)).Not())
                    res.add_clause(clause_b)

                    clause_c = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_c.add_literal(Literal(self.get_edge(v, m)))
                    clause_c.add_literal(Literal(other.get_edge(v, w)))
                    clause_c.add_literal(Literal(self.get_edge(v, w)).Not())
                    res.add_clause(clause_c)

                    clause_d = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_d.add_literal(Literal(self.get_edge(w, m)))
                    clause_d.add_literal(Literal(other.get_edge(v, w)))
                    clause_d.add_literal(Literal(self.get_edge(v, w)).Not())
                    res.add_clause(clause_d)

                    clause_e = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_e.add_literal(Literal(self.get_edge(v, m)))
                    clause_e.add_literal(Literal(other.get_edge(v, w)).Not())
                    clause_e.add_literal(Literal(self.get_edge(v, w)))
                    res.add_clause(clause_e)

                    clause_f = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_f.add_literal(Literal(self.get_edge(w, m)))
                    clause_f.add_literal(Literal(other.get_edge(v, w)).Not())
                    clause_f.add_literal(Literal(self.get_edge(v, w)))
                    res.add_clause(clause_f)
                else:
                    clause_a = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_a.add_literal(Literal(other.get_edge(v, w)))
                    clause_a.add_literal(Literal(self.get_edge(v, w)).Not())
                    res.add_clause(clause_a)

                    clause_b = self.op_encoder.clause_vec_neq(m, GSOps.LC)
                    clause_b.add_literal(Literal(other.get_edge(v, w)).Not())
                    clause_b.add_literal(Literal(self.get_edge(v, w)))
                    res.add_clause(clause_b)

        # formula part 1b (VD)
        for m in range(self.num_nodes):
            for (v, w) in Graph.all_possible_edges(self.num_nodes):
                if v != m and w != m:
                    # Edge not connected to m: (a == VD(m)) --> (x <==> x')
                    # (a == VD(m)) --> (x --> x')
                    clause_a = self.op_encoder.clause_vec_neq(m, GSOps.VD)
                    clause_a.add_literal(Literal(self.get_edge(v, w)).Not())
                    clause_a.add_literal(Literal(other.get_edge(v, w)))
                    res.add_clause(clause_a)

                    # (a == VD(m)) --> (x' --> x)
                    clause_b = self.op_encoder.clause_vec_neq(m, GSOps.VD)
                    clause_b.add_literal(Literal(self.get_edge(v, w)))
                    clause_b.add_literal(Literal(other.get_edge(v, w)).Not())
                    res.add_clause(clause_b)
                elif v == m and w == m:
                    raise AssertionError("v should never equal w")
                elif v == m or w == m:
                    # Edge connected to m, delete
                    # (a == VD(m)) --> (x' == False)
                    clause_a = self.op_encoder.clause_vec_neq(m, GSOps.VD)
                    clause_a.add_literal(Literal(other.get_edge(v, w)).Not())
                    res.add_clause(clause_a)

        # formula part 1c (EF)
        for idx, (j, k) in enumerate(self.op_encoder.allowed_ef_edges):
            # add (y = idx ^ z = EF) --> x_{jk}' <==> ~x_{jk}
            clause_a = self.op_encoder.clause_vec_neq(y_val=idx, z_val=GSOps.EF)
            clause_a.add_literal(Literal(self.get_edge(j, k)))
            clause_a.add_literal(Literal(other.get_edge(j, k)))
            res.add_clause(clause_a)

            clause_b = self.op_encoder.clause_vec_neq(y_val=idx, z_val=GSOps.EF)
            clause_b.add_literal(Literal(self.get_edge(j, k)).Not())
            clause_b.add_literal(Literal(other.get_edge(j, k)).Not())
            res.add_clause(clause_b)

            # add (y = idx ^ z = EF) --> x_{uv}' <==> x_{uv}
            for (u, v) in Graph.all_possible_edges(self.num_nodes):
                if not Graph.edge_equal((u, v), (j, k)):
                    clause_a = self.op_encoder.clause_vec_neq(y_val=idx, z_val=GSOps.EF)
                    clause_a.add_literal(Literal(self.get_edge(u, v)))
                    clause_a.add_literal(Literal(other.get_edge(u, v)).Not())
                    res.add_clause(clause_a)

                    clause_b = self.op_encoder.clause_vec_neq(y_val=idx, z_val=GSOps.EF)
                    clause_b.add_literal(Literal(self.get_edge(u, v)).Not())
                    clause_b.add_literal(Literal(other.get_edge(u, v)))
                    res.add_clause(clause_b)


        # formula part 2: for all invalid "op IDs" the graph should remain the same
        # 2a. Handle LC + VD, nodeIDs between num_nodes and 2^node_bits
        if GraphEncoding.use_leq:
            # add (y > ops) --> Id(x,x') <==> ( y <= ops  v  Id(x,x') )
            for clause_leq in self.op_encoder.clauses_vec_leq(self.num_nodes-1):
                for (u, v) in Graph.all_possible_edges(self.num_nodes):
                    clause_a = clause_leq.copy()
                    clause_a.add_literal(Literal(self.get_edge(u, v)))
                    clause_a.add_literal(Literal(other.get_edge(u, v)).Not())
                    res.add_clause(clause_a)

                    clause_b = clause_leq.copy()
                    clause_b.add_literal(Literal(self.get_edge(u, v)).Not())
                    clause_b.add_literal(Literal(other.get_edge(u, v)))
                    res.add_clause(clause_b)
        else:
            # forall k > ops, add (y == k) --> Id(x,x') <==> ( y != k  v  Id(x,x') )
            for k in range(self.num_nodes, 2**self.op_encoder.num_nodevars):
                for (u, v) in Graph.all_possible_edges(self.num_nodes):
                    clause_a = self.op_encoder.clause_vec_neq(k)
                    clause_a.add_literal(Literal(self.get_edge(u, v)))
                    clause_a.add_literal(Literal(other.get_edge(u,v)).Not())
                    res.add_clause(clause_a)

                    clause_b = self.op_encoder.clause_vec_neq(k)
                    clause_b.add_literal(Literal(self.get_edge(u, v)).Not())
                    clause_b.add_literal(Literal(other.get_edge(u,v)))
                    res.add_clause(clause_b)

        # 2b. Handle EF (unused edge IDs)
        if self.op_encoder.allow_ef:
            if GraphEncoding.use_leq:
                # add (y > allowed_ef_edges-1 ^ z = EF) --> Id(x,x')
                for clause_leq in self.op_encoder.clauses_vec_leq(y_val=len(self.op_encoder.allowed_ef_edges)-1, z_val=GSOps.EF):
                    for (u, v) in Graph.all_possible_edges(self.num_nodes):
                        clause_a = clause_leq.copy()
                        clause_a.add_literal(Literal(self.get_edge(u, v)))
                        clause_a.add_literal(Literal(other.get_edge(u, v)).Not())
                        res.add_clause(clause_a)

                        clause_b = clause_leq.copy()
                        clause_b.add_literal(Literal(self.get_edge(u, v)).Not())
                        clause_b.add_literal(Literal(other.get_edge(u, v)))
                        res.add_clause(clause_b)
            else:
                # forall k > allowed_ef_edges, add (y == k ^ z = EF) --> Id(x,x')
                for k in range(len(self.op_encoder.allowed_ef_edges), 2**self.op_encoder.num_nodevars):
                    for (u, v) in Graph.all_possible_edges(self.num_nodes):
                        clause_a = self.op_encoder.clause_vec_neq(y_val=k, z_val=GSOps.EF)
                        clause_a.add_literal(Literal(self.get_edge(u, v)))
                        clause_a.add_literal(Literal(other.get_edge(u,v)).Not())
                        res.add_clause(clause_a)

                        clause_b = self.op_encoder.clause_vec_neq(y_val=k, z_val=GSOps.EF)
                        clause_b.add_literal(Literal(self.get_edge(u, v)).Not())
                        clause_b.add_literal(Literal(other.get_edge(u,v)))
                        res.add_clause(clause_b)

        # 2c. Handle unused OPIds
        # forall unused opids, add (z == opid) --> Id(x,x')
        for opid in self.op_encoder.unused_opids:
            for (u, v) in Graph.all_possible_edges(self.num_nodes):
                clause_a = self.op_encoder.clause_vec_neq(z_val=opid)
                clause_a.add_literal(Literal(self.get_edge(u, v)))
                clause_a.add_literal(Literal(other.get_edge(u,v)).Not())
                res.add_clause(clause_a)

                clause_b = self.op_encoder.clause_vec_neq(z_val=opid)
                clause_b.add_literal(Literal(self.get_edge(u, v)).Not())
                clause_b.add_literal(Literal(other.get_edge(u,v)))
                res.add_clause(clause_b)


        if dummy_clauses:
            vars_a_prime = other.op_encoder.opvars
            for a in vars_a_prime:
                clause = Clause()
                clause.add_literal(Literal(a))
                clause.add_literal(Literal(a).Not())
                res.add_clause(clause)

        return res




class GraphStateBMC:
    """
    Translate a source_graph + target_graph + transition relation (given by
    get_update_formula_with()) to a BMC SAT query.
    """

    def __init__(self, source_graph, target_graph, number_of_steps=1, allowed_ef_edges=None):

        if number_of_steps < 1:
            raise ValueError

        if source_graph.num_nodes != target_graph.num_nodes:
            raise ValueError

        self.source_graph = source_graph
        self.target_graph = target_graph
        self.number_of_steps = number_of_steps
        self.sequence = [GraphEncoding(tstep=k,
                                       graph=source_graph,
                                       allowed_ef_edges=allowed_ef_edges) for k in range(number_of_steps)] + \
                        [GraphEncoding(tstep=self.number_of_steps,
                                       graph=target_graph,
                                       allowed_ef_edges=allowed_ef_edges)]
        self.source = self.sequence[0]
        self.target = self.sequence[-1]


    def generate_bmc_cnf(self, dummy_clauses=False):
        """
        Actually generate the BMC CNF formula.
        """
        res = CNF()

        # Source state (sequence[0])
        res.merge(self.source.cnf)

        for t in range(self.number_of_steps):
            # Transition from state at time t to time t+1
            gv_first = self.sequence[t]
            gv_second = self.sequence[t + 1]
            res.merge(gv_first.get_update_formula_with(other=gv_second, dummy_clauses=dummy_clauses))

        # Target state (sequence[k])
        res.merge(self.target.cnf)

        return res


    def retrieve_operations(self, model, k, nqubits):
        """
        Interpret a given satisfying assignment as a list of operations (LC+VD).
        """
        y_vals = [{} for _ in range(k)]
        z_vals = [{} for _ in range(k)]
        op_seq = ['' for _ in range(k)]
        op_decoder = self.sequence[0].op_encoder

        for var in model:
            var_info = str(var).split('_') # e.g. ['y', 'opid_bit', 't']
            if var_info[0] == 'y':
                t = int(var_info[2])
                opid_bit = int(var_info[1])
                y_vals[t][opid_bit] = 1 if z3.is_true(model[var]) else 0
            elif var_info[0] == 'z':
                t = int(var_info[2])
                opid_bit = int(var_info[1])
                z_vals[t][opid_bit] = 1 if z3.is_true(model[var]) else 0

        for i in range(k):
            int_y = bits_to_int(y_vals[i])
            int_z = bits_to_int(z_vals[i])
            op_seq[i] = op_decoder.get_name(int_y, int_z)
        return op_seq


    def dimacs_source(self):
        """
        Return a DIMACS format string encoding the source state CNF.
        """
        return self.source.cnf.dimacs(variable_indices=self.bdd_state_var_indices())


    def dimacs_target(self):
        """
        Return a DIMACS format string encoding the target state CNF.
        """
        return self.target.cnf.dimacs(variable_indices=self.bdd_state_var_indices())


    def dimacs_transition_relation(self, dummy_clauses):
        """
        Return a DIMACS format string transition relation for a single step.
        """
        cnf = self.source.get_update_formula_with(other=self.target, dummy_clauses=dummy_clauses)
        return cnf.dimacs(variable_indices=self.bdd_rel_var_indices())


    def bdd_state_var_indices(self):
        """
        The state vars (both source/target) are odd variables starting at 1.
        """
        source_vars = self.source.get_adjacency_lists_flattened() + self.source.op_encoder.opvars
        target_vars = self.target.get_adjacency_lists_flattened() + self.target.op_encoder.opvars
        assert len(source_vars) == len(target_vars)

        ret = {}
        for i, (var_source, var_target) in enumerate(zip(source_vars, target_vars)):
            ret[var_source] = 2 * i + 1
            ret[var_target] = 2 * i + 1

        return ret


    def bdd_rel_var_indices(self):
        """
        The rel vars are odd for source (unprimed) and even for target (primed).
        """
        source_vars = self.source.get_adjacency_lists_flattened() + self.source.op_encoder.opvars
        target_vars = self.target.get_adjacency_lists_flattened() + self.target.op_encoder.opvars
        assert len(source_vars) == len(target_vars)

        ret = {}
        for i, (var_source, var_target) in enumerate(zip(source_vars, target_vars)):
            ret[var_source] = 2 * i + 1
            ret[var_target] = 2 * i + 2

        return ret


def main():
    """
    Actual main function, such that the linter stops complaining about global
    scope variables.
    """

    # run w/o EF
    print("Test without edge creation:")
    steps = 1
    nqubits = 3
    source = GraphFactory.get_complete_graph(nqubits)
    target = GraphFactory.get_complete_graph(nqubits)
    target.set_edge(0, 1, False)
    target.set_edge(1, 2, False)
    print("Source graph: ", source)
    print("Target graph: ", target)

    s = z3.Solver()
    gvs = GraphStateBMC(source, target, steps)
    bmc_cnf = gvs.generate_bmc_cnf(dummy_clauses=False)
    for clause in bmc_cnf.clauses:
        s.add(clause.to_formula())


    check = s.check()
    print(check)
    if check == z3.sat:
        ops = gvs.retrieve_operations(s.model(), steps, nqubits)
        print("Operations: ", ops)


    # run w/ EF
    print("\nTest with edge creation:")
    steps = 1
    nqubits = 3
    source = GraphFactory.get_empty_graph(nqubits)
    target = GraphFactory.get_empty_graph(nqubits)
    target.set_edge(1, 2, True)
    allowed_ef = [(1, 2)]
    print("Source graph: ", source)
    print("Target graph: ", target)

    s = z3.Solver()
    gs_bmc = GraphStateBMC(source, target, steps, allowed_ef)
    bmc_cnf = gs_bmc.generate_bmc_cnf()
    for clause in bmc_cnf.clauses:
        s.add(clause.to_formula())

    check = s.check()
    print(check)
    if check == z3.sat:
        ops = gs_bmc.retrieve_operations(s.model(), steps, nqubits)
        print("Operations: ", ops)


if __name__ == "__main__":
    main()
