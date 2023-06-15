"""
Custom implementation of CNF class (maybe replace with z3/pysat CNF)
"""
import z3

class Variable:

    def __init__(self, variable_name):
        self.variable_name = variable_name
        self.z3variable = z3.Bool(self.variable_name)

    def to_formula(self):
        return self.z3variable

    def __repr__(self):
        return f"{self.variable_name}"


class Literal:

    def __init__(self, variable, negated=False):
        # TODO: replace negated=False with, is_positive=True
        assert isinstance(variable, Variable)
        self.variable = variable
        self.negated = negated

    def to_formula(self):
        if self.negated:
            return z3.Not(self.variable.to_formula())
        else:
            return self.variable.to_formula()

    def to_pysat_literal(self, mapping):
        """
        Return the current literal in pysat format.
        """
        if self.negated:
            return -mapping[self.variable]
        else:
            return mapping[self.variable]

    def toggle(self):
        self.negated = not self.negated
        return self

    def Not(self):
        return self.toggle()

    def dimacs(self, variable_index):
        if self.negated:
            return "-{}".format(variable_index)
        else:
            return "{}".format(variable_index)

    def __repr__(self):
        if self.negated:
            return "-{}".format(self.variable.variable_name)
        else:
            return "{}".format(self.variable.variable_name)


class Clause:

    def __init__(self, literals=None):
        if literals is None:
            self.literals = []
        else:
            self.literals = literals

    def copy(self):
        """
        Returns a (shallow) copy of self.
        """
        copy = Clause()
        for lit in self.literals:
            copy.add_literal(lit)
        return copy

    def add_literal(self, literal):
        assert isinstance(literal, Literal)
        self.literals.append(literal)

    def to_formula(self):
        return z3.Or([literal.to_formula() for literal in self.literals])

    def to_pysat_clause(self, mapping):
        """
        Return the current clause in pysat format.
        """
        return [literal.to_pysat_literal(mapping) for literal in self.literals]

    def merge(self, other):
        for literal in other.literals:
            self.add_literal(literal)

    def variables(self):
        """get set of variables occuring in this clause"""
        ret = set()
        for literal in self.literals:
            ret.add(literal.variable)
        return ret

    def dimacs(self, variable_indices):
        """
        Parameters
        ----------
        variable_order: dictionary Variable -> int

        """
        #assert(set(variable_indices.keys()) == self.variables())

        ret = ""
        for literal in self.literals:
            variable_index = variable_indices[literal.variable]
            ret += "{} ".format(literal.dimacs(variable_index=variable_index))
        return ret

    @property
    def num_literals(self):
        return len(self.literals)

    def __repr__(self):
        ret = ""
        for ix in range(self.num_literals - 1):
            ret += "{} v ".format(self.literals[ix])
        ret += str(self.literals[-1])
        return ret


class CNF:

    def __init__(self, clauses=None):
        if clauses is None:
            self.clauses = []
        else:
            self.clauses = clauses

    def add_clause(self, clause):
        assert(isinstance(clause, Clause))
        self.clauses.append(clause)

    def to_formula(self):
        """
        Returns the current CNF as a z3 formula.
        """
        # TODO: rename to 'to_z3_formula'
        return z3.And([c.to_formula() for c in self.clauses])

    def to_pysat_clauses(self):
        """
        Returns the current CNF in the format pysat uses (w/ positive and
        negative ints for positive and negative literals).
        """

        # map vars to unique integers
        mapping = {}
        i = 1
        for var in self.variables():
            mapping[var] = i # (note that the keys are variables, not var names)
            i += 1

        # apply map to all clauses
        res = []
        for clause in self.clauses:
            res.append(clause.to_pysat_clause(mapping))

        return res

    def merge(self, other):
        for c in other.clauses:
            self.add_clause(c)

    def variables(self):
        """get set of variables occuring in this formula"""
        ret = set()
        for clause in self.clauses:
            for variable in clause.variables():
                ret.add(variable)
        return ret

    @property
    def number_of_variables(self):
        return len(list(self.variables()))

    @property
    def number_of_clauses(self):
        return len(self.clauses)

    def get_variable_map(self):
        """
        Get a map from Variable (with string name) to ints
        """
        res = {}
        for i, v in enumerate(self.variables()):
            res[v] = i + 1 # (start numbering from 1)
        return res

    def dimacs(self, variable_indices):
        """
        Parameters
        ----------
        variable_indices: dictionary Variable -> int

        """
        #assert(set(variable_indices.keys()) == self.variables())

        ret = f"p cnf {self.number_of_variables} {self.number_of_clauses}\n"

        for clause in self.clauses:
            clause_variable_indices = {}
            for variable in clause.variables():
                clause_variable_indices[variable] = variable_indices[variable]
            ret += clause.dimacs(variable_indices=clause_variable_indices) + "0\n"

        # adding comments, i.e. lines of the form 'c [variable-index] [variable-name]',
        # sorted by index of the variable
        comments = {}
        for variable in self.variables():
            comments[variable_indices[variable]] = variable.variable_name
        comments = {k: v for k, v in sorted(comments.items(), key=lambda item: item[0])}

        for index, name in comments.items():
            ret += f"c {index} {name}\n"

        return ret

    def __repr__(self):
        ret = ""
        for i in range(self.number_of_clauses - 1):
            ret += f"({self.clauses[i]}) ^ "
        ret += f"({self.clauses[-1]})"
        return ret
