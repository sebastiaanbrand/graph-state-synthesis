use std::cmp::max;
use std::fmt;


#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Clause {
    literals: Vec<i32>
}

impl Clause {

    pub fn new() -> Self {
        Self { literals : Vec::new() }
    }

    pub fn from_literal(lit: i32) -> Self {
        Self { literals: Vec::from([lit]) }
    }

    pub fn add_literal(&mut self, lit: i32) {
        self.literals.push(lit);
    }

    pub fn add_literals(&mut self, literals: Vec<i32>) {
        for lit in literals.iter() {
            self.literals.push(*lit);
        }
    }

    /// add literals from other to self
    /// (computes the OR of two clauses)
    pub fn add_from_clause(&mut self, other: Clause) {
        for lit in other.literals.iter() {
            self.literals.push(*lit);
        }
    }

}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut as_string = String::new();
        as_string += "(";
        for lit in self.literals.iter() {
            as_string += &lit.to_string();
            as_string += ",";
        }
        as_string += ")";
        write!(f, "{}", as_string)
    }
}



pub struct CNF {
    clauses: Vec<Clause>
}

impl CNF {

    pub fn new() -> Self {
        Self { clauses: Vec::new() }
    }

    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    /// add clauses from other to self
    /// (computes the AND of two CNF formulas)
    pub fn add_clauses(&mut self, other: Self) {
        for clause in other.clauses.iter() {
            self.add_clause(clause.clone());
        }
    }

    pub fn nvars(&self) -> usize {
        let mut max_var = 0;
        for clause in self.clauses.iter() {
            for lit in clause.literals.iter() {
                max_var = max(max_var, i32::abs(*lit));
            }
        }
        max_var as usize
    }

    pub fn nclauses(&self) -> usize {
        self.clauses.len()
    }

    /// Return a DIMACS format string of CNF formula.
    pub fn to_dimacs(&self) -> String {
        let mut dimacs = format!("p cnf {} {}\n", self.nvars(), self.nclauses());
        for clause in self.clauses.iter() {
            for lit in clause.literals.iter() {
                dimacs += &format!("{} ", lit);
            }
            dimacs += "0\n"
        }
        dimacs
    }

}

impl fmt::Display for CNF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut as_string = String::new();
        as_string += "(";
        for clause in self.clauses.iter() {
            as_string += &clause.to_string()
        }
        as_string += ")";
        write!(f, "{}", as_string)
    }
}


/// Returns clauses that encode vars == value.
pub fn encode_eq(vars: &Vec<u32>, value: u32) -> CNF {
    //let mut clauses: Vec<Clause> = Vec::with_capacity(vars.len());
    let mut clauses = CNF::new();
    let binary_string = format!("{value:0width$b}", width = vars.len());
    assert!(binary_string.len() == vars.len());
    for (i, b_i) in binary_string.chars().enumerate() {
        let mut clause = Clause::new();
        if b_i == '0' {
            clause.add_literal(-(vars[i] as i32));
        }
        else {
            clause.add_literal(vars[i] as i32);
        }
        clauses.add_clause(clause);
    }
    clauses
}

/// Returns a clause which encodes vars != value.
/// (see Eq. (12) in https://arxiv.org/pdf/2309.03593)
pub fn encode_neq(vars: &Vec<u32>, value: u32) -> Clause {
    let mut clause = Clause::new();
    let binary_string = format!("{value:0width$b}", width = vars.len());
    assert!(binary_string.len() == vars.len());
    for (i, b_i) in binary_string.chars().enumerate() {
        if b_i == '0' {
            clause.add_literal(vars[i] as i32);
        }
        else {
            clause.add_literal(-(vars[i] as i32));
        }
    }
    clause
}

/// Returns clauses that encode vars <= value.
/// See Eq. (13) in https://arxiv.org/pdf/2309.03593.
pub fn encode_leq(vars: &Vec<u32>, value: u32) -> CNF {
    // NOTE: Eq. (13) assumes b_i is the i-th bit from the right (such that the
    // least significant bit (LSB) is b_0, and that value(b) = SUM_i (b_i * 2^i).
    // However, here, binary_string is indexed left to right, such that b_0 is
    // the most significant bit (MSB).
    let mut clauses = CNF::new();
    let binary_string = format!("{value:0width$b}", width = vars.len());
    assert!(binary_string.len() == vars.len());
    for (i, b_i) in binary_string.chars().enumerate() {
        if b_i == '0' {
            let mut clause = Clause::new();
            clause.add_literal(-(vars[i] as i32));
            // indices are reversed so j > i becomes j < i
            for (j, b_j) in binary_string.chars().enumerate() {
                if j == i { break; } // j < i
                if b_j == '1' {
                    clause.add_literal(-(vars[j] as i32));
                }
            }
            clauses.add_clause(clause);
        }
    }
    clauses
}
