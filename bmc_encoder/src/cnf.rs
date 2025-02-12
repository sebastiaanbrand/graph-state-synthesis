use std::collections::{BTreeSet,HashSet};
use std::cmp::max;
use std::fmt;


#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Clause {
    literals: BTreeSet<i32>
}

impl Clause {

    pub fn new() -> Self {
        Self { literals: BTreeSet::new() }
    }

    pub fn from_literal(lit: i32) -> Self {
        Self { literals: BTreeSet::from([lit]) }
    }

    pub fn add_literal(&mut self, lit: i32) {
        self.literals.insert(lit);
    }

    pub fn add_literals(&mut self, literals: Vec<i32>) {
        for lit in literals.iter() {
            self.literals.insert(*lit);
        }
    }

    /// add literals from other to self
    /// (computes the OR of two clauses)
    pub fn add_from_clause(&mut self, other: Clause) {
        for lit in other.literals.iter() {
            self.literals.insert(*lit);
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
    clauses: HashSet<Clause>
}

impl CNF {

    pub fn new() -> Self {
        Self { clauses: HashSet::new() }
    }

    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.insert(clause);
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

