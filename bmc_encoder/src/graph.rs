// TODO: maybe no graph creation algs here, only read
// from tgf and write to dimacs
use std::collections::HashSet;
use std::cmp::{min, max};
use std::fs::read_to_string;
use std::fmt;


pub struct Graph {
    n: u32,
    edges: HashSet<(u32,u32)>
}


impl Graph {

    /// Creates an empty graph with `n` nodes.
    pub fn new(n: u32) -> Self {
        Self { n , edges: HashSet::new() }
    }

    /// Reads the graph given in a TGF (trivial graph format) string.
    pub fn from_tgf(tgf_string: &str) -> Self {
        let mut g = Self::new(0);
        let mut n = 0;
        for line in tgf_string.lines() {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if !tokens[0].starts_with('#') {
                let u = tokens[0].parse::<u32>().unwrap();
                let v = tokens[1].parse::<u32>().unwrap();
                // TGF starts numbering nodes at 1, we start at 0 internally
                g.add_edge(u-1, v-1);
                n = max(n, max(u, v));
            }
        }
        g.n = n;
        g
    }

    /// Reads the graph given in a TGF (trivial graph format) file.
    pub fn from_tgf_file(filename: &str) -> Self {
        let tgf_string = read_to_string(filename).unwrap();
        let g = Self::from_tgf(&tgf_string);
        g
    }

    /// Returns the number of nodes of the graph.
    pub fn nodes(&self) -> u32 {
        self.n
    }

    /// Get all nodes with at least one edge.
    fn get_non_isolated_nodes(&self) -> HashSet<u32> {
        let mut nodes = HashSet::new();
        for (u, v) in self.edges.iter() {
            nodes.insert(*u);
            nodes.insert(*v);
        }
        nodes
    }

    /// Get all nodes with no connected edges.
    pub fn get_isolated_nodes(&self) -> HashSet<u32> {
        let all_nodes = HashSet::from_iter(0..self.n);
        let isolated = all_nodes.difference(&self.get_non_isolated_nodes()).cloned().collect();
        isolated
    }

    /// Increase the domain to include isolated nodes.
    pub fn extend_nodes_to(&mut self, n: u32) {
        if n >= self.n {
            self.n = n;
        }
        else {
            panic!("Can only increase the number of nodes (add isolated nodes).");
        }
    }

    /// Returns true iff the graph contains the edge `(u,v)`.
    pub fn contains(&self, u: u32, v: u32) -> bool {
        self.edges.contains(&(min(u,v), max(u,v)))
    }

    /// Adds the edge `(u,v)`` to the graph.
    pub fn add_edge(&mut self, u: u32, v: u32) {
        self.edges.insert((min(u,v), max(u,v)));
    }

    /// Removes the edge `(u,v)` from the graph.
    pub fn remove_edge(&mut self, u: u32, v: u32) {
        self.edges.remove(&(min(u,v), max(u,v)));
    }
        
}


impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut edges = String::new();
        for (u,v) in self.edges.iter() {
            edges += format!("({},{}),", u, v).as_str();
        }
        write!(f, "Graph(n={}, edges=[{}])", self.n, edges)
    }
}
