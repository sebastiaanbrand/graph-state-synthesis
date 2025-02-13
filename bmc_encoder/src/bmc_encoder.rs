use crate::graph::Graph;
use crate::cnf::{CNF,Clause,encode_eq,encode_leq,encode_neq};
use std::cmp::{min, max};

pub enum GSOps {
    LC,
    VD,
    EF,
    Id,
    NumOps // sorry
}


/// BMC encoder based on the adjacency matrix, using one var for each pair of nodes.
pub struct BMCEncoder { 
    x_vars: Vec<Vec<Vec<u32>>>, // vars[t][u][v] -> edge (u,v) at time t
    y_vars: Vec<Vec<u32>>,      // vars[t] -> \vec y at time t
    w_vars: Vec<Vec<u32>>,      // vars[t][i] -> EF allowed_efs[i] at time t
    z_vars: Vec<Vec<u32>>,      // vars[t] -> \vec z at time t
    nvars: u32,                 // number of variables used
    n: u32,                     // number of qubits / nodes
    depth: u32,                 // BMC search depth
    allowed_efs: Vec<(u32,u32)> // list of pairs of nodes between which EFs are allowed
}


impl BMCEncoder {


    pub fn new(source: &Graph, target: &Graph, depth: u32, allowed_efs: &Vec<(u32,u32)>) -> Self {
        assert!(source.nodes() == target.nodes());

        // compute values for required number of vars
        let m = depth as usize;
        let n = source.nodes() as usize;
        let n_opvars = ((GSOps::NumOps as u32) as f64).log2().ceil() as usize;
        let mut n_efvars = 0;
        if allowed_efs.len() > 0 {
            n_efvars = (allowed_efs.len() as f64).log2().ceil() as usize;
        }

        // init vars
        let mut x_vars = vec![vec![vec![0;n]; n]; m+1];
        let mut y_vars = vec![vec![0;n]; m];
        let mut w_vars = vec![vec![0;n_efvars]; m];
        let mut z_vars = vec![vec![0;n_opvars]; m]; 
        let mut var = 1;
        for t in 0..m {
            for u in 0..n {
                for v in u+1..n {
                    x_vars[t][u][v] = var;
                    var += 1;
                }
            }
            for i in 0..z_vars[0].len() {
                z_vars[t][i] = var;
                var += 1;
            }
            for i in 0..y_vars[0].len() {
                y_vars[t][i] = var;
                var += 1;
            }
            for i in 0..w_vars[0].len() {
                w_vars[t][i] = var;
                var += 1;
            }
        }
        // final "fence post"
        for u in 0..n {
            for v in u+1..n {
                x_vars[m][u][v] = var;
                var += 1;
            }
        }

        Self { x_vars, y_vars, w_vars, z_vars, nvars: var-1, 
               n: n as u32, depth, allowed_efs: allowed_efs.clone() }
    }


    pub fn encode_bmc(&self, source: &Graph, target: &Graph, depth: u32) -> CNF {
        let mut cnf = CNF::new();
        cnf.add_clauses(self.encode_graph(source, 0));
        for t in 0..depth {
            cnf.add_clauses(self.encode_transition(t));
        }
        cnf.add_clauses(self.encode_graph(target, depth));
        cnf
    }


    fn encode_graph(&self, g: &Graph, t: u32) -> CNF {
        let mut res = CNF::new();
        for u in 0..g.nodes() {
            for v in u+1..g.nodes() {
                let lit = self.edge_var(t, u, v);
                if g.contains(u, v) {
                    res.add_clause(Clause::from_literal(lit))
                }
                else {
                    res.add_clause(Clause::from_literal(-lit))
                }
            }
        }
        res
    }


    fn encode_transition(&self, t: u32) -> CNF {
        
        let mut res = CNF::new();

        // forall k : (y = k  ^ z = GSOps::LC) --> LC(k)
        // (see Eq. (4, 14) in https://arxiv.org/pdf/2309.03593)
        for k in 0..self.n {
            for u in 0..self.n {
                for v in u+1..self.n {
                    if u != k && v != k {
                        let mut clause_a = Clause::new();
                        clause_a.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_a.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_a.add_literal(-self.edge_var(t, u, k));
                        clause_a.add_literal(-self.edge_var(t, v, k));
                        clause_a.add_literal(self.edge_var(t+1, u, v));
                        clause_a.add_literal(self.edge_var(t, u, v));
                        res.add_clause(clause_a);

                        let mut clause_b = Clause::new();
                        clause_b.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_b.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_b.add_literal(-self.edge_var(t, u, k));
                        clause_b.add_literal(-self.edge_var(t, v, k));
                        clause_b.add_literal(-self.edge_var(t+1, u, v));
                        clause_b.add_literal(-self.edge_var(t, u, v));
                        res.add_clause(clause_b);

                        let mut clause_c = Clause::new();
                        clause_c.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_c.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_c.add_literal(self.edge_var(t, u, k));
                        clause_c.add_literal(self.edge_var(t+1, u, v));
                        clause_c.add_literal(-self.edge_var(t, u, v));
                        res.add_clause(clause_c);

                        let mut clause_d = Clause::new();
                        clause_d.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_d.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_d.add_literal(self.edge_var(t, v, k));
                        clause_d.add_literal(self.edge_var(t+1, u, v));
                        clause_d.add_literal(-self.edge_var(t, u, v));
                        res.add_clause(clause_d);

                        let mut clause_e = Clause::new();
                        clause_e.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_e.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_e.add_literal(self.edge_var(t, u, k));
                        clause_e.add_literal(-self.edge_var(t+1, u,v));
                        clause_e.add_literal(self.edge_var(t, u, v));
                        res.add_clause(clause_e);

                        let mut clause_f = Clause::new();
                        clause_f.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_f.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_f.add_literal(self.edge_var(t, v, k));
                        clause_f.add_literal(-self.edge_var(t+1, u, v));
                        clause_f.add_literal(self.edge_var(t, u, v));
                        res.add_clause(clause_f);
                    }
                    else {
                        let mut clause_a = Clause::new();
                        clause_a.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_a.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_a.add_literal(self.edge_var(t+1, u, v));
                        clause_a.add_literal(-self.edge_var(t, u, v));
                        res.add_clause(clause_a);

                        let mut clause_b = Clause::new();
                        clause_b.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_b.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::LC as u32));
                        clause_b.add_literal(-self.edge_var(t+1, u, v));
                        clause_b.add_literal(self.edge_var(t, u, v));
                        res.add_clause(clause_b);
                    }
                }
            }
        }

        // forall k : (y = k  ^ z = GSOps::VD) --> VD(k)
        // (see Eqs. (5, 15) in https://arxiv.org/pdf/2309.03593)
        for k in 0..self.n {
            for u in 0..self.n {
                for v in u+1..self.n {
                    if u == k || v == k {
                        let mut clause_a = Clause::new();
                        clause_a.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_a.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::VD as u32));
                        clause_a.add_literal(-self.edge_var(t+1, u, v));
                        res.add_clause(clause_a);
                    }
                    else {
                        let mut clause_a = Clause::new();
                        clause_a.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_a.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::VD as u32));
                        clause_a.add_literal(self.edge_var(t+1, u, v));
                        clause_a.add_literal(-self.edge_var(t, u, v));
                        res.add_clause(clause_a);

                        let mut clause_b = Clause::new();
                        clause_b.add_from_clause(encode_neq(&self.y_vars[t as usize], k));
                        clause_b.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::VD as u32));
                        clause_b.add_literal(-self.edge_var(t+1, u, v));
                        clause_b.add_literal(self.edge_var(t, u, v));
                        res.add_clause(clause_b);
                    }
                }
            }
        }

        // forall (u,v)_k \in allowed_EF : (w? = k  ^ z = GSOps::EF) --> EF_i(u,v)
        // (see Eqs. (6, 16) in https://arxiv.org/pdf/2309.03593)
        for (i, (u_i, v_i)) in self.allowed_efs.iter().enumerate() {
            let mut clause_a = Clause::new();
            clause_a.add_from_clause(encode_neq(&self.w_vars[t as usize], i as u32));
            clause_a.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::EF as u32));
            clause_a.add_literal(self.edge_var(t, *u_i, *v_i));
            clause_a.add_literal(self.edge_var(t+1, *u_i, *v_i));
            res.add_clause(clause_a);

            let mut clause_b = Clause::new();
            clause_b.add_from_clause(encode_neq(&self.w_vars[t as usize], i as u32));
            clause_b.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::EF as u32));
            clause_b.add_literal(-self.edge_var(t, *u_i, *v_i));
            clause_b.add_literal(-self.edge_var(t+1, *u_i, *v_i));
            res.add_clause(clause_b);

            for u in 0..self.n {
                for v in u+1..self.n {
                    // assumes (u_i < v_i) TODO: normalize this at init
                    if u != *u_i || v != *v_i {
                        let mut clause_a = Clause::new();
                        clause_a.add_from_clause(encode_neq(&self.w_vars[t as usize], i as u32));
                        clause_a.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::EF as u32));
                        clause_a.add_literal(self.edge_var(t+1, u, v));
                        clause_a.add_literal(-self.edge_var(t, u, v));
                        res.add_clause(clause_a);
        
                        let mut clause_b = Clause::new();
                        clause_b.add_from_clause(encode_neq(&self.w_vars[t as usize], i as u32));
                        clause_b.add_from_clause(encode_neq(&self.z_vars[t as usize], GSOps::EF as u32));
                        clause_b.add_literal(-self.edge_var(t+1, u, v));
                        clause_b.add_literal(self.edge_var(t, u, v));
                        res.add_clause(clause_b);
                    }
                }
            }
        }

        // (z == GSOps::Id) --> Id
        // (see Eq. (17, 18) in https://arxiv.org/pdf/2309.03593)
        for u in 0..self.n {
            for v in u+1..self.n {
                let mut clause_a = encode_neq(&self.z_vars[t as usize], GSOps::Id as u32);
                clause_a.add_literal(self.edge_var(t+1, u, v));
                clause_a.add_literal(-self.edge_var(t, u, v));
                res.add_clause(clause_a);

                let mut clause_b = encode_neq(&self.z_vars[t as usize], GSOps::Id as u32);
                clause_b.add_literal(-self.edge_var(t+1, u, v));
                clause_b.add_literal(self.edge_var(t, u, v));
                res.add_clause(clause_b);
            }
        }

        // \vec y <= |V| - 1
        res.add_clauses(encode_leq(&self.y_vars[t as usize], self.n - 1));

        // \vec z < |ops| - 1
        res.add_clauses(encode_leq(&self.z_vars[t as usize], GSOps::NumOps as u32 - 1));

        // If no EF, \vec z != GSOps::EF
        if self.allowed_efs.len() == 0 {
            res.add_clause(encode_neq(&self.z_vars[t as usize], GSOps::EF as u32));
        }

        // If EFs, \vec ef <= |allowed_ef| - 1
        if self.allowed_efs.len() > 0 {
            res.add_clauses(encode_leq(&self.w_vars[t as usize], self.allowed_efs.len() as u32 - 1));
        }

        res
    }


    /// Force all VDs at the end, on nodes which are isolated in target but not in source.
    pub fn encode_vds_end(&self, source: &Graph, target: &Graph, depth: u32) -> CNF {
        let vd_nodes: Vec<u32> = (&target.get_isolated_nodes() - &source.get_isolated_nodes()).into_iter().collect();
        let mut vd_idx = 0;
        let mut clauses = CNF::new();
        for t in (0..depth).rev() {
            if vd_idx < vd_nodes.len() {
                // force OP at time t to be VD(isolated node)
                clauses.add_clauses(encode_eq(&self.z_vars[t as usize], GSOps::VD as u32));
                clauses.add_clauses(encode_eq(&self.y_vars[t as usize], vd_nodes[vd_idx]));
                vd_idx += 1;
            }
            else {
                // force OP at time t to not be a VD(any)
                clauses.add_clause(encode_neq(&self.z_vars[t as usize], GSOps::VD as u32));
            }
        }
        clauses
    }


    pub fn decode_model_operations(&self, model: Vec<i32>, depth: u32) -> Vec<String> {
        let mut interp: Vec<String> = Vec::with_capacity(depth as usize);

        let mut assignments = vec!['*';(self.nvars+1) as usize];
        for lit in &model {
            let var = (i32::abs(*lit)) as usize;
            if lit > &0 {
                assignments[var] = '1';
            }
            else {
                assignments[var] = '0';
            }
        }
        // TODO: check for '*' after index 0?

        for t in 0..depth as usize {
            // compute op from z
            let mut op = 0;
            let z_size = self.z_vars[t].len();
            for i in 0..z_size {
                if assignments[self.z_vars[t][i] as usize] == '1' {
                    op += 1 << (z_size - 1 - i);
                }
            }
            // compute node from y
            let mut node = 0;
            let y_size = self.y_vars[t].len();
            for i in 0..y_size {
                if assignments[self.y_vars[t][i] as usize] == '1' {
                    node += 1 << (y_size - 1 - i);
                }
            }
            // compute which EF from w
            let mut ef_idx = 0;
            let w_size = self.w_vars[t].len();
            for i in 0..w_size {
                if assignments[self.w_vars[t][i] as usize] == '1' {
                    ef_idx += 1 << (w_size - 1 - i);
                }
            }
            
            // format result
            if op == (GSOps::LC as i32) {
                interp.push(format!("LC({})", node));
            }
            else if op == (GSOps::VD as i32) {
                interp.push(format!("VD({})", node));
            }
            else if op == (GSOps::EF as i32) {
                let (u, v) = self.allowed_efs[ef_idx];
                interp.push(format!("EF({},{})", u, v));
            }
            else if op == (GSOps::Id as i32) {
                interp.push(format!("Id"));
            }
            else {
                panic!("Cannot decode op for z={}", op);
            }
        }

        interp
    }


    fn edge_var(&self, t: u32, e1: u32, e2: u32) -> i32 {
        let u = min(e1, e2);
        let v = max(e1, e2);
        self.x_vars[t as usize][u as usize][v as usize] as i32
    }


    pub fn print_vars(&self) {
        for t in 0..self.depth {
            for u in 0..self.n {
                for v in u+1..self.n {
                    println!("x_{t}_{u}_{v} = {}", 
                             self.x_vars[t as usize][u as usize][v as usize])
                }
            }
            for i in 0..self.z_vars[0].len() {
                println!("z_{t}_{i} = {}", self.z_vars[t as usize][i as usize])
            }
            for i in 0..self.y_vars[0].len() {
                println!("y_{t}_{i} = {}", self.y_vars[t as usize][i as usize])
            }
            for i in 0..self.w_vars[0].len() {
                println!("w_{t}_{i} = {}", self.w_vars[t as usize][i as usize])
            }
        }
        let t = self.depth;
        for u in 0..self.n {
            for v in u+1..self.n {
                println!("x_{t}_{u}_{v} = {}", 
                         self.x_vars[t as usize][u as usize][v as usize])
            }
        }
    }

}

