#![allow(dead_code)]
/**
 * Run BMC search.
*/
mod graph;
mod cnf;
mod bmc_encoder;
use graph::Graph;
use bmc_encoder::{BMCEncoder,BMCEncoderAM};
use std::env;
use std::cmp::max;


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 {

        let mut source = Graph::from_tgf_file(&args[1]);
        let mut target = Graph::from_tgf_file(&args[2]);
        source.extend_nodes_to(max(source.nodes(), target.nodes()));
        target.extend_nodes_to(max(source.nodes(), target.nodes()));
        
        let depth = 3;
        let mut allowed_ef = Vec::new();
        allowed_ef.push((0,1));
        allowed_ef.push((0,2));
        allowed_ef.push((1,2));
        let encoder: BMCEncoderAM = BMCEncoder::new(&source, &target, depth, &allowed_ef);
        println!("{}", source.nodes());
        println!("{}", target.nodes());
        let cnf = encoder.encode_bmc(&source, &target, depth);
        println!("{}", cnf.to_dimacs());
    }
    else {
        println!("Error: expected arguments source.tfg target.tfg");
    }
}
