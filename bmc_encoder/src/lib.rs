pub mod graph;
pub mod cnf;
pub mod bmc_encoder;

use graph::Graph;
use bmc_encoder::{BMCEncoder,BMCEncoderAM};
use pyo3::prelude::*;


#[pyfunction]
#[pyo3(signature = (source, target, num_nodes, depth, allowed_efs=Vec::new(), force_vds_end=false))]
fn encode_bmc(source: &str, target: &str, num_nodes:u32, depth: u32, allowed_efs: Vec<(u32,u32)>, force_vds_end: bool) -> PyResult<String> {
    let mut source = Graph::from_tgf(source);
    let mut target = Graph::from_tgf(target);
    source.extend_nodes_to(num_nodes);
    target.extend_nodes_to(num_nodes);
    let encoder: BMCEncoderAM = BMCEncoder::new(&source, &target, depth, &allowed_efs);
    let mut cnf = encoder.encode_bmc(&source, &target, depth);
    if force_vds_end {
        cnf.add_clauses(encoder.encode_vds_end(&source, &target, depth));
    }
    Ok(cnf.to_dimacs())
}

#[pyfunction]
#[pyo3(signature = (model, num_nodes, depth, allowed_efs=Vec::new()))]
fn decode_model(model: Vec<i32>, num_nodes:u32, depth: u32, allowed_efs: Vec<(u32,u32)>) -> PyResult<Vec<String>> {
    let source = Graph::new(num_nodes);
    let target = Graph::new(num_nodes);
    let encoder: BMCEncoderAM = BMCEncoder::new(&source, &target, depth, &allowed_efs);
    Ok(encoder.get_model_operations(model, depth))
}

/// A Python module implemented in Rust.
#[pymodule]
fn gs_bmc_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_bmc, m)?)?;
    m.add_function(wrap_pyfunction!(decode_model, m)?)?;
    Ok(())
}
