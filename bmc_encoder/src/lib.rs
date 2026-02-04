pub mod graph;
pub mod cnf;
pub mod bmc_encoder;

use graph::Graph;
use bmc_encoder::{BMCEncoder,GSOps};
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;


#[pyfunction]
#[pyo3(signature = (source, target, num_nodes, depth, allowed_efs=Vec::new(), force_vds_end=false, form=Vec::new()))]
fn encode_bmc(source: &str, target: &str, num_nodes:u32, depth: u32, allowed_efs: Vec<(u32,u32)>, force_vds_end: bool, form: Vec<u32>) -> PyResult<String> {
    let mut source = Graph::from_tgf(source);
    let mut target = Graph::from_tgf(target);
    source.extend_nodes_to(num_nodes);
    target.extend_nodes_to(num_nodes);
    let encoder = BMCEncoder::new(&source, &target, depth, &allowed_efs);
    let mut cnf;
    if form.len() == 0 {
        cnf = encoder.encode_bmc(&source, &target, depth);
    }
    else {
        // TODO: assert depth == form.len() or just ignore depth when form is set?
        cnf = encoder.encode_bmc_with_form(&source, &target, form);
    }
    if force_vds_end {
        cnf.add_clauses(encoder.encode_vds_end(&source, &target, depth));
    }
    Ok(cnf.to_dimacs())
}


#[pyfunction]
fn op_id(name: &str) -> PyResult<u32> {
    match name.to_lowercase().as_str() {
        "lc"  => Ok(GSOps::LC as u32),
        "vd"  => Ok(GSOps::VD as u32),
        "ef"  => Ok(GSOps::EF as u32),
        "cz"  => Ok(GSOps::EF as u32),
        "*"   => Ok(GSOps::AnyOp as u32),
        "any" => Ok(GSOps::AnyOp as u32),
        _     => Err(PyErr::new::<PyTypeError, _>("Invalid op ID"))
    }
}


#[pyfunction]
#[pyo3(signature = (model, num_nodes, depth, allowed_efs=Vec::new()))]
fn decode_model(model: Vec<i32>, num_nodes:u32, depth: u32, allowed_efs: Vec<(u32,u32)>) -> PyResult<Vec<String>> {
    let source = Graph::new(num_nodes);
    let target = Graph::new(num_nodes);
    let encoder = BMCEncoder::new(&source, &target, depth, &allowed_efs);
    Ok(encoder.decode_model_operations(model, depth))
}


/// A Python module implemented in Rust.
#[pymodule]
fn gs_bmc_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_bmc, m)?)?;
    m.add_function(wrap_pyfunction!(op_id, m)?)?;
    m.add_function(wrap_pyfunction!(decode_model, m)?)?;
    Ok(())
}
