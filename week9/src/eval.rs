use crate::plan::{Plan, Tree};
use crate::uai::{AppliedFactor, AppliedModel};
use serde::Serialize;
use std::collections::BTreeMap;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

#[derive(Clone, Debug, Serialize)]
pub struct TensorResult {
    pub evidence_mass: f64,
    pub log_evidence: f64,
    pub marginals: BTreeMap<usize, Vec<f64>>,
    pub joints: BTreeMap<String, Vec<f64>>,
}

struct NodeTensor {
    labels: Vec<usize>,
    tensor: Tensor,
    log_scale: f64,
}

fn column_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in 1..dims.len() {
        strides[i] = strides[i - 1] * dims[i - 1];
    }
    strides
}

fn row_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Reinterpret the UAI row-major table as column-major without changing its
/// logical scope order.
fn uai_to_col_major(dims: &[usize], values: &[f64]) -> Vec<f64> {
    let row = row_strides(dims);
    let col = column_strides(dims);
    let mut out = vec![0.0; values.len()];
    for (index, &value) in values.iter().enumerate() {
        let mut rest = index;
        let mut target = 0usize;
        for axis in 0..dims.len() {
            let state = rest / row[axis];
            rest %= row[axis];
            target += state * col[axis];
        }
        out[target] = value;
    }
    out
}

fn notation_char(index: usize) -> char {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    ALPHABET[index] as char
}

fn einsum_for(left: &[usize], right: &[usize], output: &[usize]) -> Result<String, String> {
    let mut labels = left
        .iter()
        .chain(right)
        .chain(output)
        .copied()
        .collect::<Vec<_>>();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() > 52 {
        return Err(format!(
            "a local contraction uses {} labels; maximum is 52",
            labels.len()
        ));
    }
    let code = |axis: usize| notation_char(labels.iter().position(|x| *x == axis).unwrap());
    Ok(format!(
        "{},{}->{}",
        left.iter().map(|x| code(*x)).collect::<String>(),
        right.iter().map(|x| code(*x)).collect::<String>(),
        output.iter().map(|x| code(*x)).collect::<String>(),
    ))
}

fn leaf_tensor(factor: &AppliedFactor) -> Result<(Tensor, f64), String> {
    let total: f64 = factor.values.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return Err(format!(
            "factor {:?} has non-positive or non-finite total {total}",
            factor.vars
        ));
    }
    let normalized: Vec<f64> = factor.values.iter().map(|x| x / total).collect();
    let data = uai_to_col_major(&factor.dims, &normalized);
    let tensor =
        Tensor::from_vec_col_major(factor.dims.clone(), data).map_err(|e| e.to_string())?;
    Ok((tensor, total.ln()))
}

fn contract_node(
    tree: &Tree,
    model: &AppliedModel,
    engine: &mut CpuEngine,
) -> Result<NodeTensor, String> {
    match tree {
        Tree::Leaf { tensor } => {
            let factor = &model.factors[*tensor];
            let (tensor, scale) = leaf_tensor(factor)?;
            Ok(NodeTensor {
                labels: factor.vars.clone(),
                tensor,
                log_scale: scale,
            })
        }
        Tree::Node {
            left,
            right,
            retained,
            ..
        } => {
            let lhs = contract_node(left, model, engine)?;
            let rhs = contract_node(right, model, engine)?;
            let equation = einsum_for(&lhs.labels, &rhs.labels, retained)?;
            let mut output = [&lhs.tensor, &rhs.tensor]
                .einsum(&equation, engine)
                .map_err(|e| format!("{equation}: {e}"))?;
            let values = output
                .as_slice::<f64>()
                .map_err(|e| e.to_string())?
                .to_vec();
            let total: f64 = values.iter().sum();
            if !(total > 0.0 && total.is_finite()) {
                return Err(format!("contraction {equation} has invalid total {total}"));
            }
            let normalized: Vec<f64> = values.iter().map(|x| x / total).collect();
            output = Tensor::from_vec_col_major(
                retained.iter().map(|v| model.dims[*v]).collect::<Vec<_>>(),
                normalized,
            )
            .map_err(|e| e.to_string())?;
            Ok(NodeTensor {
                labels: retained.clone(),
                tensor: output,
                log_scale: lhs.log_scale + rhs.log_scale + total.ln(),
            })
        }
    }
}

fn evaluate(plan: &Plan, model: &AppliedModel) -> Result<(Vec<f64>, f64), String> {
    let mut engine = CpuEngine::serial();
    let result = contract_node(&plan.tree, model, &mut engine)?;
    if result.labels != plan.output {
        return Err(format!(
            "root labels {:?} differ from requested output {:?}",
            result.labels, plan.output
        ));
    }
    Ok((
        result
            .tensor
            .as_slice::<f64>()
            .map_err(|e| e.to_string())?
            .to_vec(),
        model.scalar.ln() + result.log_scale,
    ))
}

pub fn query_marginal(
    plan: &Plan,
    model: &AppliedModel,
    vars: &[usize],
) -> Result<Vec<f64>, String> {
    let query_plan = plan.with_output(vars.to_vec())?;
    evaluate(&query_plan, model).map(|x| x.0)
}

pub fn exact(
    plan: &Plan,
    model: &AppliedModel,
    queries: &[usize],
    joint: Option<(usize, usize)>,
) -> Result<TensorResult, String> {
    let scalar_plan = plan.with_output(Vec::new())?;
    let (scalar, log_evidence) = evaluate(&scalar_plan, model)?;
    if scalar.len() != 1 {
        return Err("scalar evidence contraction did not return one entry".into());
    }
    let mut marginals = BTreeMap::new();
    for &var in queries {
        if let Some(&state) = model.observations.get(&var) {
            let mut values = vec![0.0; model.dims[var]];
            values[state] = 1.0;
            marginals.insert(var, values);
        } else {
            marginals.insert(var, query_marginal(plan, model, &[var])?);
        }
    }
    let mut joints = BTreeMap::new();
    if let Some((a, b)) = joint {
        let values = query_marginal(plan, model, &[a, b])?;
        joints.insert(format!("{},{}", model_name(a), model_name(b)), values);
    }
    Ok(TensorResult {
        evidence_mass: log_evidence.exp(),
        log_evidence,
        marginals,
        joints,
    })
}

fn model_name(var: usize) -> String {
    ["A", "T", "S", "L", "B", "E", "X", "D"]
        .get(var)
        .map(|x| x.to_string())
        .unwrap_or_else(|| var.to_string())
}
