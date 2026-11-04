use crate::uai::{AppliedFactor, AppliedModel};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct Graph {
    dims: Vec<usize>,
    factors: Vec<AppliedFactor>,
    edge_vars: Vec<usize>,
    edge_factors: Vec<usize>,
    touching_var: Vec<Vec<usize>>,
    factor_pos_edge: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct Messages {
    var_to_factor: Vec<Vec<f64>>,
    factor_to_var: Vec<Vec<f64>>,
}

impl Graph {
    fn new(model: &AppliedModel) -> Self {
        let factors = model.factors.clone();
        let mut touching_var = vec![Vec::new(); model.vars];
        let mut factor_pos_edge = vec![Vec::new(); factors.len()];
        let mut edge_vars = Vec::new();
        let mut edge_factors = Vec::new();
        for (f, factor) in factors.iter().enumerate() {
            for &v in &factor.vars {
                let edge = edge_vars.len();
                edge_vars.push(v);
                edge_factors.push(f);
                touching_var[v].push(edge);
                factor_pos_edge[f].push(edge);
            }
        }
        Self {
            dims: model.dims.clone(),
            factors,
            edge_vars,
            edge_factors,
            touching_var,
            factor_pos_edge,
        }
    }

    fn uniform(&self) -> Messages {
        let var_to_factor = self
            .edge_vars
            .iter()
            .map(|&v| vec![1.0 / self.dims[v] as f64; self.dims[v]])
            .collect();
        let factor_to_var = self
            .edge_vars
            .iter()
            .map(|&v| vec![1.0 / self.dims[v] as f64; self.dims[v]])
            .collect();
        Messages {
            var_to_factor,
            factor_to_var,
        }
    }
}

fn normalize(values: &mut [f64]) -> Result<(), String> {
    let total: f64 = values.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return Err(format!(
            "message proposal cannot be normalized; total={total}"
        ));
    }
    for value in values {
        *value /= total;
    }
    Ok(())
}

fn sweep(
    graph: &Graph,
    current: &Messages,
    include_recipient: bool,
) -> Result<(Messages, u128), String> {
    let mut proposed = Messages {
        var_to_factor: vec![Vec::new(); current.var_to_factor.len()],
        factor_to_var: vec![Vec::new(); current.var_to_factor.len()],
    };
    for var in 0..graph.dims.len() {
        for &edge in &graph.touching_var[var] {
            let recipient = graph.edge_factors[edge];
            let mut message = vec![1.0; graph.dims[var]];
            for &source in &graph.touching_var[var] {
                if include_recipient || graph.edge_factors[source] != recipient {
                    for (state, value) in message.iter_mut().enumerate() {
                        *value *= current.factor_to_var[source][state];
                    }
                }
            }
            normalize(&mut message)?;
            proposed.var_to_factor[edge] = message;
        }
    }

    let mut work = 0u128;
    for (f, factor) in graph.factors.iter().enumerate() {
        let entries = factor.values.len();
        for (vpos, &edge) in graph.factor_pos_edge[f].iter().enumerate() {
            let var = graph.edge_vars[edge];
            let mut message = vec![0.0; graph.dims[var]];
            for (index, &weight) in factor.values.iter().enumerate() {
                let mut rest = index;
                let mut product = weight;
                let mut var_state = 0usize;
                for axis in (0..factor.vars.len()).rev() {
                    let state = rest % factor.dims[axis];
                    rest /= factor.dims[axis];
                    if axis == vpos {
                        var_state = state;
                    } else if include_recipient
                        || graph.edge_vars[graph.factor_pos_edge[f][axis]] != var
                    {
                        let source_edge = graph.factor_pos_edge[f][axis];
                        product *= current.var_to_factor[source_edge][state];
                    }
                }
                message[var_state] += product;
            }
            normalize(&mut message)?;
            proposed.factor_to_var[edge] = message;
        }
        work += entries as u128;
    }
    Ok((proposed, work))
}

fn beliefs(graph: &Graph, messages: &Messages) -> Result<BTreeMap<usize, Vec<f64>>, String> {
    let mut out = BTreeMap::new();
    for var in 0..graph.dims.len() {
        if graph.touching_var[var].is_empty() {
            continue;
        }
        let mut belief = vec![1.0; graph.dims[var]];
        for &edge in &graph.touching_var[var] {
            for (state, value) in belief.iter_mut().enumerate() {
                *value *= messages.factor_to_var[edge][state];
            }
        }
        normalize(&mut belief)?;
        out.insert(var, belief);
    }
    Ok(out)
}

fn residual(a: &Messages, b: &Messages) -> f64 {
    a.var_to_factor
        .iter()
        .zip(&b.var_to_factor)
        .chain(a.factor_to_var.iter().zip(&b.factor_to_var))
        .flat_map(|(x, y)| x.iter().zip(y).map(|(a, b)| (a - b).abs()))
        .fold(0.0, f64::max)
}

fn max_belief_error(beliefs: &BTreeMap<usize, Vec<f64>>, exact: &BTreeMap<usize, Vec<f64>>) -> f64 {
    beliefs
        .iter()
        .filter_map(|(v, values)| exact.get(v).map(|e| (values, e)))
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn query_errors(
    beliefs: &BTreeMap<usize, Vec<f64>>,
    exact: &BTreeMap<usize, Vec<f64>>,
) -> BTreeMap<usize, f64> {
    beliefs
        .iter()
        .filter_map(|(var, values)| {
            exact.get(var).map(|expected| {
                (
                    *var,
                    values
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max),
                )
            })
        })
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct SweepRecord {
    pub sweep: usize,
    pub beliefs: BTreeMap<usize, Vec<f64>>,
    pub undamped_residual: f64,
    pub cumulative_update_work: u128,
    pub max_marginal_error: f64,
    pub query_errors: BTreeMap<usize, f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BpResult {
    pub settings: BTreeMap<String, serde_json::Value>,
    pub final_marginals: BTreeMap<usize, Vec<f64>>,
    pub sweeps: Vec<SweepRecord>,
    pub sweeps_run: usize,
    pub returned_undamped_residual: f64,
    pub converged: bool,
    pub cumulative_update_work: u128,
    pub max_marginal_error: f64,
}

pub fn run(
    model: &AppliedModel,
    damping: f64,
    tolerance: f64,
    max_sweeps: usize,
    exact: &BTreeMap<usize, Vec<f64>>,
) -> Result<BpResult, String> {
    if !(0.0 < damping && damping <= 1.0) {
        return Err("damping must lie in (0,1]".into());
    }
    if tolerance <= 0.0 || !tolerance.is_finite() {
        return Err("tolerance must be positive".into());
    }
    if max_sweeps == 0 {
        return Err("max-sweeps must be positive".into());
    }
    let graph = Graph::new(model);
    let mut current = graph.uniform();
    let mut records = Vec::new();
    let mut work = 0u128;
    let mut final_residual = f64::INFINITY;
    let mut final_marginals = BTreeMap::new();
    for sweep_index in 1..=max_sweeps {
        let (proposed, sweep_work) = sweep(&graph, &current, false)?;
        let change = residual(&proposed, &current);
        let proposed_beliefs = beliefs(&graph, &proposed)?;
        let proposed_error = max_belief_error(&proposed_beliefs, exact);
        let proposed_query_errors = query_errors(&proposed_beliefs, exact);
        work += sweep_work;
        records.push(SweepRecord {
            sweep: sweep_index,
            beliefs: proposed_beliefs.clone(),
            undamped_residual: change,
            cumulative_update_work: work,
            max_marginal_error: proposed_error,
            query_errors: proposed_query_errors,
        });
        if change <= tolerance || sweep_index == max_sweeps {
            final_residual = change;
            final_marginals = proposed_beliefs;
            break;
        }
        let mut damped = Messages {
            var_to_factor: vec![Vec::new(); current.var_to_factor.len()],
            factor_to_var: vec![Vec::new(); current.var_to_factor.len()],
        };
        for edge in 0..current.var_to_factor.len() {
            damped.var_to_factor[edge] = current.var_to_factor[edge]
                .iter()
                .zip(&proposed.var_to_factor[edge])
                .map(|(old, new)| (1.0 - damping) * old + damping * new)
                .collect();
            damped.factor_to_var[edge] = current.factor_to_var[edge]
                .iter()
                .zip(&proposed.factor_to_var[edge])
                .map(|(old, new)| (1.0 - damping) * old + damping * new)
                .collect();
        }
        current = damped;
    }
    Ok(BpResult {
        settings: BTreeMap::from([
            ("method".into(), serde_json::json!("bp")),
            ("damping".into(), serde_json::json!(damping)),
            ("tolerance".into(), serde_json::json!(tolerance)),
            ("max-sweeps".into(), serde_json::json!(max_sweeps)),
            (
                "updates".into(),
                serde_json::json!("synchronous, normalized proposals"),
            ),
        ]),
        final_marginals: final_marginals.clone(),
        sweeps_run: records.len(),
        returned_undamped_residual: final_residual,
        converged: final_residual <= tolerance,
        cumulative_update_work: work,
        max_marginal_error: max_belief_error(&final_marginals, exact),
        sweeps: records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> AppliedModel {
        let factors = vec![
            AppliedFactor {
                vars: vec![1],
                dims: vec![2],
                values: vec![0.2, 0.8],
            },
            AppliedFactor {
                vars: vec![1, 0],
                dims: vec![2, 2],
                values: vec![0.9, 0.1, 0.2, 0.8],
            },
        ];
        AppliedModel {
            vars: 2,
            dims: vec![2, 2],
            observations: BTreeMap::new(),
            factors,
            scalar: 1.0,
        }
    }

    #[test]
    fn bp_is_exact_on_a_factor_tree_and_recipient_inclusion_is_wrong() {
        let model = tree();
        let graph = Graph::new(&model);
        let correct = run(&model, 1.0, 1e-14, 100, &BTreeMap::new()).unwrap();
        assert!(
            (correct.final_marginals[&0][0] - 0.34).abs() < 1e-12,
            "X belief: {:?}",
            correct.final_marginals[&0]
        );
        assert!(
            (correct.final_marginals[&1][0] - 0.2).abs() < 1e-12,
            "E belief: {:?}",
            correct.final_marginals[&1]
        );

        let current = graph.uniform();
        let (proposal, _) = sweep(&graph, &current, true).unwrap();
        let wrong = beliefs(&graph, &proposal).unwrap();
        assert!((wrong[&0][0] - 0.34).abs() > 1e-3);
    }
}
