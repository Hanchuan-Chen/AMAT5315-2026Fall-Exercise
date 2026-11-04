use crate::uai::{AppliedModel, Model};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct ExactResult {
    pub evidence_mass: f64,
    pub marginals: BTreeMap<usize, Vec<f64>>,
    pub joints: BTreeMap<String, Vec<f64>>,
}

impl ExactResult {
    pub fn asia_enumeration(model: &AppliedModel) -> Self {
        assert_eq!(model.vars, 8, "enumeration is the ASIA cross-check");
        let mut evidence = 0.0;
        let mut marginals: BTreeMap<usize, Vec<f64>> = (0..model.vars)
            .map(|v| (v, vec![0.0; model.dims[v]]))
            .collect();
        let mut joint = vec![0.0; model.dims[1] * model.dims[3]];
        for index in 0..(1usize << model.vars) {
            let assignment: Vec<usize> = (0..model.vars)
                .map(|v| (index >> (model.vars - 1 - v)) & 1)
                .collect();
            if model.observations.iter().any(|(&v, &s)| assignment[v] != s) {
                continue;
            }
            let mut probability = model.scalar;
            for factor in &model.factors {
                probability *= model.factor_value(factor, &assignment);
            }
            evidence += probability;
            for (&var, values) in marginals.iter_mut() {
                values[assignment[var]] += probability;
            }
            joint[assignment[1] * model.dims[3] + assignment[3]] += probability;
        }
        for values in marginals.values_mut() {
            for value in values {
                *value /= evidence;
            }
        }
        for value in &mut joint {
            *value /= evidence;
        }
        let mut joints = BTreeMap::new();
        joints.insert("T,L".to_string(), joint);
        Self {
            evidence_mass: evidence,
            marginals,
            joints,
        }
    }

    pub fn table_check(model: &Model) -> serde_json::Value {
        let factor = model
            .factors
            .iter()
            .find(|f| f.vars == [1, 0])
            .expect("ASIA has p(T|A) with scope T,A");
        serde_json::json!({
            "scope": factor.vars,
            "p_T_given_A": factor.values,
            "p(T=0|A=0)": factor.values[0],
            "p(T=0)": 0.01 * 0.05 + 0.99 * 0.01,
        })
    }
}

pub fn max_abs_error(a: &BTreeMap<usize, Vec<f64>>, b: &BTreeMap<usize, Vec<f64>>) -> f64 {
    a.iter()
        .flat_map(|(v, av)| av.iter().zip(&b[v]).map(move |(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}
