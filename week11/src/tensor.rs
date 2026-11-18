use crate::model::Network;
use omeco::{EinCode, GreedyMethod, NestedEinsum, TreeSA, contraction_complexity, optimize_code};
use rand::Rng;
use std::collections::HashMap;

pub type Tree = NestedEinsum<usize>;
type TropValue = (f64, u128);
type TropIndex = HashMap<u64, Vec<(u64, TropValue)>>;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub labels: Vec<usize>,
    pub values: HashMap<u64, f64>,
}

#[derive(Clone, Debug)]
pub struct CountTensor {
    pub labels: Vec<usize>,
    pub values: HashMap<u64, u128>,
}

fn code_for(network: &Network) -> EinCode<usize> {
    EinCode::new(
        network.factors.iter().map(|f| f.vars.clone()).collect(),
        Vec::new(),
    )
}

pub fn plan(network: &Network, order: &str) -> Result<(Tree, f64, f64), String> {
    plan_with(network, order, false)
}

pub fn plan_fast(network: &Network, order: &str) -> Result<(Tree, f64, f64), String> {
    plan_with(network, order, true)
}

fn plan_with(network: &Network, order: &str, fast: bool) -> Result<(Tree, f64, f64), String> {
    let code = code_for(network);
    let sizes = HashMap::from_iter((0..network.nvars).map(|v| (v, 2usize)));
    let tree = match order {
        "greedy" => optimize_code(&code, &sizes, &GreedyMethod::default()),
        "treesa" if fast => optimize_code(&code, &sizes, &TreeSA::fast()),
        "treesa" => optimize_code(&code, &sizes, &TreeSA::default()),
        _ => return Err(format!("unknown contraction order {order}")),
    }
    .ok_or_else(|| "omeco failed to produce a contraction tree".to_string())?;
    let metric = contraction_complexity(&tree, &sizes, &code.ixs);
    Ok((tree, metric.tc, metric.sc))
}

fn assignment(labels: &[usize], key: u64) -> HashMap<usize, u8> {
    labels
        .iter()
        .enumerate()
        .map(|(pos, &label)| (label, ((key >> (labels.len() - 1 - pos)) & 1) as u8))
        .collect()
}

fn key_for(labels: &[usize], values: &HashMap<usize, u8>) -> u64 {
    labels
        .iter()
        .fold(0u64, |acc, label| (acc << 1) | values[label] as u64)
}

fn project(labels: &[usize], key: u64, shared: &[usize]) -> u64 {
    let mut out = 0;
    for label in shared {
        let pos = labels.iter().position(|x| x == label).unwrap();
        out = (out << 1) | ((key >> (labels.len() - 1 - pos)) & 1);
    }
    out
}

fn shared_labels(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter()
        .filter(|label| b.contains(label))
        .copied()
        .collect()
}

fn log_add_simple(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        let hi = a.max(b);
        hi + ((a - hi).exp() + (b - hi).exp()).ln()
    }
}

fn decode(labels: &[usize], key: u64) -> HashMap<usize, u8> {
    assignment(labels, key)
}

fn compatible(a: &HashMap<usize, u8>, b: &HashMap<usize, u8>) -> bool {
    a.iter().all(|(k, v)| b.get(k).is_none_or(|w| w == v))
}

fn leaf_log_tensor(network: &Network, index: usize, beta: f64) -> Tensor {
    let factor = &network.factors[index];
    let min = factor.values.iter().copied().fold(f64::INFINITY, f64::min);
    let mut values = HashMap::new();
    for (key, &energy) in factor.values.iter().enumerate() {
        values.insert(key as u64, -beta * (energy - min));
    }
    Tensor {
        labels: factor.vars.clone(),
        values,
    }
}

pub fn factor_offsets(network: &Network) -> f64 {
    network
        .factors
        .iter()
        .map(|factor| factor.values.iter().copied().fold(f64::INFINITY, f64::min))
        .sum()
}

fn contract_log_nodes(
    tree: &Tree,
    network: &Network,
    beta: f64,
    tensors: &mut Vec<Tensor>,
) -> usize {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let id = tensors.len();
            tensors.push(leaf_log_tensor(network, *tensor_index, beta));
            id
        }
        NestedEinsum::Node { args, eins } => {
            let a = contract_log_nodes(&args[0], network, beta, tensors);
            let b = contract_log_nodes(&args[1], network, beta, tensors);
            let ta = &tensors[a];
            let tb = &tensors[b];
            let shared = shared_labels(&ta.labels, &tb.labels);
            let mut right_by_shared: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
            for (&key, &value) in &tb.values {
                right_by_shared
                    .entry(project(&tb.labels, key, &shared))
                    .or_default()
                    .push((key, value));
            }
            let mut values: HashMap<u64, f64> = HashMap::new();
            for (&ka, &va) in &ta.values {
                let aa = decode(&ta.labels, ka);
                let Some(matches) = right_by_shared.get(&project(&ta.labels, ka, &shared)) else {
                    continue;
                };
                for &(kb, vb) in matches {
                    let bb = decode(&tb.labels, kb);
                    let mut combined = aa.clone();
                    combined.extend(bb);
                    let out = key_for(&eins.iy, &combined);
                    let value = va + vb;
                    values
                        .entry(out)
                        .and_modify(|old| *old = log_add_simple(*old, value))
                        .or_insert(value);
                }
            }
            let id = tensors.len();
            tensors.push(Tensor {
                labels: eins.iy.clone(),
                values,
            });
            id
        }
    }
}

pub fn count_log(
    network: &Network,
    tree: &Tree,
    beta: f64,
) -> Result<(f64, Vec<Tensor>, usize), String> {
    let mut tensors = Vec::new();
    let root = contract_log_nodes(tree, network, beta, &mut tensors);
    let shifted = tensors[root]
        .values
        .get(&0)
        .copied()
        .unwrap_or(f64::NEG_INFINITY);
    if !shifted.is_finite() {
        return Err("zero partition function".to_string());
    }
    Ok((shifted - beta * factor_offsets(network), tensors, root))
}

fn leaf_count(network: &Network, index: usize) -> CountTensor {
    let factor = &network.factors[index];
    let values = factor
        .values
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != 0.0)
        .map(|(i, _)| (i as u64, 1))
        .collect();
    CountTensor {
        labels: factor.vars.clone(),
        values,
    }
}

fn contract_count_nodes(tree: &Tree, network: &Network, tensors: &mut Vec<CountTensor>) -> usize {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let id = tensors.len();
            tensors.push(leaf_count(network, *tensor_index));
            id
        }
        NestedEinsum::Node { args, eins } => {
            let a = contract_count_nodes(&args[0], network, tensors);
            let b = contract_count_nodes(&args[1], network, tensors);
            let ta = &tensors[a];
            let tb = &tensors[b];
            let shared = shared_labels(&ta.labels, &tb.labels);
            let mut right_by_shared: HashMap<u64, Vec<(u64, u128)>> = HashMap::new();
            for (&key, &value) in &tb.values {
                right_by_shared
                    .entry(project(&tb.labels, key, &shared))
                    .or_default()
                    .push((key, value));
            }
            let mut values: HashMap<u64, u128> = HashMap::new();
            for (&ka, &va) in &ta.values {
                let aa = decode(&ta.labels, ka);
                let Some(matches) = right_by_shared.get(&project(&ta.labels, ka, &shared)) else {
                    continue;
                };
                for &(kb, vb) in matches {
                    let bb = decode(&tb.labels, kb);
                    let mut combined = aa.clone();
                    combined.extend(bb);
                    let out = key_for(&eins.iy, &combined);
                    values
                        .entry(out)
                        .and_modify(|old| *old += va * vb)
                        .or_insert(va * vb);
                }
            }
            let id = tensors.len();
            tensors.push(CountTensor {
                labels: eins.iy.clone(),
                values,
            });
            id
        }
    }
}

pub fn count_sparse(
    network: &Network,
    tree: &Tree,
) -> Result<(u128, Vec<CountTensor>, usize), String> {
    let mut tensors = Vec::new();
    let root = contract_count_nodes(tree, network, &mut tensors);
    Ok((
        tensors[root].values.get(&0).copied().unwrap_or(0),
        tensors,
        root,
    ))
}

#[derive(Clone, Debug)]
pub struct TropTensor {
    pub labels: Vec<usize>,
    pub values: HashMap<u64, (f64, u128)>,
}

fn trop_leaf(network: &Network, index: usize) -> TropTensor {
    let factor = &network.factors[index];
    TropTensor {
        labels: factor.vars.clone(),
        values: factor
            .values
            .iter()
            .enumerate()
            .map(|(i, &e)| (i as u64, (e, 1)))
            .collect(),
    }
}

fn contract_trop_nodes(tree: &Tree, network: &Network, tensors: &mut Vec<TropTensor>) -> usize {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let id = tensors.len();
            tensors.push(trop_leaf(network, *tensor_index));
            id
        }
        NestedEinsum::Node { args, eins } => {
            let a = contract_trop_nodes(&args[0], network, tensors);
            let b = contract_trop_nodes(&args[1], network, tensors);
            let ta = &tensors[a];
            let tb = &tensors[b];
            let shared = shared_labels(&ta.labels, &tb.labels);
            let mut right_by_shared: TropIndex = HashMap::new();
            for (&key, &value) in &tb.values {
                right_by_shared
                    .entry(project(&tb.labels, key, &shared))
                    .or_default()
                    .push((key, value));
            }
            let mut values: HashMap<u64, (f64, u128)> = HashMap::new();
            for (&ka, &(ea, ga)) in &ta.values {
                let aa = decode(&ta.labels, ka);
                let Some(matches) = right_by_shared.get(&project(&ta.labels, ka, &shared)) else {
                    continue;
                };
                for &(kb, (eb, gb)) in matches {
                    let bb = decode(&tb.labels, kb);
                    let mut combined = aa.clone();
                    combined.extend(bb);
                    let out = key_for(&eins.iy, &combined);
                    let pair = (ea + eb, ga * gb);
                    match values.get_mut(&out) {
                        None => {
                            values.insert(out, pair);
                        }
                        Some(old) if pair.0 < old.0 => *old = pair,
                        Some(old) if pair.0 == old.0 => old.1 += pair.1,
                        Some(_) => {}
                    }
                }
            }
            let id = tensors.len();
            tensors.push(TropTensor {
                labels: eins.iy.clone(),
                values,
            });
            id
        }
    }
}

pub fn tropical(network: &Network, tree: &Tree) -> (f64, u128) {
    let mut tensors = Vec::new();
    let root = contract_trop_nodes(tree, network, &mut tensors);
    tensors[root]
        .values
        .get(&0)
        .copied()
        .unwrap_or((f64::INFINITY, 0))
}

fn tree_tensor_count(node: &Tree) -> usize {
    match node {
        NestedEinsum::Leaf { .. } => 1,
        NestedEinsum::Node { args, .. } => {
            1 + tree_tensor_count(&args[0]) + tree_tensor_count(&args[1])
        }
    }
}

fn sample_log_node(
    node: &Tree,
    tensors: &[Tensor],
    root_id: usize,
    fixed: &HashMap<usize, u8>,
    rng: &mut impl Rng,
) -> HashMap<usize, u8> {
    match node {
        NestedEinsum::Leaf { .. } => {
            let tensor = &tensors[root_id];
            if tensor.labels.iter().all(|v| fixed.contains_key(v)) {
                return fixed.clone();
            }
            let candidates: Vec<_> = tensor
                .values
                .iter()
                .filter_map(|(&key, &logw)| {
                    let done = decode(&tensor.labels, key);
                    compatible(&done, fixed).then_some((done, logw))
                })
                .collect();
            weighted_log_choice(candidates, rng)
        }
        NestedEinsum::Node { args, .. } => {
            let right_len = tree_tensor_count(&args[1]);
            let a_id = root_id - right_len - 1;
            let b_id = root_id - 1;
            let left = &tensors[a_id];
            let right = &tensors[b_id];
            let mut candidates = Vec::new();
            for (&ka, &va) in &left.values {
                let aa = decode(&left.labels, ka);
                if !compatible(&aa, fixed) {
                    continue;
                }
                for (&kb, &vb) in &right.values {
                    let bb = decode(&right.labels, kb);
                    if compatible(&aa, &bb) && compatible(&bb, fixed) {
                        let mut combined = aa.clone();
                        combined.extend(bb);
                        candidates.push((combined, va + vb));
                    }
                }
            }
            let chosen = weighted_log_choice(candidates, rng);
            let mut out = sample_log_node(&args[0], tensors, a_id, &chosen, rng);
            out.extend(sample_log_node(&args[1], tensors, b_id, &chosen, rng));
            out
        }
    }
}

fn weighted_log_choice(
    mut candidates: Vec<(HashMap<usize, u8>, f64)>,
    rng: &mut impl Rng,
) -> HashMap<usize, u8> {
    let max = candidates
        .iter()
        .map(|x| x.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = candidates.iter().map(|(_, v)| (v - max).exp()).sum();
    let mut u = rng.random::<f64>() * total;
    for (values, logw) in candidates.drain(..) {
        let w = (logw - max).exp();
        if u <= w {
            return values;
        }
        u -= w;
    }
    HashMap::new()
}

pub fn sample_log(
    network: &Network,
    tree: &Tree,
    tensors: &[Tensor],
    rng: &mut impl Rng,
) -> Vec<u8> {
    let values = sample_log_node(tree, tensors, tensors.len() - 1, &HashMap::new(), rng);
    (0..network.nvars)
        .map(|v| *values.get(&v).unwrap_or(&0))
        .collect()
}

fn sample_count_node(
    node: &Tree,
    tensors: &[CountTensor],
    root_id: usize,
    fixed: &HashMap<usize, u8>,
    rng: &mut impl Rng,
) -> HashMap<usize, u8> {
    match node {
        NestedEinsum::Leaf { .. } => {
            let tensor = &tensors[root_id];
            let candidates: Vec<_> = tensor
                .values
                .iter()
                .filter_map(|(&key, &count)| {
                    let done = decode(&tensor.labels, key);
                    compatible(&done, fixed).then_some((done, count as f64))
                })
                .collect();
            weighted_choice(candidates, rng)
        }
        NestedEinsum::Node { args, .. } => {
            let right_len = tree_tensor_count(&args[1]);
            let a_id = root_id - right_len - 1;
            let b_id = root_id - 1;
            let left = &tensors[a_id];
            let right = &tensors[b_id];
            let mut candidates = Vec::new();
            for (&ka, &ca) in &left.values {
                let aa = decode(&left.labels, ka);
                if !compatible(&aa, fixed) {
                    continue;
                }
                for (&kb, &cb) in &right.values {
                    let bb = decode(&right.labels, kb);
                    if compatible(&aa, &bb) && compatible(&bb, fixed) {
                        let mut combined = aa.clone();
                        combined.extend(bb);
                        candidates.push((combined, (ca as f64) * (cb as f64)));
                    }
                }
            }
            let chosen = weighted_choice(candidates, rng);
            let mut out = sample_count_node(&args[0], tensors, a_id, &chosen, rng);
            out.extend(sample_count_node(&args[1], tensors, b_id, &chosen, rng));
            out
        }
    }
}

fn weighted_choice(
    mut candidates: Vec<(HashMap<usize, u8>, f64)>,
    rng: &mut impl Rng,
) -> HashMap<usize, u8> {
    let total: f64 = candidates.iter().map(|x| x.1).sum();
    let mut u = rng.random::<f64>() * total;
    for (values, weight) in candidates.drain(..) {
        if u <= weight {
            return values;
        }
        u -= weight;
    }
    HashMap::new()
}

pub fn sample_count(
    network: &Network,
    tree: &Tree,
    tensors: &[CountTensor],
    rng: &mut impl Rng,
) -> Vec<u8> {
    let values = sample_count_node(tree, tensors, tensors.len() - 1, &HashMap::new(), rng);
    (0..network.nvars)
        .map(|v| *values.get(&v).unwrap_or(&0))
        .collect()
}
