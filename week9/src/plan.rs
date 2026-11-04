use omeco::{
    EinCode, GreedyMethod, Initializer, NestedEinsum, ScoreFunction, TreeSA, optimize_code,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Tree {
    Leaf {
        tensor: usize,
    },
    Node {
        left: Box<Tree>,
        right: Box<Tree>,
        inputs: Vec<Vec<usize>>,
        retained: Vec<usize>,
    },
}

impl Tree {
    pub fn leaves(&self) -> Vec<usize> {
        let mut leaves = Vec::new();
        self.collect_leaves(&mut leaves);
        leaves.sort_unstable();
        leaves
    }

    fn collect_leaves(&self, leaves: &mut Vec<usize>) {
        match self {
            Tree::Leaf { tensor } => leaves.push(*tensor),
            Tree::Node { left, right, .. } => {
                left.collect_leaves(leaves);
                right.collect_leaves(leaves);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub schema: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub evidence: String,
    pub input_labels: Vec<Vec<usize>>,
    pub output: Vec<usize>,
    pub tree: Tree,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub search_settings: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Step {
    pub index: usize,
    pub inputs: Vec<Vec<usize>>,
    pub retained: Vec<usize>,
    pub loop_labels: Vec<usize>,
    pub cost: u128,
    pub result_entries: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Complexity {
    pub work: u128,
    pub largest_tensor: u128,
    pub largest_input: u128,
    pub largest_intermediate: u128,
    pub steps: Vec<Step>,
}

fn product(labels: &[usize], dims: &[usize]) -> u128 {
    labels
        .iter()
        .map(|&label| dims[label] as u128)
        .product::<u128>()
        .max(1)
}

fn label_union(parts: &[&[usize]]) -> Vec<usize> {
    let mut set = BTreeSet::new();
    for part in parts {
        set.extend(part.iter().copied());
    }
    set.into_iter().collect()
}

fn tree_labels(tree: &Tree, inputs: &[Vec<usize>]) -> Vec<usize> {
    match tree {
        Tree::Leaf { tensor } => inputs[*tensor].clone(),
        Tree::Node { retained, .. } => retained.clone(),
    }
}

fn retained_labels(
    left: &[usize],
    right: &[usize],
    subtree_leaves: &[usize],
    input_labels: &[Vec<usize>],
    output: &[usize],
) -> Vec<usize> {
    let union = label_union(&[left, right]);
    union
        .into_iter()
        .filter(|label| {
            let inside = subtree_leaves
                .iter()
                .flat_map(|&t| input_labels[t].iter())
                .filter(|x| *x == label)
                .count();
            let all = input_labels
                .iter()
                .flatten()
                .filter(|x| *x == label)
                .count();
            inside < all || output.contains(label)
        })
        .collect()
}

fn make_node(left: Tree, right: Tree, input_labels: &[Vec<usize>], output: &[usize]) -> Tree {
    let li = tree_labels(&left, input_labels);
    let ri = tree_labels(&right, input_labels);
    let subtree = left
        .leaves()
        .into_iter()
        .chain(right.leaves())
        .collect::<Vec<_>>();
    let retained = retained_labels(&li, &ri, &subtree, input_labels, output);
    Tree::Node {
        left: Box::new(left),
        right: Box::new(right),
        inputs: vec![li, ri],
        retained,
    }
}

fn from_omeco(nested: &NestedEinsum<usize>, input_labels: &[Vec<usize>], output: &[usize]) -> Tree {
    match nested {
        NestedEinsum::Leaf { tensor_index } => Tree::Leaf {
            tensor: *tensor_index,
        },
        NestedEinsum::Node { args, .. } => make_node(
            from_omeco(&args[0], input_labels, output),
            from_omeco(&args[1], input_labels, output),
            input_labels,
            output,
        ),
    }
}

fn eincode(input_labels: &[Vec<usize>], output: &[usize]) -> EinCode<usize> {
    EinCode::new(input_labels.to_vec(), output.to_vec())
}

fn size_dict(dims: &[usize]) -> HashMap<usize, usize> {
    dims.iter().copied().enumerate().collect()
}

pub fn greedy(
    input_labels: Vec<Vec<usize>>,
    output: Vec<usize>,
    dims: &[usize],
    evidence: &str,
) -> Result<Plan, String> {
    let code = eincode(&input_labels, &output);
    let sizes = size_dict(dims);
    let nested = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or("omeco greedy could not plan this network")?;
    let tree = from_omeco(&nested, &input_labels, &output);
    Ok(Plan {
        schema: "umat5315-week9-order-v1".into(),
        method: "greedy".into(),
        seed: None,
        evidence: evidence.into(),
        tree,
        input_labels,
        output,
        search_settings: BTreeMap::new(),
    })
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub fn random(
    input_labels: Vec<Vec<usize>>,
    output: Vec<usize>,
    evidence: &str,
    seed: u64,
) -> Result<Plan, String> {
    if input_labels.is_empty() {
        return Err("cannot plan an empty factor network".into());
    }
    let mut rng = SplitMix64(seed);
    let mut work: Vec<Tree> = (0..input_labels.len())
        .map(|tensor| Tree::Leaf { tensor })
        .collect();
    while work.len() > 1 {
        let i = rng.index(work.len());
        let mut j = rng.index(work.len() - 1);
        if j >= i {
            j += 1;
        }
        let right = work.remove(j.max(i));
        let left = work.remove(j.min(i));
        work.push(make_node(left, right, &input_labels, &output));
    }
    let tree = work.pop().unwrap();
    Ok(Plan {
        schema: "amat5315-week9-order-v1".into(),
        method: "random".into(),
        seed: Some(seed),
        evidence: evidence.into(),
        input_labels,
        output,
        tree,
        search_settings: BTreeMap::new(),
    })
}

pub fn treesa(
    input_labels: Vec<Vec<usize>>,
    output: Vec<usize>,
    dims: &[usize],
    evidence: &str,
    seed: u64,
) -> Result<Plan, String> {
    let code = eincode(&input_labels, &output);
    let sizes = size_dict(dims);
    let mut rng = SplitMix64(seed);
    let restarts = 3usize;
    let trials = 6usize;
    let iterations = 40usize;
    let mut best: Option<(u128, Tree, f64, usize)> = None;
    let mut schedules = Vec::new();
    for restart in 0..restarts {
        let strength = 18.0 + (rng.next() % 1000) as f64 / 1000.0 * 18.0;
        let levels = 180usize;
        let betas: Vec<f64> = (0..levels)
            .map(|i| 0.01 + (strength - 0.01) * i as f64 / (levels - 1) as f64)
            .collect();
        let config = TreeSA {
            betas: betas.clone(),
            ntrials: trials,
            niters: iterations,
            initializer: Initializer::Random,
            score: ScoreFunction::time_optimized(),
            decomposition_type: omeco::expr_tree::DecompositionType::Tree,
        };
        let nested = optimize_code(&code, &sizes, &config)
            .ok_or("omeco TreeSA could not plan this network")?;
        let tree = from_omeco(&nested, &input_labels, &output);
        let complexity = analyze_tree(&tree, &input_labels, dims)?;
        schedules.push(serde_json::json!({
            "restart": restart,
            "beta_min": 0.01,
            "beta_max": strength,
            "levels": levels,
            "work": complexity.work,
        }));
        if best
            .as_ref()
            .is_none_or(|(old_work, _, _, _)| complexity.work < *old_work)
        {
            best = Some((complexity.work, tree, strength, restart));
        }
    }
    let (_, tree, _, _) = best.expect("TreeSA ran at least one restart");
    Ok(Plan {
        schema: "amat5315-week9-order-v1".into(),
        method: "treesa".into(),
        seed: Some(seed),
        evidence: evidence.into(),
        input_labels,
        output,
        tree,
        search_settings: BTreeMap::from([
            ("optimizer".into(), serde_json::json!("omeco 0.2.6 TreeSA")),
            ("restarts".into(), serde_json::json!(restarts)),
            ("trials_per_restart".into(), serde_json::json!(trials)),
            (
                "iterations_per_temperature".into(),
                serde_json::json!(iterations),
            ),
            (
                "initializer".into(),
                serde_json::json!("Random, seeded by omeco trial RNG"),
            ),
            ("score".into(), serde_json::json!("time_optimized: tc only")),
            ("decomposition".into(), serde_json::json!("Tree")),
            ("schedules".into(), serde_json::json!(schedules)),
        ]),
    })
}

fn analyze_node(
    tree: &Tree,
    dims: &[usize],
    steps: &mut Vec<Step>,
) -> Result<(u128, u128), String> {
    match tree {
        Tree::Leaf { .. } => Ok((0, 1)),
        Tree::Node {
            left,
            right,
            inputs,
            retained,
        } => {
            if inputs.len() != 2 {
                return Err("each plan node must have two inputs".into());
            }
            let (left_work, left_max) = analyze_node(left, dims, steps)?;
            let (right_work, right_max) = analyze_node(right, dims, steps)?;
            let loop_labels = label_union(&[&inputs[0], &inputs[1]]);
            let cost = product(&loop_labels, dims);
            let result_entries = product(retained, dims);
            let index = steps.len();
            steps.push(Step {
                index,
                inputs: inputs.clone(),
                retained: retained.clone(),
                loop_labels,
                cost,
                result_entries,
            });
            Ok((
                left_work + right_work + cost,
                left_max.max(right_max).max(result_entries),
            ))
        }
    }
}

pub fn analyze_tree(
    tree: &Tree,
    input_labels: &[Vec<usize>],
    dims: &[usize],
) -> Result<Complexity, String> {
    let mut steps = Vec::new();
    let (work, largest_intermediate) = analyze_node(tree, dims, &mut steps)?;
    let largest_input = input_labels
        .iter()
        .map(|labels| product(labels, dims))
        .max()
        .unwrap_or(1);
    let largest_tensor = largest_input.max(largest_intermediate);
    Ok(Complexity {
        work,
        largest_tensor,
        largest_input,
        largest_intermediate,
        steps,
    })
}

impl Plan {
    pub fn complexity(&self, dims: &[usize]) -> Result<Complexity, String> {
        analyze_tree(&self.tree, &self.input_labels, dims)
    }

    pub fn with_output(&self, output: Vec<usize>) -> Result<Self, String> {
        let tree = rewrite_output(&self.tree, &self.input_labels, &output);
        Ok(Self {
            output,
            tree,
            ..self.clone()
        })
    }

    pub fn validate(&self, input_labels: &[Vec<usize>], output: &[usize]) -> Result<(), String> {
        if self.input_labels != input_labels {
            return Err("saved plan input factors do not match this model/evidence".into());
        }
        if self.output != output {
            return Err("saved plan requests a different scalar output".into());
        }
        let leaves = self.tree.leaves();
        if leaves != (0..input_labels.len()).collect::<Vec<_>>() {
            return Err("saved plan does not use every factor exactly once".into());
        }
        Ok(())
    }
}

fn rewrite_output(tree: &Tree, input_labels: &[Vec<usize>], output: &[usize]) -> Tree {
    match tree {
        Tree::Leaf { tensor } => Tree::Leaf { tensor: *tensor },
        Tree::Node { left, right, .. } => make_node(
            rewrite_output(left, input_labels, output),
            rewrite_output(right, input_labels, output),
            input_labels,
            output,
        ),
    }
}
