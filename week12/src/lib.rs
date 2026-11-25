use std::collections::BTreeSet;
use std::time::Instant;

use highs::{ColProblem, HighsModelStatus, RowProblem, Sense};
use serde::{Deserialize, Serialize};

pub const FRACTIONAL_EPS: f64 = 1e-6;
pub const PRUNE_EPS: f64 = 1e-6;
pub const BRANCH_PRUNE_GAIN: f64 = 1e6;

pub const FEATURES: [&str; 12] = [
    "x_val",
    "frac_dist",
    "field_h",
    "abs_field_h",
    "deg_norm",
    "bond_bias",
    "eff_field",
    "abs_eff_field",
    "mean_bond_dual",
    "mean_bond_gap",
    "mean_nbr_frac",
    "gap_ratio",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub j: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpinGlass {
    pub id: usize,
    pub n: usize,
    pub edges: Vec<Edge>,
    pub fields: Vec<f64>,
}

impl SpinGlass {
    pub fn square() -> Self {
        Self {
            id: 0,
            n: 4,
            edges: vec![
                Edge { u: 0, v: 1, j: 1.0 },
                Edge { u: 1, v: 2, j: 1.0 },
                Edge { u: 2, v: 3, j: 1.0 },
                Edge {
                    u: 3,
                    v: 0,
                    j: -1.0,
                },
            ],
            fields: vec![0.0; 4],
        }
    }

    pub fn tetrahedron() -> Self {
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        Self {
            id: 0,
            n: 4,
            edges: pairs
                .into_iter()
                .map(|(u, v)| Edge { u, v, j: 1.0 })
                .collect(),
            fields: vec![1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn degrees(&self) -> Vec<usize> {
        let mut degree = vec![0; self.n];
        for edge in &self.edges {
            degree[edge.u] += 1;
            degree[edge.v] += 1;
        }
        degree
    }

    pub fn incident(&self) -> Vec<Vec<usize>> {
        let mut incident = vec![Vec::new(); self.n];
        for (e, edge) in self.edges.iter().enumerate() {
            incident[edge.u].push(e);
            incident[edge.v].push(e);
        }
        incident
    }

    pub fn physical_energy(&self, x: &[f64]) -> f64 {
        let spin: Vec<f64> = x.iter().map(|bit| 1.0 - 2.0 * bit).collect();
        self.edges
            .iter()
            .map(|e| e.j * spin[e.u] * spin[e.v])
            .sum::<f64>()
            + self
                .fields
                .iter()
                .zip(&spin)
                .map(|(h, s)| h * s)
                .sum::<f64>()
    }

    pub fn nearest_rounding(&self, x: &[f64]) -> (Vec<f64>, f64) {
        let bits: Vec<f64> = x
            .iter()
            .map(|value| if *value >= 0.5 { 1.0 } else { 0.0 })
            .collect();
        let energy = self.physical_energy(&bits);
        (bits, energy)
    }

    pub fn fractional_count(x: &[f64]) -> usize {
        x.iter()
            .filter(|value| **value > FRACTIONAL_EPS && **value < 1.0 - FRACTIONAL_EPS)
            .count()
    }
}

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, k: usize) -> usize {
        (self.next_u64() % k as u64) as usize
    }
}

pub fn random_instance(n: usize, m: usize, seed: u64, id: usize) -> SpinGlass {
    assert!(n >= 2 && m <= n * (n - 1) / 2);
    let mut initial = SplitMix64::new(seed ^ (id as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    let mut generator = SplitMix64::new(initial.next_u64());

    let mut edges = BTreeSet::new();
    for i in 0..n {
        while incident_count(&edges, i) < 2.min(n - 1) {
            let j = generator.index(n);
            if j != i {
                edges.insert((i.min(j), i.max(j)));
            }
        }
    }
    while edges.len() < m {
        let i = generator.index(n);
        let j = generator.index(n);
        if i != j {
            edges.insert((i.min(j), i.max(j)));
        }
    }

    let graph: Vec<(usize, usize)> = edges.into_iter().collect();
    let couplings: Vec<f64> = graph
        .iter()
        .map(|_| {
            if generator.next_u64() & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let fields = (0..n)
        .map(|_| (generator.next_u64() % 3) as f64 - 1.0)
        .collect();
    SpinGlass {
        id,
        n,
        edges: graph
            .into_iter()
            .zip(couplings)
            .map(|((u, v), j)| Edge { u, v, j })
            .collect(),
        fields,
    }
}

fn incident_count(edges: &BTreeSet<(usize, usize)>, vertex: usize) -> usize {
    edges
        .iter()
        .filter(|(u, v)| *u == vertex || *v == vertex)
        .count()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LpStatus {
    Optimal,
    Infeasible,
}

#[derive(Clone, Debug)]
pub struct LpResult {
    pub status: LpStatus,
    pub energy: f64,
    pub spins: Vec<f64>,
    pub bonds: Vec<f64>,
    pub row_duals: Vec<f64>,
}

fn build_problem(
    graph: &SpinGlass,
    lower: &[f64],
    upper: &[f64],
    integer_spins: bool,
) -> ColProblem {
    let mut problem = ColProblem::default();
    let mut rows = Vec::with_capacity(4 * graph.edges.len());
    for _ in &graph.edges {
        rows.push(problem.add_row(..=0.0));
        rows.push(problem.add_row(..=2.0));
        rows.push(problem.add_row(..=0.0));
        rows.push(problem.add_row(..=0.0));
    }
    for i in 0..graph.n {
        let mut factors = Vec::new();
        for (e, edge) in graph.edges.iter().enumerate() {
            if edge.u == i {
                factors.push((rows[4 * e], -1.0));
                factors.push((rows[4 * e + 1], 1.0));
                factors.push((rows[4 * e + 2], 1.0));
                factors.push((rows[4 * e + 3], -1.0));
            } else if edge.v == i {
                factors.push((rows[4 * e], -1.0));
                factors.push((rows[4 * e + 1], 1.0));
                factors.push((rows[4 * e + 2], -1.0));
                factors.push((rows[4 * e + 3], 1.0));
            }
        }
        let bounds = lower[i]..=upper[i];
        if integer_spins {
            problem.add_integer_column(-2.0 * graph.fields[i], bounds, &factors);
        } else {
            problem.add_column(-2.0 * graph.fields[i], bounds, &factors);
        }
    }
    for (e, edge) in graph.edges.iter().enumerate() {
        problem.add_column(
            -2.0 * edge.j,
            0.0..=1.0,
            [
                (rows[4 * e], 1.0),
                (rows[4 * e + 1], 1.0),
                (rows[4 * e + 2], -1.0),
                (rows[4 * e + 3], -1.0),
            ],
        );
    }
    problem
}

pub fn solve_lp(graph: &SpinGlass, lower: &[f64], upper: &[f64]) -> Result<LpResult, String> {
    let problem = build_problem(graph, lower, upper, false);
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1);
    let solved = model.solve();
    if matches!(
        solved.status(),
        HighsModelStatus::Infeasible | HighsModelStatus::UnboundedOrInfeasible
    ) {
        return Ok(LpResult {
            status: LpStatus::Infeasible,
            energy: f64::INFINITY,
            spins: Vec::new(),
            bonds: Vec::new(),
            row_duals: Vec::new(),
        });
    }
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("LP status was {:?}", solved.status()));
    }
    let solution = solved.get_solution();
    let constant = graph.edges.iter().map(|e| e.j).sum::<f64>() + graph.fields.iter().sum::<f64>();
    Ok(LpResult {
        status: LpStatus::Optimal,
        energy: solved.objective_value() + constant,
        spins: solution.columns()[..graph.n].to_vec(),
        bonds: solution.columns()[graph.n..].to_vec(),
        row_duals: solution.dual_rows().to_vec(),
    })
}

pub fn solve_milp(graph: &SpinGlass) -> Result<(Vec<f64>, f64), String> {
    let lower = vec![0.0; graph.n];
    let upper = vec![1.0; graph.n];
    let problem = build_problem(graph, &lower, &upper, true);
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1);
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("MILP status was {:?}", solved.status()));
    }
    let bits: Vec<f64> = solved.get_solution().columns()[..graph.n]
        .iter()
        .map(|value| if *value >= 0.5 { 1.0 } else { 0.0 })
        .collect();
    let energy = graph.physical_energy(&bits);
    Ok((bits, energy))
}

pub fn solve_diet() -> Result<(Vec<f64>, f64), String> {
    let mut problem = RowProblem::default();
    let x1 = problem.add_column(6.0, 0.0..);
    let x2 = problem.add_column(3.5, 0.0..);
    problem.add_row(8.0.., [(x2, 7.0)]);
    problem.add_row(15.0.., [(x1, 4.0), (x2, 2.0)]);
    problem.add_row(3.0.., [(x1, 2.0), (x2, 1.0)]);
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1);
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("diet LP status was {:?}", solved.status()));
    }
    Ok((
        solved.get_solution().columns().to_vec(),
        solved.objective_value(),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Policy {
    MostFrac,
    Strong,
    Neural,
    RootNeural,
}

impl Policy {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "most-frac" => Ok(Self::MostFrac),
            "strong" => Ok(Self::Strong),
            "neural" => Ok(Self::Neural),
            "root-neural" => Ok(Self::RootNeural),
            _ => Err(format!("unknown policy {value}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::MostFrac => "most-frac",
            Self::Strong => "strong",
            Self::Neural => "neural",
            Self::RootNeural => "root-neural",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkWeights {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Probe {
    pub instance: usize,
    pub node: usize,
    pub candidates: Vec<usize>,
    pub features: Vec<Vec<f64>>,
    pub logits: Vec<f64>,
    pub probabilities: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSplit {
    pub train_instances: Vec<usize>,
    pub validation_instances: Vec<usize>,
    pub overlap: Vec<usize>,
    pub train_nodes: usize,
    pub validation_nodes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFile {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub weights: NetworkWeights,
    pub checkpoint_epoch: usize,
    pub best_validation_cross_entropy: f64,
    pub split: ModelSplit,
    pub probes: Vec<Probe>,
}

impl ModelFile {
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let model: Self = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        if model.mean.len() != FEATURES.len() || model.std.len() != FEATURES.len() {
            return Err("model feature statistics have the wrong length".into());
        }
        Ok(model)
    }

    pub fn logits(&self, features: &[f64]) -> Result<f64, String> {
        if features.len() != FEATURES.len() {
            return Err("feature vector must have 12 entries".into());
        }
        let hidden = self.weights.b1.len();
        if self.weights.w1.len() != FEATURES.len()
            || self.weights.w2.len() != hidden
            || self.weights.w1.iter().any(|row| row.len() != hidden)
        {
            return Err("inconsistent network weight shapes".into());
        }
        let mut logit = self.weights.b2;
        for j in 0..hidden {
            let mut value = self.weights.b1[j];
            for (k, feature) in features.iter().enumerate() {
                let std = self.std[k].max(1e-6);
                value += self.weights.w1[k][j] * (feature - self.mean[k]) / std;
            }
            logit += self.weights.w2[j] * value.tanh();
        }
        Ok(logit)
    }

    pub fn probabilities(&self, features: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let logits = features
            .iter()
            .map(|f| self.logits(f))
            .collect::<Result<Vec<_>, _>>()?;
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|value| (value - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        Ok((logits, exps.iter().map(|value| value / sum).collect()))
    }

    pub fn probe_error(&self) -> Result<f64, String> {
        let mut largest: f64 = 0.0;
        for probe in &self.probes {
            let (_, probabilities) = self.probabilities(&probe.features)?;
            if probabilities.len() != probe.probabilities.len() {
                return Err("probe probability length mismatch".into());
            }
            for (rust, jax) in probabilities.iter().zip(&probe.probabilities) {
                largest = largest.max((rust - jax).abs());
            }
        }
        Ok(largest)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LabelCandidate {
    pub spin: usize,
    pub gain0: f64,
    pub gain1: f64,
    pub score: f64,
    pub pruned_children: usize,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LabelNode {
    pub instance: usize,
    pub node: usize,
    pub candidates: Vec<usize>,
    pub features: Vec<Vec<f64>>,
    pub gains0: Vec<f64>,
    pub gains1: Vec<f64>,
    pub scores: Vec<f64>,
    pub pruned_children: Vec<usize>,
    pub best_candidates: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SplitRecord {
    pub spin: usize,
    pub x: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TraceNode {
    pub node: usize,
    pub branch: Vec<(usize, u8)>,
    pub status: String,
    pub lp_bound: f64,
    pub spins: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InstanceSearch {
    pub id: usize,
    pub energy: f64,
    pub error: f64,
    pub nodes: usize,
    pub lp_solves: usize,
    pub wall_ms: f64,
    pub lp_ms: f64,
    pub inference_ms: f64,
    pub first_splits: Vec<SplitRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchOutput {
    pub energy: f64,
    pub nodes: usize,
    pub lp_solves: usize,
    pub wall_ms: f64,
    pub lp_ms: f64,
    pub inference_ms: f64,
    pub first_splits: Vec<SplitRecord>,
    pub trace: Vec<TraceNode>,
    pub labels: Vec<LabelNode>,
}

#[derive(Clone, Debug)]
struct NodeBounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
    branch: Vec<(usize, u8)>,
}

#[derive(Clone, Debug)]
struct ProbeCandidate {
    spin: usize,
    gain0: f64,
    gain1: f64,
    score: f64,
    pruned_children: usize,
}

pub fn candidate_features(
    graph: &SpinGlass,
    i: usize,
    candidate: &[usize],
    lp: &LpResult,
    incumbent: f64,
) -> Vec<f64> {
    let incident = graph.incident();
    let degree = graph.degrees();
    let max_degree = degree.iter().copied().max().unwrap_or(1).max(1);
    let s: Vec<f64> = lp
        .spins
        .iter()
        .map(|x| 1.0 - 2.0 * x.clamp(0.0, 1.0))
        .collect();
    let frac: Vec<f64> = lp
        .spins
        .iter()
        .map(|x| 2.0 * x.clamp(0.0, 1.0).min(1.0 - x.clamp(0.0, 1.0)))
        .collect();
    let dual_scale = lp
        .row_duals
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let incident_edges = &incident[i];
    let local_degree = incident_edges.len().max(1);
    let positive = incident_edges
        .iter()
        .filter(|e| graph.edges[**e].j > 0.0)
        .count();
    let negative = incident_edges
        .iter()
        .filter(|e| graph.edges[**e].j < 0.0)
        .count();
    let bond_bias = (positive as f64 - negative as f64) / local_degree as f64;
    let mut effective_field = graph.fields[i];
    let mut mean_bond_dual = 0.0;
    let mut mean_bond_gap = 0.0;
    let mut neighbor_frac = 0.0;
    for &e in incident_edges {
        let edge = &graph.edges[e];
        let neighbor = if edge.u == i { edge.v } else { edge.u };
        effective_field += edge.j * s[neighbor];
        let dual_sum = lp.row_duals[4 * e..4 * e + 4]
            .iter()
            .map(|value| value.abs())
            .sum::<f64>();
        mean_bond_dual += dual_sum / dual_scale;
        let expected = 0.5 * (1.0 - s[i] * s[neighbor]);
        mean_bond_gap += (lp.bonds[e] - expected).abs();
        if candidate.contains(&neighbor) {
            neighbor_frac += frac[neighbor];
        }
    }
    let local_degree_f = incident_edges.len().max(1) as f64;
    mean_bond_dual /= local_degree_f;
    mean_bond_gap /= local_degree_f;
    neighbor_frac /= local_degree_f;
    let effective_field = effective_field / (incident_edges.len() as f64 + 1.0);
    let gap_ratio = if incumbent.is_finite() {
        ((incumbent - lp.energy) / incumbent.abs().max(1.0)).clamp(0.0, 2.0)
    } else {
        1.0
    };
    vec![
        lp.spins[i].clamp(0.0, 1.0),
        frac[i],
        graph.fields[i],
        graph.fields[i].abs(),
        degree[i] as f64 / max_degree as f64,
        bond_bias,
        effective_field,
        effective_field.abs(),
        mean_bond_dual,
        mean_bond_gap,
        neighbor_frac,
        gap_ratio,
    ]
}

fn select_most_frac(lp: &LpResult, candidates: &[usize]) -> usize {
    candidates
        .iter()
        .copied()
        .min_by(|a, b| {
            let da = (lp.spins[*a] - 0.5).abs();
            let db = (lp.spins[*b] - 0.5).abs();
            da.partial_cmp(&db).unwrap().then(a.cmp(b))
        })
        .expect("candidate list is nonempty")
}

fn probe_strong(
    graph: &SpinGlass,
    node: &NodeBounds,
    lp: &LpResult,
    candidates: &[usize],
    incumbent: f64,
) -> Result<Vec<ProbeCandidate>, String> {
    let mut probes = Vec::with_capacity(candidates.len());
    for &i in candidates {
        let mut gains = [0.0; 2];
        let mut pruned = 0;
        for (bit, gain) in gains.iter_mut().enumerate() {
            let mut lower = node.lower.clone();
            let mut upper = node.upper.clone();
            lower[i] = bit as f64;
            upper[i] = bit as f64;
            let child = solve_lp(graph, &lower, &upper)?;
            *gain = if child.status == LpStatus::Infeasible
                || (incumbent.is_finite() && child.energy >= incumbent - PRUNE_EPS)
            {
                BRANCH_PRUNE_GAIN
            } else {
                child.energy - lp.energy
            };
            if *gain >= BRANCH_PRUNE_GAIN {
                pruned += 1;
            }
        }
        probes.push(ProbeCandidate {
            spin: i,
            gain0: gains[0],
            gain1: gains[1],
            score: gains[0].max(1e-6) * gains[1].max(1e-6),
            pruned_children: pruned,
        });
    }
    Ok(probes)
}

fn best_strong(probes: &[ProbeCandidate]) -> usize {
    let mut best = 0;
    for i in 1..probes.len() {
        if probes[i].score > probes[best].score + 1e-15 {
            best = i;
        }
    }
    best
}

fn select_neural(
    graph: &SpinGlass,
    candidates: &[usize],
    lp: &LpResult,
    incumbent: f64,
    model: &ModelFile,
) -> Result<(usize, f64), String> {
    let mut best_spin = candidates[0];
    let mut best_score = f64::NEG_INFINITY;
    for &spin in candidates {
        let features = candidate_features(graph, spin, candidates, lp, incumbent);
        let score = model.logits(&features)?;
        if score > best_score + 1e-15 {
            best_score = score;
            best_spin = spin;
        }
    }
    Ok((best_spin, best_score))
}

fn select_root_neural(
    graph: &SpinGlass,
    candidates: &[usize],
    lp: &LpResult,
    model: &ModelFile,
    root_scores: &mut Option<Vec<f64>>,
    inference_ms: &mut f64,
) -> Result<usize, String> {
    if root_scores.is_none() {
        let started = Instant::now();
        let features: Vec<Vec<f64>> = (0..graph.n)
            .map(|spin| candidate_features(graph, spin, candidates, lp, f64::INFINITY))
            .collect();
        let raw: Vec<f64> = features
            .iter()
            .map(|feature| model.logits(feature))
            .collect::<Result<Vec<_>, _>>()?;
        let mean = raw.iter().sum::<f64>() / raw.len() as f64;
        let variance = raw.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / raw.len() as f64;
        let scale = variance.sqrt().max(1e-6);
        *root_scores = Some(raw.iter().map(|s| (s - mean) / scale).collect());
        *inference_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    let scores = root_scores.as_ref().unwrap();
    let mut best = candidates[0];
    let mut best_score = scores[best] + 3.0 * 2.0 * lp.spins[best].min(1.0 - lp.spins[best]);
    for &i in &candidates[1..] {
        let score = scores[i] + 3.0 * 2.0 * lp.spins[i].min(1.0 - lp.spins[i]);
        if score > best_score + 1e-15 {
            best = i;
            best_score = score;
        }
    }
    Ok(best)
}

#[derive(Clone, Debug)]
pub struct SearchConfig<'a> {
    pub policy: Policy,
    pub model: Option<&'a ModelFile>,
    pub export_labels: bool,
    pub trace: bool,
}

pub fn branch_and_bound(
    graph: &SpinGlass,
    config: &SearchConfig<'_>,
) -> Result<SearchOutput, String> {
    if matches!(config.policy, Policy::Neural | Policy::RootNeural) && config.model.is_none() {
        return Err("neural policy requires a model".into());
    }
    let wall_started = Instant::now();
    let mut stack = vec![NodeBounds {
        lower: vec![0.0; graph.n],
        upper: vec![1.0; graph.n],
        branch: Vec::new(),
    }];
    let mut incumbent = f64::INFINITY;
    let mut nodes = 0;
    let mut lp_solves = 0;
    let mut lp_ms = 0.0;
    let mut inference_ms = 0.0;
    let mut trace = Vec::new();
    let mut labels = Vec::new();
    let mut root_scores = None;
    let mut first_splits = Vec::new();

    while let Some(node) = stack.pop() {
        nodes += 1;
        let solve_started = Instant::now();
        let lp = solve_lp(graph, &node.lower, &node.upper)?;
        lp_solves += 1;
        lp_ms += solve_started.elapsed().as_secs_f64() * 1000.0;
        let mut status = "fractional";
        if lp.status == LpStatus::Infeasible {
            status = "infeasible";
        } else if incumbent.is_finite() && lp.energy >= incumbent - PRUNE_EPS {
            status = "pruned_bound";
        }
        if status != "fractional" {
            if config.trace {
                trace.push(TraceNode {
                    node: nodes,
                    branch: node.branch,
                    status: status.into(),
                    lp_bound: lp.energy,
                    spins: lp.spins,
                });
            }
            continue;
        }
        let candidates: Vec<usize> = (0..graph.n)
            .filter(|i| lp.spins[*i] > FRACTIONAL_EPS && lp.spins[*i] < 1.0 - FRACTIONAL_EPS)
            .collect();
        if candidates.is_empty() {
            incumbent = graph.physical_energy(&lp.spins);
            status = "incumbent";
            if config.trace {
                trace.push(TraceNode {
                    node: nodes,
                    branch: node.branch,
                    status: status.into(),
                    lp_bound: lp.energy,
                    spins: lp.spins,
                });
            }
            continue;
        }

        let mut probes = Vec::new();
        let selected = match config.policy {
            Policy::MostFrac => select_most_frac(&lp, &candidates),
            Policy::Strong => {
                let started = Instant::now();
                probes = probe_strong(graph, &node, &lp, &candidates, incumbent)?;
                lp_ms += started.elapsed().as_secs_f64() * 1000.0;
                lp_solves += 2 * candidates.len();
                probes[best_strong(&probes)].spin
            }
            Policy::Neural => {
                let started = Instant::now();
                let selected =
                    select_neural(graph, &candidates, &lp, incumbent, config.model.unwrap())?.0;
                inference_ms += started.elapsed().as_secs_f64() * 1000.0;
                selected
            }
            Policy::RootNeural => select_root_neural(
                graph,
                &candidates,
                &lp,
                config.model.unwrap(),
                &mut root_scores,
                &mut inference_ms,
            )?,
        };

        let instance_labels = labels
            .iter()
            .filter(|label: &&LabelNode| label.instance == graph.id)
            .count();
        if config.export_labels
            && config.policy == Policy::Strong
            && candidates.len() >= 2
            && instance_labels < 80
        {
            let best_score = probes[best_strong(&probes)].score;
            let tolerance = 1e-9 * best_score.abs().max(1.0);
            let best_candidates = probes
                .iter()
                .filter(|probe| (probe.score - best_score).abs() <= tolerance)
                .map(|probe| probe.spin)
                .collect();
            labels.push(LabelNode {
                instance: graph.id,
                node: nodes,
                candidates: candidates.clone(),
                features: candidates
                    .iter()
                    .map(|spin| candidate_features(graph, *spin, &candidates, &lp, incumbent))
                    .collect(),
                gains0: probes.iter().map(|probe| probe.gain0).collect(),
                gains1: probes.iter().map(|probe| probe.gain1).collect(),
                scores: probes.iter().map(|probe| probe.score).collect(),
                pruned_children: probes.iter().map(|probe| probe.pruned_children).collect(),
                best_candidates,
            });
        }

        if first_splits.len() < 2 {
            first_splits.push(SplitRecord {
                spin: selected,
                x: lp.spins[selected],
            });
        }
        if config.trace {
            trace.push(TraceNode {
                node: nodes,
                branch: node.branch.clone(),
                status: status.into(),
                lp_bound: lp.energy,
                spins: lp.spins.clone(),
            });
        }

        let near = if lp.spins[selected] >= 0.5 { 1 } else { 0 };
        let far = 1 - near;
        let mut children = Vec::with_capacity(2);
        for bit in [far, near] {
            let mut lower = node.lower.clone();
            let mut upper = node.upper.clone();
            lower[selected] = bit as f64;
            upper[selected] = bit as f64;
            let mut branch = node.branch.clone();
            branch.push((selected, bit));
            children.push(NodeBounds {
                lower,
                upper,
                branch,
            });
        }
        stack.extend(children);
    }
    if !incumbent.is_finite() {
        return Err("branch-and-bound found no incumbent".into());
    }
    Ok(SearchOutput {
        energy: incumbent,
        nodes,
        lp_solves,
        wall_ms: wall_started.elapsed().as_secs_f64() * 1000.0,
        lp_ms,
        inference_ms,
        first_splits,
        trace,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn generator_matches_contract() {
        let graph = random_instance(22, 52, 2026, 0);
        assert_eq!(graph.edges.len(), 52);
        assert_eq!(graph.fields.len(), 22);
        assert!(
            graph
                .edges
                .windows(2)
                .all(|pair| { (pair[0].u, pair[0].v) < (pair[1].u, pair[1].v) })
        );
        assert!(graph.degrees().iter().all(|degree| *degree >= 2));
    }

    #[test]
    fn gap_benchmark_matches_sheet_reference() {
        let mut exact_sum = 0.0;
        let mut gap_sum = 0.0;
        let mut largest_excess: f64 = 0.0;
        for id in 0..16 {
            let graph = random_instance(22, 52, 2026, id);
            let lp = solve_lp(&graph, &[0.0; 22], &[1.0; 22]).unwrap();
            let (_, exact) = solve_milp(&graph).unwrap();
            let (_, rounded) = graph.nearest_rounding(&lp.spins);
            exact_sum += exact;
            gap_sum += exact - lp.energy;
            largest_excess = largest_excess.max(rounded - exact);
        }
        assert_abs_diff_eq!(exact_sum / 16.0, -38.25, epsilon = 1e-9);
        assert_abs_diff_eq!(gap_sum / 16.0, 13.75, epsilon = 1e-9);
        assert_abs_diff_eq!(largest_excess, 60.0, epsilon = 1e-9);
    }

    #[test]
    fn opening_square_and_diet() {
        let square = SpinGlass::square();
        let lp = solve_lp(&square, &[0.0; 4], &[1.0; 4]).unwrap();
        assert_abs_diff_eq!(lp.energy, -4.0, epsilon = 1e-9);
        assert!(lp.spins.iter().all(|x| (*x - 0.5).abs() < 1e-9));
        let (_, exact) = solve_milp(&square).unwrap();
        assert_abs_diff_eq!(exact, -2.0, epsilon = 1e-9);
        let (_, rounded) = square.nearest_rounding(&lp.spins);
        assert_abs_diff_eq!(rounded, 2.0, epsilon = 1e-9);
        let (diet, cost) = solve_diet().unwrap();
        assert_abs_diff_eq!(diet[0], 89.0 / 28.0, epsilon = 1e-7);
        assert_abs_diff_eq!(diet[1], 8.0 / 7.0, epsilon = 1e-7);
        assert_abs_diff_eq!(cost, 323.0 / 14.0, epsilon = 1e-7);
    }

    #[test]
    fn tetrahedron_visits_five_nodes() {
        let graph = SpinGlass::tetrahedron();
        let result = branch_and_bound(
            &graph,
            &SearchConfig {
                policy: Policy::MostFrac,
                model: None,
                export_labels: false,
                trace: true,
            },
        )
        .unwrap();
        assert_abs_diff_eq!(result.energy, -3.0, epsilon = 1e-9);
        assert_eq!(result.nodes, 5);
        assert_eq!(result.lp_solves, result.nodes);
        assert_eq!(result.trace.len(), 5);
        assert_abs_diff_eq!(result.trace[0].lp_bound, -6.0, epsilon = 1e-9);
        assert_eq!(result.trace[0].branch, vec![]);
        assert_eq!(result.trace[1].branch, vec![(0, 1)]);
        assert_eq!(result.trace[2].branch, vec![(0, 1), (1, 1)]);
    }
}
