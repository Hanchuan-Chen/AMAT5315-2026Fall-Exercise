use std::time::Instant;

use serde::Serialize;

use crate::ising::{a_index, site_tensor};
use crate::mps::{self, CompressionDiagnostic, Core};

#[derive(Serialize)]
pub struct SavedCore {
    pub shape: [usize; 3],
    pub data: Vec<f64>,
}

#[derive(Serialize)]
pub struct BoundaryEvidence {
    pub method: &'static str,
    pub model: &'static str,
    pub geometry: &'static str,
    pub boundary: &'static str,
    pub hamiltonian: &'static str,
    pub l: usize,
    pub k: f64,
    pub chi: usize,
    pub log_z: f64,
    pub g: f64,
    pub solver_time_ms: f64,
    pub scales: Vec<f64>,
    pub log_scale: f64,
    pub compression: Vec<CompressionDiagnostic>,
    pub final_tensors: Vec<SavedCore>,
    pub recomputation: RecomputationData,
}

#[derive(Serialize)]
pub struct RecomputationData {
    pub site_tensor: Vec<f64>,
    pub missing_bond: [f64; 2],
    pub instructions: &'static str,
}

pub fn solve(l: usize, k: f64, chi: usize) -> BoundaryEvidence {
    assert!(l >= 2);
    assert!(k >= 0.0);
    assert!(chi >= 1);
    let start = Instant::now();
    let (a, _w, b0) = site_tensor(k);
    let b_vec = [b0, 0.0];
    let mut cores = vec![Core::new(1, 1); l];
    for core in &mut cores {
        core.set(0, 0, 0, b_vec[0]);
        core.set(0, 1, 0, b_vec[1]);
    }
    let mut scales = Vec::with_capacity(l);
    let mut compression = Vec::with_capacity(l);
    let mut log_scale = 0.0;

    for y in 0..l {
        let mut enlarged = Vec::with_capacity(l);
        for (x, old) in cores.iter().enumerate() {
            let left_leg = if x == 0 { 1 } else { 2 };
            let right_leg = if x == l - 1 { 1 } else { 2 };
            let left_dim = old.left * left_leg;
            let right_dim = old.right * right_leg;
            let mut next = Core::new(left_dim, right_dim);
            for alpha in 0..old.left {
                for left in 0..left_leg {
                    for down in 0..2 {
                        for beta in 0..old.right {
                            for right in 0..right_leg {
                                let mut value = 0.0;
                                for up in 0..2 {
                                    let tensor = if x == 0 {
                                        close_left(&a, &b_vec, up, down, right)
                                    } else if x == l - 1 {
                                        close_right(&a, &b_vec, up, down, left)
                                    } else {
                                        a[a_index(right, up, left, down)]
                                    };
                                    value += old.get(alpha, up, beta) * tensor;
                                }
                                next.set(
                                    alpha * left_leg + left,
                                    down,
                                    beta * right_leg + right,
                                    value,
                                );
                            }
                        }
                    }
                }
            }
            enlarged.push(next);
        }
        let (compressed, diagnostic) = mps::compress(enlarged, chi, y);
        cores = compressed;
        let scale = mps::last_norm(cores.last().unwrap());
        mps::scale_last(cores.last_mut().unwrap(), scale);
        scales.push(scale);
        log_scale += scale.ln();
        compression.push(diagnostic);
    }

    let mut vector = vec![1.0];
    for core in &cores {
        let mut next = vec![0.0; core.right];
        for (alpha, &left_value) in vector.iter().enumerate().take(core.left) {
            for (p, &bond_value) in b_vec.iter().enumerate() {
                let coefficient = left_value * bond_value;
                for (beta, slot) in next.iter_mut().enumerate() {
                    *slot += coefficient * core.get(alpha, p, beta);
                }
            }
        }
        vector = next;
    }
    let log_z = log_scale + vector[0].ln();
    let solver_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let final_tensors = cores
        .iter()
        .map(|core| SavedCore {
            shape: [core.left, 2, core.right],
            data: core.data.clone(),
        })
        .collect();

    BoundaryEvidence {
        method: "boundary",
        model: "zero-field square-lattice Ising; H=-sum_edges s_i s_j",
        geometry: "square",
        boundary: "open",
        hamiltonian: "H=-sum_edges s_i*s_j; J=k_B=1",
        l,
        k,
        chi,
        log_z,
        g: log_z / (l * l) as f64,
        solver_time_ms,
        scales,
        log_scale,
        compression,
        final_tensors,
        recomputation: RecomputationData {
            site_tensor: a.to_vec(),
            missing_bond: b_vec,
            instructions: "site tensor axes [r,u,l,d]; final MPS cores [left,down,right]; close bottom with b",
        },
    }
}

fn close_left(a: &[f64; 16], b: &[f64; 2], up: usize, down: usize, right: usize) -> f64 {
    (0..2)
        .map(|left| b[left] * a[a_index(right, up, left, down)])
        .sum()
}

fn close_right(a: &[f64; 16], b: &[f64; 2], up: usize, down: usize, left: usize) -> f64 {
    (0..2)
        .map(|right| b[right] * a[a_index(right, up, left, down)])
        .sum()
}
