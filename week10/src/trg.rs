use std::time::Instant;

use nalgebra::DMatrix;
use serde::Serialize;

use crate::ising::site_tensor;

#[derive(Clone, Serialize)]
pub struct SavedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Serialize)]
pub struct TrgStep {
    pub step: usize,
    pub sites: usize,
    pub max_abs_before: f64,
    pub singular_values: [Vec<f64>; 2],
    pub splits: [SavedTensor; 4],
}

#[derive(Serialize)]
pub struct TrgEvidence {
    pub method: &'static str,
    pub model: &'static str,
    pub geometry: &'static str,
    pub boundary: &'static str,
    pub hamiltonian: &'static str,
    pub l: usize,
    pub k: f64,
    pub chi: usize,
    pub steps: usize,
    pub log_z: f64,
    pub g: f64,
    pub solver_time_ms: f64,
    pub scales: Vec<f64>,
    pub log_scale: f64,
    pub closure: f64,
    pub history: Vec<TrgStep>,
    pub final_tensor: SavedTensor,
}

pub fn solve(l: usize, k: f64, chi: usize) -> TrgEvidence {
    assert!(l.is_power_of_two() && l >= 2);
    assert!(k >= 0.0);
    assert!(chi >= 1);
    let start = Instant::now();
    let (a, _w, _) = site_tensor(k);
    let mut tensor = a.to_vec();
    let mut dim = 2;
    let steps = 2 * l.trailing_zeros() as usize;
    let mut sites = l * l;
    let mut scales = Vec::with_capacity(steps);
    let mut log_scale = 0.0;
    let mut history = Vec::with_capacity(steps);

    for step in 0..steps {
        let max_abs = tensor.iter().map(|value| value.abs()).fold(0.0, f64::max);
        for value in &mut tensor {
            *value /= max_abs;
        }
        scales.push(max_abs);
        log_scale += sites as f64 * max_abs.ln();

        let (s1, s3, sv1) = split(&tensor, dim, chi, true);
        let (s2, s4, sv2) = split(&tensor, dim, chi, false);
        let new_dim = s1.shape[2];
        let d2 = dim * dim;
        let n2 = new_dim * new_dim;
        let left: DMatrix<f64> = DMatrix::from_fn(d2, n2, |wb, ru| {
            let (w, b) = (wb / dim, wb % dim);
            let (r, u) = (ru / new_dim, ru % new_dim);
            (0..dim).map(|a| s1.get(w, a, r) * s2.get(a, b, u)).sum()
        });
        let right: DMatrix<f64> = DMatrix::from_fn(d2, n2, |wb, ld| {
            let (w, b) = (wb / dim, wb % dim);
            let (l, d) = (ld / new_dim, ld % new_dim);
            (0..dim).map(|g| s3.get(l, b, g) * s4.get(d, g, w)).sum()
        });
        let regrouped: DMatrix<f64> = left.transpose() * right;
        let mut next = vec![0.0; new_dim.pow(4)];
        for r in 0..new_dim {
            for u in 0..new_dim {
                for l in 0..new_dim {
                    for d in 0..new_dim {
                        next[((r * new_dim + u) * new_dim + l) * new_dim + d] =
                            regrouped[(r * new_dim + u, l * new_dim + d)];
                    }
                }
            }
        }
        history.push(TrgStep {
            step,
            sites,
            max_abs_before: max_abs,
            singular_values: [sv1, sv2],
            splits: [s1.clone(), s2, s3, s4],
        });
        tensor = next;
        dim = new_dim;
        sites /= 2;
    }

    let mut closure = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            closure += tensor[((i * dim + j) * dim + i) * dim + j];
        }
    }
    let log_z = log_scale + closure.ln();
    TrgEvidence {
        method: "trg",
        model: "zero-field square-lattice Ising; H=-sum_edges s_i s_j",
        geometry: "square",
        boundary: "periodic",
        hamiltonian: "H=-sum_edges s_i*s_j; J=k_B=1",
        l,
        k,
        chi,
        steps,
        log_z,
        g: log_z / (l * l) as f64,
        solver_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        scales,
        log_scale,
        closure,
        history,
        final_tensor: SavedTensor {
            shape: vec![dim, dim, dim, dim],
            data: tensor,
        },
    }
}

fn split(
    tensor: &[f64],
    dim: usize,
    chi: usize,
    first: bool,
) -> (SavedTensor, SavedTensor, Vec<f64>) {
    let matrix = DMatrix::from_fn(dim * dim, dim * dim, |row, col| {
        let (i0, i1) = (row / dim, row % dim);
        let (j0, j1) = (col / dim, col % dim);
        if first {
            tensor[((j0 * dim + i1) * dim + i0) * dim + j1]
        } else {
            tensor[((j1 * dim + j0) * dim + i1) * dim + i0]
        }
    });
    let svd = matrix.svd(true, true);
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let kept = singular_values
        .iter()
        .take(chi)
        .take_while(|&&value| value > 1e-12)
        .count()
        .max(1);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut left = SavedTensor {
        shape: vec![dim, dim, kept],
        data: vec![0.0; dim * dim * kept],
    };
    let mut right = SavedTensor {
        shape: vec![kept, dim, dim],
        data: vec![0.0; kept * dim * dim],
    };
    for i in 0..dim {
        for j in 0..dim {
            for q in 0..kept {
                left.data[(i * dim + j) * kept + q] =
                    u[(i * dim + j, q)] * singular_values[q].sqrt();
                right.data[(q * dim + i) * dim + j] =
                    singular_values[q].sqrt() * vt[(q, i * dim + j)];
            }
        }
    }
    (left, right, singular_values)
}

impl SavedTensor {
    fn get(&self, a: usize, b: usize, c: usize) -> f64 {
        self.data[(a * self.shape[1] + b) * self.shape[2] + c]
    }
}
