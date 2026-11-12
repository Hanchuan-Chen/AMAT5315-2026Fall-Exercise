use nalgebra::DMatrix;
use serde::Serialize;

#[derive(Clone)]
pub struct Core {
    pub left: usize,
    pub right: usize,
    pub data: Vec<f64>,
}

impl Core {
    pub fn new(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![0.0; left * 2 * right],
        }
    }

    pub fn get(&self, a: usize, p: usize, b: usize) -> f64 {
        self.data[(a * 2 + p) * self.right + b]
    }

    pub fn set(&mut self, a: usize, p: usize, b: usize, value: f64) {
        self.data[(a * 2 + p) * self.right + b] = value;
    }
}

#[derive(Clone, Serialize)]
pub struct BondDiagnostic {
    pub index: usize,
    pub kept: usize,
    pub discarded_squared: f64,
}

#[derive(Clone, Serialize)]
pub struct CompressionDiagnostic {
    pub row: usize,
    pub bonds: Vec<BondDiagnostic>,
}

pub fn compress(
    mut cores: Vec<Core>,
    chi: usize,
    row: usize,
) -> (Vec<Core>, CompressionDiagnostic) {
    let l = cores.len();
    for i in (1..l).rev() {
        let core = cores[i].clone();
        let a_dim = core.left;
        let b_dim = core.right;
        let m = DMatrix::from_fn(a_dim, 2 * b_dim, |a, p_b| {
            core.get(a, p_b / b_dim, p_b % b_dim)
        });
        let svd = m.svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let q = svd.singular_values.len();
        let mut right = Core::new(q, b_dim);
        for qq in 0..q {
            for p in 0..2 {
                for b in 0..b_dim {
                    right.set(qq, p, b, vt[(qq, p * b_dim + b)]);
                }
            }
        }
        let previous = &mut cores[i - 1];
        let mut updated = Core::new(previous.left, q);
        for a in 0..previous.left {
            for p in 0..2 {
                for qq in 0..q {
                    let mut value = 0.0;
                    for old_right in 0..previous.right {
                        value += previous.get(a, p, old_right)
                            * u[(old_right, qq)]
                            * svd.singular_values[qq];
                    }
                    updated.set(a, p, qq, value);
                }
            }
        }
        cores[i - 1] = updated;
        cores[i] = right;
    }

    let mut bonds = Vec::with_capacity(l.saturating_sub(1));
    for i in 0..l - 1 {
        let core = cores[i].clone();
        let m = DMatrix::from_fn(core.left * 2, core.right, |a_p, b| {
            core.get(a_p / 2, a_p % 2, b)
        });
        let svd = m.svd(true, true);
        let kept = chi.min(svd.singular_values.len());
        let discarded_squared = svd
            .singular_values
            .iter()
            .skip(kept)
            .map(|value| value * value)
            .sum();
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let mut left = Core::new(core.left, kept);
        for a in 0..core.left {
            for p in 0..2 {
                for q in 0..kept {
                    left.set(a, p, q, u[(a * 2 + p, q)]);
                }
            }
        }
        let next = &mut cores[i + 1];
        let mut updated = Core::new(kept, next.right);
        for q in 0..kept {
            for p in 0..2 {
                for b in 0..next.right {
                    let mut value = 0.0;
                    for old_left in 0..next.left {
                        value +=
                            svd.singular_values[q] * vt[(q, old_left)] * next.get(old_left, p, b);
                    }
                    updated.set(q, p, b, value);
                }
            }
        }
        cores[i] = left;
        cores[i + 1] = updated;
        bonds.push(BondDiagnostic {
            index: i,
            kept,
            discarded_squared,
        });
    }
    (cores, CompressionDiagnostic { row, bonds })
}

pub fn last_norm(core: &Core) -> f64 {
    core.data
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

pub fn scale_last(core: &mut Core, scale: f64) {
    for value in &mut core.data {
        *value /= scale;
    }
}
