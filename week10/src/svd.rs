use nalgebra::{DMatrix, DVector, SVD};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::Serialize;

#[derive(Serialize)]
pub struct SvdEvidence {
    pub size: usize,
    pub seed: u64,
    pub rng: &'static str,
    pub distribution: &'static str,
    pub singular_values: Vec<f64>,
    pub ranks: Vec<usize>,
    pub matrix: Vec<f64>,
    pub reconstructions: Vec<Vec<f64>>,
}

pub fn generate(size: usize, seed: u64, ranks: &[usize]) -> SvdEvidence {
    let mut rng = StdRng::seed_from_u64(seed);
    let matrix = DMatrix::from_fn(size, size, |_, _| rng.sample(StandardNormal));
    let SVD {
        u,
        singular_values,
        v_t,
        ..
    } = matrix.clone().svd(true, true);
    let u = u.unwrap();
    let v_t = v_t.unwrap();

    let reconstructions = ranks
        .iter()
        .map(|&rank| {
            let u_r = u.columns(0, rank).into_owned();
            let vt_r = v_t.rows(0, rank).into_owned();
            let s_r = DVector::from_iterator(rank, singular_values.iter().copied().take(rank));
            let reconstruction: DMatrix<f64> = u_r * DMatrix::from_diagonal(&s_r) * vt_r;
            reconstruction.iter().copied().collect()
        })
        .collect();

    SvdEvidence {
        size,
        seed,
        rng: "rand 0.8.8 StdRng (ChaCha12 seeded through SeedableRng::seed_from_u64)",
        distribution: "independent standard normal (rand_distr 0.4.3 StandardNormal)",
        singular_values: singular_values.iter().copied().collect(),
        ranks: ranks.to_vec(),
        matrix: matrix.iter().copied().collect(),
        reconstructions,
    }
}
