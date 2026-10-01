//! Enzyme kernel checks: the 2-D timestep and its forward and reverse derivatives.

use seismic::deriv::{self, Grid};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn timestep_forward_and_reverse_are_transposes() {
    let grid = Grid {
        nx: 5,
        nz: 4,
        invdx2: 1.0,
        dt: 0.2,
    };
    let n = grid.n();
    let mut rng = Rng::new(2026);
    let prev: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let c: Vec<f64> = (0..n).map(|_| 1.5 + 0.2 * rng.next()).collect();
    let footprint: Vec<f64> = (0..n).map(|_| rng.next().abs()).collect();
    let sigma: Vec<f64> = (0..n).map(|_| 0.1 * rng.next().abs()).collect();
    let den: Vec<f64> = sigma.iter().map(|s| 1.0 + grid.dt * s).collect();
    let factor = 0.7;

    let mut primal = vec![0.0; n];
    deriv::primal(&grid, &prev, &u, &c, &footprint, &sigma, &den, factor, &mut primal);

    let dprev: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let du: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let dc: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let mut out = vec![0.0; n];
    let mut dout = vec![0.0; n];
    deriv::forward(
        &grid, &prev, &dprev, &u, &du, &c, &dc, &footprint, &sigma, &den, factor,
        &mut out, &mut dout,
    );
    for (a, b) in primal.iter().zip(&out) {
        assert!((a - b).abs() < 1e-13);
    }

    let seed: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let mut seed_live = seed.clone();
    let mut ap = vec![0.0; n];
    let mut au = vec![0.0; n];
    let mut ac = vec![0.0; n];
    let mut scratch = vec![0.0; n];
    deriv::reverse(
        &grid, &prev, &mut ap, &u, &mut au, &c, &mut ac, &footprint, &sigma, &den, factor,
        &mut scratch, &mut seed_live,
    );
    let lhs = dot(&dout, &seed);
    let rhs = dot(&dprev, &ap) + dot(&du, &au) + dot(&dc, &ac);
    let relative = (lhs - rhs).abs() / lhs.abs().max(rhs.abs());
    assert!(relative < 1e-12, "transpose identity: {lhs} vs {rhs} ({relative})");
}

#[test]
fn reverse_matches_the_explicit_velocity_adjoint() {
    // A hand-written transpose of the same stencil, independent of Enzyme.
    let grid = Grid {
        nx: 5,
        nz: 4,
        invdx2: 1.0,
        dt: 0.2,
    };
    let n = grid.n();
    let mut rng = Rng::new(7);
    let prev: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let c: Vec<f64> = (0..n).map(|_| 1.5 + 0.2 * rng.next()).collect();
    let footprint = vec![0.0; n];
    let sigma = vec![0.0; n];
    let den = vec![1.0; n];
    let factor = 0.0;
    let seed: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let mut seed_live = seed.clone();
    let mut ap = vec![0.0; n];
    let mut au = vec![0.0; n];
    let mut ac = vec![0.0; n];
    let mut scratch = vec![0.0; n];
    deriv::reverse(
        &grid, &prev, &mut ap, &u, &mut au, &c, &mut ac, &footprint, &sigma, &den, factor,
        &mut scratch, &mut seed_live,
    );
    let dt2 = grid.dt * grid.dt;
    for z in 1..grid.nz - 1 {
        for x in 1..grid.nx - 1 {
            let i = z * grid.nx + x;
            let lap = u[i - 1] + u[i + 1] + u[i - grid.nx] + u[i + grid.nx] - 4.0 * u[i];
            assert!((ac[i] - 2.0 * dt2 * c[i] * lap * seed[i]).abs() < 1e-12);
        }
    }
}
