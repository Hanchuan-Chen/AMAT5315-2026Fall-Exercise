//! `field`: write a velocity field to stdout, as `week4/field.design.toml` fixes.
//!
//! `field taylor-green` writes the exact solution of Equation 15 at time `t`
//! (default 0); `field random` writes the seeded band-limited initial field
//! with `E(0) = 0.5`.  One JSON object leaves on stdout; `fluid` reads it.

use clap::{Parser, Subcommand};
use rand::{Rng, SeedableRng};
use serde_json::json;
use week4::line::{Complex, wavenumbers};
use week4::solver::Solver;

#[derive(Parser)]
#[command(name = "field", about = "write a velocity field to stdout")]
struct Cli {
    #[command(subcommand)]
    case: Case,
}

#[derive(Subcommand)]
enum Case {
    /// u = cos(x) sin(y) exp(-2 nu t), v = -sin(x) cos(y) exp(-2 nu t)
    #[command(name = "taylor-green")]
    TaylorGreen {
        /// grid points per side
        #[arg(long)]
        n: usize,
        /// time of the exact solution; default 0, the initial field
        #[arg(long, default_value_t = 0.0)]
        t: f64,
        /// viscosity of the decay; required if t > 0
        #[arg(long)]
        nu: Option<f64>,
    },
    /// vorticity modes of equal amplitude, k-min <= |k| <= k-max, E(0) = 0.5
    #[command(name = "random")]
    Random {
        /// grid points per side
        #[arg(long)]
        n: usize,
        /// seed of the phases, uniform in [0, 2pi)
        #[arg(long)]
        seed: u64,
        /// lowest |k| kept, inclusive
        #[arg(long = "k-min")]
        k_min: f64,
        /// highest |k| kept, inclusive
        #[arg(long = "k-max")]
        k_max: f64,
    },
}

fn main() {
    let cli = Cli::parse();
    let (case, n, seed, band, u, v) = match cli.case {
        Case::TaylorGreen { n, t, nu } => {
            let nu = match (t > 0.0, nu) {
                (true, None) => {
                    eprintln!("field: nu is required when t > 0");
                    std::process::exit(2);
                }
                (_, Some(nu)) => nu,
                (false, None) => 0.0,
            };
            let dx = 2.0 * std::f64::consts::PI / n as f64;
            let decay = (-2.0 * nu * t).exp();
            let mut u = vec![0.0; n * n];
            let mut v = vec![0.0; n * n];
            for y in 0..n {
                for x in 0..n {
                    let (xx, yy) = (x as f64 * dx, y as f64 * dx);
                    u[y * n + x] = xx.cos() * yy.sin() * decay;
                    v[y * n + x] = -xx.sin() * yy.cos() * decay;
                }
            }
            ("taylor-green".to_string(), n, None, None, u, v)
        }
        Case::Random {
            n,
            seed,
            k_min,
            k_max,
        } => {
            let solver = Solver::new(n, 0.0);
            let k = wavenumbers(n);
            // Every mode of the band, in an order independent of n.
            let mut modes: Vec<(i64, i64)> = Vec::new();
            for &ky in &k {
                for &kx in &k {
                    let r2 = kx * kx + ky * ky;
                    if r2 >= k_min * k_min && r2 <= k_max * k_max {
                        modes.push((ky as i64, kx as i64));
                    }
                }
            }
            modes.sort_unstable();
            // E(0) = 1/2 sum |omega_k|^2 / k^2 = 0.5 fixes the amplitude, the
            // same for every mode.
            let sum_inv_k2: f64 = modes
                .iter()
                .map(|(ky, kx)| 1.0 / (kx * kx + ky * ky) as f64)
                .sum();
            let amplitude = 1.0 / sum_inv_k2.sqrt();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut hat = vec![Complex::zero(); n * n];
            let scale = (n * n) as f64 * amplitude;
            for (ky, kx) in &modes {
                // one of each conjugate pair carries a fresh phase
                if !(*ky > 0 || (*ky == 0 && *kx > 0)) {
                    continue;
                }
                let phase: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let jx = k.iter().position(|&v| v as i64 == *kx).unwrap();
                let ly = k.iter().position(|&v| v as i64 == *ky).unwrap();
                let i = ly * n + jx;
                let mirror_x = (n - jx) % n;
                let mirror_y = (n - ly) % n;
                let m = mirror_y * n + mirror_x;
                let value = Complex::new(scale * phase.cos(), scale * phase.sin());
                hat[i] = value;
                hat[m] = value.conj();
                if i == m {
                    hat[i] = Complex::new(scale, 0.0);
                }
            }
            let omega = solver.ifft2(&hat);
            let (u, v) = solver.velocity(&omega);
            (
                "random".to_string(),
                n,
                Some(seed),
                Some(vec![k_min, k_max]),
                u,
                v,
            )
        }
    };
    let out = json!({
        "case": case,
        "n": n,
        "seed": seed,
        "k_band": band,
        "u": u,
        "v": v,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
