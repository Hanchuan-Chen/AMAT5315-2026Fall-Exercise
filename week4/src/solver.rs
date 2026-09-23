//! Part 2: the two-dimensional incompressible vorticity equation on `[0, 2pi)^2`.
//!
//! Fourier pseudospectral derivatives, the two-thirds rule on the retained
//! vorticity and on every product, one Poisson solve per rate evaluation, and
//! the energy and enstrophy of Equation 14.  The state is the vorticity on the
//! `n x n` grid in the sheet's row-major order, `index = y * n + x`.

use std::f64::consts::PI;

use crate::integrator::Integrator;
use crate::line::{Complex, fft, ifft, wavenumbers};

/// The periodic grid and the spectral operators that act on it.
pub struct Solver {
    pub n: usize,
    pub nu: f64,
    /// `k_x` on the grid, the Nyquist value included.
    kx: Vec<f64>,
    /// `k_y` on the grid, the Nyquist value included.
    ky: Vec<f64>,
    /// `i k_x` for the first derivative, zero at the Nyquist mode.
    ikx: Vec<Complex>,
    /// `i k_y` for the first derivative, zero at the Nyquist mode.
    iky: Vec<Complex>,
    /// `-(k_x^2 + k_y^2)`, the Laplacian.
    k2: Vec<f64>,
    /// The two-thirds rule, `|k_x|, |k_y| <= floor(n/3)`.
    keep: Vec<bool>,
}

impl Solver {
    pub fn new(n: usize, nu: f64) -> Self {
        assert!(n.is_power_of_two(), "the grid side must be a power of two");
        let k = wavenumbers(n);
        let nyquist = -(n as f64) / 2.0;
        let cutoff = (n / 3) as f64;
        let mut ikx = vec![Complex::zero(); n * n];
        let mut iky = vec![Complex::zero(); n * n];
        let mut k2 = vec![0.0; n * n];
        let mut keep = vec![false; n * n];
        let mut kx_flat = vec![0.0; n * n];
        let mut ky_flat = vec![0.0; n * n];
        for (ly, &ky) in k.iter().enumerate() {
            for (jx, &kx) in k.iter().enumerate() {
                let i = ly * n + jx;
                let odd_x = if kx == nyquist { 0.0 } else { kx };
                let odd_y = if ky == nyquist { 0.0 } else { ky };
                ikx[i] = Complex::new(0.0, odd_x);
                iky[i] = Complex::new(0.0, odd_y);
                k2[i] = kx * kx + ky * ky;
                keep[i] = kx.abs() <= cutoff && ky.abs() <= cutoff;
                kx_flat[i] = kx;
                ky_flat[i] = ky;
            }
        }
        Solver {
            n,
            nu,
            kx: kx_flat,
            ky: ky_flat,
            ikx,
            iky,
            k2,
            keep,
        }
    }

    /// Zero every mode outside the two-thirds rule.
    pub fn mask(&self, field_hat: &mut [Complex]) {
        for (value, &keep) in field_hat.iter_mut().zip(&self.keep) {
            if !keep {
                *value = Complex::zero();
            }
        }
    }

    fn fft2_inplace(&self, field: &mut [Complex]) {
        let n = self.n;
        let mut buf = vec![Complex::zero(); n];
        for ly in 0..n {
            buf.copy_from_slice(&field[ly * n..ly * n + n]);
            fft(&mut buf);
            field[ly * n..ly * n + n].copy_from_slice(&buf);
        }
        for jx in 0..n {
            for ly in 0..n {
                buf[ly] = field[ly * n + jx];
            }
            fft(&mut buf);
            for ly in 0..n {
                field[ly * n + jx] = buf[ly];
            }
        }
    }

    fn ifft2_inplace(&self, field_hat: &mut [Complex]) {
        let n = self.n;
        let mut buf = vec![Complex::zero(); n];
        for jx in 0..n {
            for ly in 0..n {
                buf[ly] = field_hat[ly * n + jx];
            }
            ifft(&mut buf);
            for ly in 0..n {
                field_hat[ly * n + jx] = buf[ly];
            }
        }
        for ly in 0..n {
            buf.copy_from_slice(&field_hat[ly * n..ly * n + n]);
            ifft(&mut buf);
            field_hat[ly * n..ly * n + n].copy_from_slice(&buf);
        }
    }

    pub fn fft2(&self, field: &[f64]) -> Vec<Complex> {
        let mut hat: Vec<Complex> = field
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        self.fft2_inplace(&mut hat);
        hat
    }

    pub fn ifft2(&self, field_hat: &[Complex]) -> Vec<f64> {
        let mut work = field_hat.to_vec();
        self.ifft2_inplace(&mut work);
        work.into_iter().map(|value| value.re).collect()
    }

    /// One derivative through a multiplier held in FFT order.
    fn derivative(&self, field: &[f64], multiplier: &[Complex]) -> Vec<f64> {
        let mut hat = self.fft2(field);
        for (value, m) in hat.iter_mut().zip(multiplier) {
            *value = *value * *m;
        }
        self.ifft2(&hat)
    }

    /// `d/dx`, the Nyquist row set to zero.
    pub fn dx(&self, field: &[f64]) -> Vec<f64> {
        self.derivative(field, &self.ikx)
    }

    /// `d/dy`, the Nyquist column set to zero.
    pub fn dy(&self, field: &[f64]) -> Vec<f64> {
        self.derivative(field, &self.iky)
    }

    /// `d^2/dx^2`.
    pub fn dxx(&self, field: &[f64]) -> Vec<f64> {
        let m: Vec<Complex> = self.kx.iter().map(|k| Complex::new(-k * k, 0.0)).collect();
        self.derivative(field, &m)
    }

    /// `d^2/dx dy`.
    pub fn dxdy(&self, field: &[f64]) -> Vec<f64> {
        let m: Vec<Complex> = self
            .kx
            .iter()
            .zip(&self.ky)
            .map(|(kx, ky)| Complex::new(-(kx * ky), 0.0))
            .collect();
        self.derivative(field, &m)
    }

    /// `-(k_x^2 + k_y^2) f`, the Laplacian.
    pub fn laplacian(&self, field: &[f64]) -> Vec<f64> {
        let m: Vec<Complex> = self.k2.iter().map(|&k2| Complex::new(-k2, 0.0)).collect();
        self.derivative(field, &m)
    }

    /// Second-order centred differences along each axis, periodic wrapping.
    pub fn fd_dx(field: &[f64], n: usize) -> Vec<f64> {
        let dx = 2.0 * PI / n as f64;
        let mut out = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                let up = field[y * n + (x + 1) % n];
                let down = field[y * n + (x + n - 1) % n];
                out[y * n + x] = (up - down) / (2.0 * dx);
            }
        }
        out
    }

    pub fn fd_dxx(field: &[f64], n: usize) -> Vec<f64> {
        let dx = 2.0 * PI / n as f64;
        let mut out = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                let up = field[y * n + (x + 1) % n];
                let down = field[y * n + (x + n - 1) % n];
                out[y * n + x] = (up - 2.0 * field[y * n + x] + down) / (dx * dx);
            }
        }
        out
    }

    pub fn fd_dxdy(field: &[f64], n: usize) -> Vec<f64> {
        let dx = 2.0 * PI / n as f64;
        let mut out = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                let a = field[((y + 1) % n) * n + (x + 1) % n];
                let b = field[((y + 1) % n) * n + (x + n - 1) % n];
                let c = field[((y + n - 1) % n) * n + (x + 1) % n];
                let d = field[((y + n - 1) % n) * n + (x + n - 1) % n];
                out[y * n + x] = (a - b - c + d) / (4.0 * dx * dx);
            }
        }
        out
    }

    pub fn fd_laplacian(field: &[f64], n: usize) -> Vec<f64> {
        let dx = 2.0 * PI / n as f64;
        let mut out = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                let up = field[y * n + (x + 1) % n];
                let down = field[y * n + (x + n - 1) % n];
                let left = field[((y + n - 1) % n) * n + x];
                let right = field[((y + 1) % n) * n + x];
                out[y * n + x] = (up + down + left + right - 4.0 * field[y * n + x]) / (dx * dx);
            }
        }
        out
    }

    /// The streamfunction of a vorticity field: `psi_hat = omega_hat / k^2`,
    /// with the zero mode fixed at zero.
    fn streamfunction(&self, omega_hat: &[Complex]) -> Vec<Complex> {
        omega_hat
            .iter()
            .zip(&self.k2)
            .map(|(value, &k2)| {
                if k2 == 0.0 {
                    Complex::zero()
                } else {
                    value.scale(1.0 / k2)
                }
            })
            .collect()
    }

    /// The velocity from a vorticity field: `u = d_y psi`, `v = -d_x psi`.
    pub fn velocity(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut hat = self.fft2(omega);
        self.mask(&mut hat);
        let psi = self.streamfunction(&hat);
        let uh: Vec<Complex> = psi.iter().zip(&self.iky).map(|(p, m)| *p * *m).collect();
        let vh: Vec<Complex> = psi
            .iter()
            .zip(&self.ikx)
            .map(|(p, m)| (*p * *m).scale(-1.0))
            .collect();
        (self.ifft2(&uh), self.ifft2(&vh))
    }

    /// The vorticity of a velocity field: `omega = d_x v - d_y u`.
    pub fn vorticity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        let dxv = self.dx(v);
        let dyu = self.dy(u);
        let omega: Vec<f64> = dxv.iter().zip(&dyu).map(|(a, b)| a - b).collect();
        let mut hat = self.fft2(&omega);
        self.mask(&mut hat);
        self.ifft2(&hat)
    }

    /// The rate function of the vorticity equation, Equation 3: advection
    /// through the streamfunction and viscous diffusion, every product
    /// dealiased by the two-thirds rule.
    pub fn rhs(&self, omega: &[f64]) -> Vec<f64> {
        let mut hat = self.fft2(omega);
        self.mask(&mut hat);
        let psi = self.streamfunction(&hat);
        let uh: Vec<Complex> = psi.iter().zip(&self.iky).map(|(p, m)| *p * *m).collect();
        let vh: Vec<Complex> = psi
            .iter()
            .zip(&self.ikx)
            .map(|(p, m)| (*p * *m).scale(-1.0))
            .collect();
        let u = self.ifft2(&uh);
        let v = self.ifft2(&vh);
        let mut hx = hat.clone();
        for (value, m) in hx.iter_mut().zip(&self.ikx) {
            *value = *value * *m;
        }
        let mut hy = hat.clone();
        for (value, m) in hy.iter_mut().zip(&self.iky) {
            *value = *value * *m;
        }
        let omega_x = self.ifft2(&hx);
        let omega_y = self.ifft2(&hy);
        let product: Vec<f64> = (0..self.n * self.n)
            .map(|i| u[i] * omega_x[i] + v[i] * omega_y[i])
            .collect();
        let mut product_hat = self.fft2(&product);
        self.mask(&mut product_hat);
        let rate_hat: Vec<Complex> = product_hat
            .iter()
            .zip(hat.iter().zip(&self.k2))
            .map(|(p, (w, k2))| p.scale(-1.0) + w.scale(-self.nu * k2))
            .collect();
        self.ifft2(&rate_hat)
    }

    /// One step of the integrator on the vorticity.
    pub fn step(&self, integrator: &dyn Integrator, omega: &[f64], h: f64) -> Vec<f64> {
        let mut rate = |y: &[f64], out: &mut [f64]| {
            out.copy_from_slice(&self.rhs(y));
        };
        integrator.step(omega, h, &mut rate)
    }

    /// The kinetic energy and the enstrophy, Equation 14.
    pub fn energy_enstrophy(&self, u: &[f64], v: &[f64], omega: &[f64]) -> (f64, f64) {
        let count = (self.n * self.n) as f64;
        let energy = 0.5 * u.iter().zip(v).map(|(a, b)| a * a + b * b).sum::<f64>() / count;
        let enstrophy = 0.5 * omega.iter().map(|w| w * w).sum::<f64>() / count;
        (energy, enstrophy)
    }
}

/// The grid function of Part 2's derivative check.
pub fn sin3x_cos2y(n: usize) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    let mut g = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            g[y * n + x] = (3.0 * x as f64 * dx).sin() * (2.0 * y as f64 * dx).cos();
        }
    }
    g
}

/// The analytic derivatives of `g = sin(3x) cos(2y)`.
pub fn analytic(n: usize) -> [Vec<f64>; 4] {
    let dx = 2.0 * PI / n as f64;
    let mut dxg = vec![0.0; n * n];
    let mut dxxg = vec![0.0; n * n];
    let mut dxdyg = vec![0.0; n * n];
    let mut lap = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let (xx, yy) = (x as f64 * dx, y as f64 * dx);
            let g = (3.0 * xx).sin() * (2.0 * yy).cos();
            let i = y * n + x;
            dxg[i] = 3.0 * (3.0 * xx).cos() * (2.0 * yy).cos();
            dxxg[i] = -9.0 * g;
            dxdyg[i] = -6.0 * (3.0 * xx).cos() * (2.0 * yy).sin();
            lap[i] = -13.0 * g;
        }
    }
    [dxg, dxxg, dxdyg, lap]
}

pub fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod dump {
    //! The derivative comparison of Part 2 check 1, printed by
    //! `scripts/derivatives.py`.

    use super::*;
    use serde_json::{Value, json};

    fn row(name: &str, fd32: f64, fd64: f64, fourier: f64) -> Value {
        json!({
            "name": name,
            "fd32": fd32,
            "fd64": fd64,
            "fourier": fourier,
            "ratio": fd32 / fd64,
        })
    }

    #[test]
    fn derivative_dump() {
        let n = 32;
        let solver = Solver::new(n, 0.0);
        let g = sin3x_cos2y(n);
        let exact = analytic(n);
        let fourier = [
            solver.dx(&g),
            solver.dxx(&g),
            solver.dxdy(&g),
            solver.laplacian(&g),
        ];
        let fd = [
            Solver::fd_dx(&g, n),
            Solver::fd_dxx(&g, n),
            Solver::fd_dxdy(&g, n),
            Solver::fd_laplacian(&g, n),
        ];
        let g64 = sin3x_cos2y(64);
        let exact64 = analytic(64);
        let fd_wide = [
            Solver::fd_dx(&g64, 64),
            Solver::fd_dxx(&g64, 64),
            Solver::fd_dxdy(&g64, 64),
            Solver::fd_laplacian(&g64, 64),
        ];
        let names = ["dx", "dxx", "dxdy", "laplacian"];
        let rows: Vec<Value> = (0..4)
            .map(|i| {
                row(
                    names[i],
                    max_abs_error(&fd[i], &exact[i]),
                    max_abs_error(&fd_wide[i], &exact64[i]),
                    max_abs_error(&fourier[i], &exact[i]),
                )
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&json!({"n": n, "rows": rows})).unwrap()
        );
    }
}
