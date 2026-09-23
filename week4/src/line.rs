//! Part 1: the advection-diffusion line `u_t + c u_x = nu u_xx` on `[0, 2pi)`.
//!
//! The state is the pulse sampled at `n` points.  The rate function is
//! Equation 6, once with Fourier multipliers and once with centred finite
//! differences (Equation 8), and the exact solution of the periodic Gaussian
//! is available for comparison.  The complex FFT used here and by the solver
//! lives in this module.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

use crate::integrator::{Integrator, Rate};

/// A complex number, enough arithmetic for the Fourier multipliers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    pub fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }

    pub fn scale(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }

    pub fn conj(self) -> Complex {
        Complex::new(self.re, -self.im)
    }

    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

fn bit_reverse(a: &mut [Complex]) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
}

/// The unnormalized forward DFT: `A_k = sum_j a_j exp(-2 pi i j k / n)`.
pub fn fft(a: &mut [Complex]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "the FFT needs a power-of-two length");
    bit_reverse(a);
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let w = Complex::new(angle.cos(), angle.sin());
        let mut start = 0;
        while start < n {
            let mut wk = Complex::new(1.0, 0.0);
            for j in 0..len / 2 {
                let u = a[start + j];
                let v = a[start + j + len / 2] * wk;
                a[start + j] = u + v;
                a[start + j + len / 2] = u - v;
                wk = wk * w;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// The inverse DFT with the `1/n` factor: `a_j = (1/n) sum_k A_k exp(+2 pi i j k / n)`.
pub fn ifft(a: &mut [Complex]) {
    for value in a.iter_mut() {
        value.im = -value.im;
    }
    fft(a);
    let inv = 1.0 / a.len() as f64;
    for value in a.iter_mut() {
        value.re *= inv;
        value.im = -value.im * inv;
    }
}

/// The integer wavenumbers in the FFT's order, `0, 1, ..., n/2-1, -n/2, ..., -1`.
pub fn wavenumbers(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            if j < n / 2 {
                j as f64
            } else {
                j as f64 - n as f64
            }
        })
        .collect()
}

/// A Gaussian pulse of standard deviation `sigma` centred at `center`, summed
/// over its periodic images so that it is periodic on `[0, 2pi)`.
pub fn gaussian(n: usize, sigma: f64, center: f64) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    (0..n)
        .map(|j| {
            let x = j as f64 * dx;
            let mut sum = 0.0;
            for m in -4..=4 {
                let d = x - center - 2.0 * PI * m as f64;
                sum += (-(d * d) / (2.0 * sigma * sigma)).exp();
            }
            sum
        })
        .collect()
}

/// The exact solution of Equation 6 for the periodic Gaussian: every image
/// translates by `c t` and widens to `sqrt(sigma^2 + 2 nu t)`.
pub fn exact(n: usize, sigma: f64, center: f64, c: f64, nu: f64, t: f64) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    let width = sigma * sigma + 2.0 * nu * t;
    let amplitude = sigma / width.sqrt();
    (0..n)
        .map(|j| {
            let x = j as f64 * dx;
            let mut sum = 0.0;
            for m in -4..=4 {
                let d = x - center - c * t - 2.0 * PI * m as f64;
                sum += (-(d * d) / (2.0 * width)).exp();
            }
            amplitude * sum
        })
        .collect()
}

/// Equation 6 on one periodic line.
pub struct Line {
    pub n: usize,
    pub c: f64,
    pub nu: f64,
}

impl Line {
    pub fn new(n: usize, c: f64, nu: f64) -> Self {
        Line { n, c, nu }
    }

    /// The spectrum of the Fourier rate function, Equation 9.  The Nyquist
    /// mode has no partner `+n/2`, so its first derivative is zero and it
    /// decays on the real axis.
    pub fn lambda(&self) -> Vec<Complex> {
        let nyquist = -(self.n as f64) / 2.0;
        wavenumbers(self.n)
            .into_iter()
            .map(|k| {
                let travel = if k == nyquist { 0.0 } else { -self.c * k };
                Complex::new(-self.nu * k * k, travel)
            })
            .collect()
    }

    /// The rate function with Fourier derivatives.
    pub fn rate_fourier(&self, u: &[f64], out: &mut [f64]) {
        let mut uh: Vec<Complex> = u.iter().map(|&value| Complex::new(value, 0.0)).collect();
        fft(&mut uh);
        for (value, lambda) in uh.iter_mut().zip(self.lambda()) {
            *value = *value * lambda;
        }
        ifft(&mut uh);
        for (o, value) in out.iter_mut().zip(&uh) {
            *o = value.re;
        }
    }

    /// The rate function with centred differences, Equation 8, wrapping.
    pub fn rate_fd(&self, u: &[f64], out: &mut [f64]) {
        let n = self.n;
        let dx = 2.0 * PI / n as f64;
        for j in 0..n {
            let up = u[(j + 1) % n];
            let down = u[(j + n - 1) % n];
            out[j] =
                -self.c * (up - down) / (2.0 * dx) + self.nu * (up - 2.0 * u[j] + down) / (dx * dx);
        }
    }
}

/// Advance `y` for `steps` steps of `h` and return `(t, y)`.
pub fn integrate(
    integrator: &dyn Integrator,
    rate: &mut Rate<'_>,
    y0: &[f64],
    h: f64,
    steps: usize,
) -> (f64, Vec<f64>) {
    let mut y = y0.to_vec();
    for _ in 0..steps {
        y = integrator.step(&y, h, rate);
    }
    (steps as f64 * h, y)
}

#[cfg(test)]
mod dump {
    //! The measurements the Part 1 plots are drawn from, taken by the crate's
    //! own steppers.  `scripts/line.py` runs this through `cargo test` and
    //! reads the one JSON line it prints.

    use super::*;
    use crate::integrator::{Euler, Rk2, Rk4};
    use serde_json::{Value, json};

    /// The growth factor `|R(z)|`, measured by advancing `y' = z y` one step
    /// of `h = 1` with `integrator`.
    fn growth(integrator: &dyn Integrator, z: Complex) -> f64 {
        let mut rate = |y: &[f64], out: &mut [f64]| {
            out[0] = z.re * y[0];
        };
        let y = integrator.step(&[1.0], 1.0, &mut rate);
        y[0].abs()
    }

    /// The measured growth map on the complex plane.
    fn growth_map() -> Value {
        let re_min = -5.0;
        let re_max = 1.0;
        let im_min = -4.0;
        let im_max = 4.0;
        let n_re = 241;
        let n_im = 241;
        let rk4 = Rk4::classical();
        let mut values = Vec::with_capacity(n_re * n_im);
        for i in 0..n_im {
            let im = im_min + (im_max - im_min) * i as f64 / (n_im - 1) as f64;
            for j in 0..n_re {
                let re = re_min + (re_max - re_min) * j as f64 / (n_re - 1) as f64;
                values.push(growth(&rk4, Complex::new(re, im)));
            }
        }
        json!({
            "re_min": re_min, "re_max": re_max, "im_min": im_min, "im_max": im_max,
            "n_re": n_re, "n_im": n_im, "growth": values,
        })
    }

    /// The line's modes `lambda_k h` at one step.
    fn modes(line: &Line, h: f64) -> Vec<f64> {
        line.lambda()
            .into_iter()
            .flat_map(|lambda| [lambda.re * h, lambda.im * h])
            .collect()
    }

    /// The pulse at `sigma = 0.35`, `n = 64`, `c = 1`, `nu = 0.05`, integrated
    /// by RK4 with Fourier derivatives to `t = 6`, sampled every step.
    fn pulse(step: f64) -> Value {
        let n = 64;
        let sigma = 0.35;
        let line = Line::new(n, 1.0, 0.05);
        let y0 = gaussian(n, sigma, PI / 2.0);
        let rk4 = Rk4::classical();
        let mut rate = |y: &[f64], out: &mut [f64]| line.rate_fourier(y, out);
        let mut y = y0;
        let mut t = 0.0;
        let mut times = vec![0.0];
        let mut rows: Vec<f64> = y.clone();
        while t < 6.0 - 1e-12 {
            y = rk4.step(&y, step, &mut rate);
            t += step;
            times.push(t);
            rows.extend_from_slice(&y);
        }
        json!({"t": times, "u": rows, "steps": times.len()})
    }

    /// Panel A: one lap of the `sigma = 0.25` pulse, three runs and the exact
    /// solution, with each run's maximum error.
    fn panel_a() -> Value {
        let n = 64;
        let c = 1.0;
        let nu = 0.002;
        let line = Line::new(n, c, nu);
        let y0 = gaussian(n, 0.25, PI / 2.0);
        let t_end = 2.0 * PI;

        let mut fourier = |y: &[f64], out: &mut [f64]| line.rate_fourier(y, out);
        let mut fd = |y: &[f64], out: &mut [f64]| line.rate_fd(y, out);
        let rk4 = Rk4::classical();
        let euler = Euler;

        let (t_rk4, y_rk4) = integrate(
            &rk4,
            &mut fourier,
            &y0,
            0.02,
            (t_end / 0.02).ceil() as usize,
        );
        let (_, y_fd) = integrate(&rk4, &mut fd, &y0, 0.02, (t_end / 0.02).ceil() as usize);
        let (_, y_euler) = integrate(
            &euler,
            &mut fourier,
            &y0,
            0.005,
            (t_end / 0.005).ceil() as usize,
        );
        let reference = exact(n, 0.25, PI / 2.0, c, nu, t_rk4);

        let error = |y: &[f64]| {
            y.iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        };
        json!({
            "t": t_rk4,
            "x": (0..n).map(|j| j as f64 * 2.0 * PI / n as f64).collect::<Vec<_>>(),
            "exact": reference,
            "rk4_fourier": y_rk4,
            "rk4_fd": y_fd,
            "euler_fourier": y_euler,
            "errors": {
                "rk4_fourier": error(&y_rk4),
                "rk4_fd": error(&y_fd),
                "euler_fourier": error(&y_euler),
            },
        })
    }

    /// Panel B: the error at `t = 1` against the step for the four steppers.
    fn panel_b() -> Value {
        let n = 64;
        let c = 1.0;
        let nu = 0.05;
        let line = Line::new(n, c, nu);
        let y0 = gaussian(n, 0.35, PI / 2.0);
        let steps = [0.02, 0.01, 0.005, 0.0025];
        let methods: [(&str, Box<dyn Integrator>); 4] = [
            ("euler", Box::new(Euler)),
            ("rk2", Box::new(Rk2)),
            ("rk4", Box::new(Rk4::classical())),
            (
                "rk4_equal",
                Box::new(Rk4 {
                    equal_weights: true,
                }),
            ),
        ];
        let reference = exact(n, 0.35, PI / 2.0, c, nu, 1.0);
        let mut series = serde_json::Map::new();
        for (name, integrator) in methods {
            let mut errors = Vec::new();
            for &h in &steps {
                let mut rate = |y: &[f64], out: &mut [f64]| line.rate_fourier(y, out);
                let (_, y) = integrate(
                    integrator.as_ref(),
                    &mut rate,
                    &y0,
                    h,
                    (1.0 / h).round() as usize,
                );
                errors.push(
                    y.iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max),
                );
            }
            series.insert(name.to_string(), json!(errors));
        }
        json!({"steps": steps, "errors": series})
    }

    #[test]
    fn line_dump() {
        let line = Line::new(64, 1.0, 0.05);
        let out = json!({
            "z": growth_map(),
            "modes": {
                "0.045": modes(&line, 0.045),
                "0.056": modes(&line, 0.056),
            },
            "pulses": {"0.045": pulse(0.045), "0.056": pulse(0.056)},
            "panel_a": panel_a(),
            "panel_b": panel_b(),
        });
        println!("{}", serde_json::to_string(&out).unwrap());
    }
}
