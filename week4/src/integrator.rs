//! The three explicit integrators of Part 1, behind one interface.
//!
//! The sheet asks for forward Euler, the explicit midpoint rule and classical
//! fourth-order Runge-Kutta behind one `Integrator` trait: each advances a
//! state vector by one step of a given size from a given rate function.  The
//! `Rk4::equal_weights` variant is the sheet's control experiment, the RK4
//! stage weights flattened to 1,1,1,1.

/// A rate function: `f` writes `dy/dt` at `y` into its second argument.  The
/// week-4 equations are autonomous, so time never enters.
pub type Rate<'a> = dyn FnMut(&[f64], &mut [f64]) + 'a;

/// One explicit time integrator.
pub trait Integrator {
    /// The name the `--method` flag and the plots use.
    fn name(&self) -> &'static str;
    /// Advance `y` by one step of size `h`, evaluating the rate function.
    fn step(&self, y: &[f64], h: f64, f: &mut Rate<'_>) -> Vec<f64>;
}

/// Forward Euler, Equation 10.
pub struct Euler;

impl Integrator for Euler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn step(&self, y: &[f64], h: f64, f: &mut Rate<'_>) -> Vec<f64> {
        let mut k1 = vec![0.0; y.len()];
        f(y, &mut k1);
        y.iter().zip(&k1).map(|(y, k)| y + h * k).collect()
    }
}

/// The explicit midpoint rule, Equation 10.
pub struct Rk2;

impl Integrator for Rk2 {
    fn name(&self) -> &'static str {
        "rk2"
    }

    fn step(&self, y: &[f64], h: f64, f: &mut Rate<'_>) -> Vec<f64> {
        let mut k1 = vec![0.0; y.len()];
        let mut trial = vec![0.0; y.len()];
        let mut k2 = vec![0.0; y.len()];
        f(y, &mut k1);
        for i in 0..y.len() {
            trial[i] = y[i] + 0.5 * h * k1[i];
        }
        f(&trial, &mut k2);
        y.iter().zip(&k2).map(|(y, k)| y + h * k).collect()
    }
}

/// Classical fourth-order Runge-Kutta, Equation 10.
///
/// `equal_weights` replaces the weights 1,2,2,1 by 1,1,1,1: the method still
/// converges, but only at second order, which is the signature the sheet's
/// Part 1 check looks for.
pub struct Rk4 {
    pub equal_weights: bool,
}

impl Rk4 {
    /// The classical 1,2,2,1 weights.
    pub fn classical() -> Self {
        Rk4 {
            equal_weights: false,
        }
    }
}

impl Integrator for Rk4 {
    fn name(&self) -> &'static str {
        if self.equal_weights {
            "rk4-equal"
        } else {
            "rk4"
        }
    }

    fn step(&self, y: &[f64], h: f64, f: &mut Rate<'_>) -> Vec<f64> {
        let m = y.len();
        let mut k1 = vec![0.0; m];
        let mut k2 = vec![0.0; m];
        let mut k3 = vec![0.0; m];
        let mut k4 = vec![0.0; m];
        let mut trial = vec![0.0; m];
        f(y, &mut k1);
        for i in 0..m {
            trial[i] = y[i] + 0.5 * h * k1[i];
        }
        f(&trial, &mut k2);
        for i in 0..m {
            trial[i] = y[i] + 0.5 * h * k2[i];
        }
        f(&trial, &mut k3);
        for i in 0..m {
            trial[i] = y[i] + h * k3[i];
        }
        f(&trial, &mut k4);
        let w = if self.equal_weights {
            [0.25, 0.25, 0.25, 0.25]
        } else {
            [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
        };
        (0..m)
            .map(|i| y[i] + h * (w[0] * k1[i] + w[1] * k2[i] + w[2] * k3[i] + w[3] * k4[i]))
            .collect()
    }
}

/// The integrators `fluid --method` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Euler,
    Rk2,
    Rk4,
}

impl Method {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "euler" => Ok(Method::Euler),
            "rk2" => Ok(Method::Rk2),
            "rk4" => Ok(Method::Rk4),
            other => Err(format!(
                "unknown method {other:?}; expected euler, rk2 or rk4"
            )),
        }
    }

    pub fn integrator(&self) -> Box<dyn Integrator> {
        match self {
            Method::Euler => Box::new(Euler),
            Method::Rk2 => Box::new(Rk2),
            Method::Rk4 => Box::new(Rk4::classical()),
        }
    }
}
