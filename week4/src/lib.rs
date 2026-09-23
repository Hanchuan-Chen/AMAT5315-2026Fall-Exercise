//! Week 4: the advection-diffusion line and the two-dimensional flow solver.
//!
//! `integrator` holds the three explicit steppers of Part 1, `line` the
//! periodic line they are tested on, and `solver` the pseudospectral
//! vorticity equation that the binaries `field` and `fluid` drive.

pub mod integrator;
pub mod line;
pub mod solver;

pub use integrator::{Euler, Integrator, Method, Rate, Rk2, Rk4};
pub use solver::Solver;
