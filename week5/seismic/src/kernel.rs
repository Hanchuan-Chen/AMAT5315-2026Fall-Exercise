//! Enzyme kernel: one damped acoustic timestep over a fixed grid.
//!
//! `build.rs` compiles this file alone with `-Zautodiff=Enable`; the main crate
//! never declares it as a module and reaches it only through the C ABI wrappers
//! below. The kernel is `no_std`, allocation-free and formatting-free.
//!
//! State `(prev, u) = (u^{n-1}, u^n)`; the update is
//! `out = (2u - (1 - dt*sigma)*prev + dt^2*(c^2*Lap(u) + factor*footprint)) / (1 + dt*sigma)`
//! with the five-point Laplacian, zero outer boundary and receivers sampled
//! outside the kernel.
#![no_std]
#![feature(autodiff)]

use core::autodiff::{autodiff_forward, autodiff_reverse};

unsafe extern "C" {
    fn abort() -> !;
}

#[inline(always)]
pub fn step(
    prev: &[f64],
    u: &[f64],
    c: &[f64],
    footprint: &[f64],
    sigma: &[f64],
    den: &[f64],
    nx: usize,
    nz: usize,
    invdx2: f64,
    dt: f64,
    factor: f64,
    out: &mut [f64],
) {
    let dt2 = dt * dt;
    let mut z = 1;
    while z + 1 < nz {
        let row = z * nx;
        let mut x = 1;
        while x + 1 < nx {
            let i = row + x;
            let lap = (u[i - 1] + u[i + 1] + u[i - nx] + u[i + nx] - 4.0 * u[i]) * invdx2;
            out[i] = (2.0 * u[i] - (1.0 - dt * sigma[i]) * prev[i]
                + dt2 * (c[i] * c[i] * lap + factor * footprint[i]))
                / den[i];
            x += 1;
        }
        z += 1;
    }
    let mut x = 0;
    while x < nx {
        out[x] = 0.0;
        out[(nz - 1) * nx + x] = 0.0;
        x += 1;
    }
    let mut z = 0;
    while z < nz {
        out[z * nx] = 0.0;
        out[z * nx + nx - 1] = 0.0;
        z += 1;
    }
}

#[autodiff_forward(step_forward, Dual, Dual, Dual, Const, Const, Const, Const, Const, Const, Const, Const, Dual)]
pub fn step_dual(
    prev: &[f64],
    u: &[f64],
    c: &[f64],
    footprint: &[f64],
    sigma: &[f64],
    den: &[f64],
    nx: usize,
    nz: usize,
    invdx2: f64,
    dt: f64,
    factor: f64,
    out: &mut [f64],
) {
    step(prev, u, c, footprint, sigma, den, nx, nz, invdx2, dt, factor, out)
}

#[autodiff_reverse(step_reverse, Duplicated, Duplicated, Duplicated, Const, Const, Const, Const, Const, Const, Const, Const, Duplicated)]
pub fn step_adjoint(
    prev: &[f64],
    u: &[f64],
    c: &[f64],
    footprint: &[f64],
    sigma: &[f64],
    den: &[f64],
    nx: usize,
    nz: usize,
    invdx2: f64,
    dt: f64,
    factor: f64,
    out: &mut [f64],
) {
    step(prev, u, c, footprint, sigma, den, nx, nz, invdx2, dt, factor, out)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_primal(
    prev: *const f64,
    u: *const f64,
    c: *const f64,
    footprint: *const f64,
    sigma: *const f64,
    den: *const f64,
    nx: usize,
    nz: usize,
    invdx2: f64,
    dt: f64,
    factor: f64,
    out: *mut f64,
) {
    let n = nx * nz;
    let prev = unsafe { core::slice::from_raw_parts(prev, n) };
    let u = unsafe { core::slice::from_raw_parts(u, n) };
    let c = unsafe { core::slice::from_raw_parts(c, n) };
    let footprint = unsafe { core::slice::from_raw_parts(footprint, n) };
    let sigma = unsafe { core::slice::from_raw_parts(sigma, n) };
    let den = unsafe { core::slice::from_raw_parts(den, n) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, n) };
    step(prev, u, c, footprint, sigma, den, nx, nz, invdx2, dt, factor, out);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_forward(
    prev: *const f64,
    dprev: *const f64,
    u: *const f64,
    du: *const f64,
    c: *const f64,
    dc: *const f64,
    footprint: *const f64,
    sigma: *const f64,
    den: *const f64,
    nx: usize,
    nz: usize,
    invdx2: f64,
    dt: f64,
    factor: f64,
    out: *mut f64,
    dout: *mut f64,
) {
    let n = nx * nz;
    let prev = unsafe { core::slice::from_raw_parts(prev, n) };
    let dprev = unsafe { core::slice::from_raw_parts(dprev, n) };
    let u = unsafe { core::slice::from_raw_parts(u, n) };
    let du = unsafe { core::slice::from_raw_parts(du, n) };
    let c = unsafe { core::slice::from_raw_parts(c, n) };
    let dc = unsafe { core::slice::from_raw_parts(dc, n) };
    let footprint = unsafe { core::slice::from_raw_parts(footprint, n) };
    let sigma = unsafe { core::slice::from_raw_parts(sigma, n) };
    let den = unsafe { core::slice::from_raw_parts(den, n) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, n) };
    let dout = unsafe { core::slice::from_raw_parts_mut(dout, n) };
    step_forward(prev, dprev, u, du, c, dc, footprint, sigma, den, nx, nz, invdx2, dt, factor, out, dout);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_reverse(
    prev: *const f64,
    dprev: *mut f64,
    u: *const f64,
    du: *mut f64,
    c: *const f64,
    dc: *mut f64,
    footprint: *const f64,
    sigma: *const f64,
    den: *const f64,
    nx: usize,
    nz: usize,
    invdx2: f64,
    dt: f64,
    factor: f64,
    out: *mut f64,
    dout: *mut f64,
) {
    let n = nx * nz;
    let prev = unsafe { core::slice::from_raw_parts(prev, n) };
    let dprev = unsafe { core::slice::from_raw_parts_mut(dprev, n) };
    let u = unsafe { core::slice::from_raw_parts(u, n) };
    let du = unsafe { core::slice::from_raw_parts_mut(du, n) };
    let c = unsafe { core::slice::from_raw_parts(c, n) };
    let dc = unsafe { core::slice::from_raw_parts_mut(dc, n) };
    let footprint = unsafe { core::slice::from_raw_parts(footprint, n) };
    let sigma = unsafe { core::slice::from_raw_parts(sigma, n) };
    let den = unsafe { core::slice::from_raw_parts(den, n) };
    let out = unsafe { core::slice::from_raw_parts_mut(out, n) };
    let dout = unsafe { core::slice::from_raw_parts_mut(dout, n) };
    step_reverse(prev, dprev, u, du, c, dc, footprint, sigma, den, nx, nz, invdx2, dt, factor, out, dout);
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
