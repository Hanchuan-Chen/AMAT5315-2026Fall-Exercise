//! Safe wrappers around the isolated Enzyme kernel.
//!
//! Every wrapper checks its slice lengths before the FFI call; no pointer
//! escapes one call.

unsafe extern "C" {
    fn kernel_primal(
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
    );
    fn kernel_forward(
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
    );
    fn kernel_reverse(
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
    );
}

/// Fixed grid metadata shared by every kernel call.
#[derive(Clone, Copy)]
pub struct Grid {
    pub nx: usize,
    pub nz: usize,
    pub invdx2: f64,
    pub dt: f64,
}

impl Grid {
    pub fn n(&self) -> usize {
        self.nx * self.nz
    }
}

#[allow(clippy::too_many_arguments)]
pub fn primal(
    grid: &Grid,
    prev: &[f64],
    u: &[f64],
    c: &[f64],
    footprint: &[f64],
    sigma: &[f64],
    den: &[f64],
    factor: f64,
    out: &mut [f64],
) {
    let n = grid.n();
    assert_eq!(prev.len(), n);
    assert_eq!(u.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(footprint.len(), n);
    assert_eq!(sigma.len(), n);
    assert_eq!(den.len(), n);
    assert_eq!(out.len(), n);
    unsafe {
        kernel_primal(
            prev.as_ptr(),
            u.as_ptr(),
            c.as_ptr(),
            footprint.as_ptr(),
            sigma.as_ptr(),
            den.as_ptr(),
            grid.nx,
            grid.nz,
            grid.invdx2,
            grid.dt,
            factor,
            out.as_mut_ptr(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward(
    grid: &Grid,
    prev: &[f64],
    dprev: &[f64],
    u: &[f64],
    du: &[f64],
    c: &[f64],
    dc: &[f64],
    footprint: &[f64],
    sigma: &[f64],
    den: &[f64],
    factor: f64,
    out: &mut [f64],
    dout: &mut [f64],
) {
    let n = grid.n();
    assert_eq!(prev.len(), n);
    assert_eq!(dprev.len(), n);
    assert_eq!(u.len(), n);
    assert_eq!(du.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(dc.len(), n);
    assert_eq!(footprint.len(), n);
    assert_eq!(sigma.len(), n);
    assert_eq!(den.len(), n);
    assert_eq!(out.len(), n);
    assert_eq!(dout.len(), n);
    unsafe {
        kernel_forward(
            prev.as_ptr(),
            dprev.as_ptr(),
            u.as_ptr(),
            du.as_ptr(),
            c.as_ptr(),
            dc.as_ptr(),
            footprint.as_ptr(),
            sigma.as_ptr(),
            den.as_ptr(),
            grid.nx,
            grid.nz,
            grid.invdx2,
            grid.dt,
            factor,
            out.as_mut_ptr(),
            dout.as_mut_ptr(),
        );
    }
}

/// One timestep VJP: the output seed `dout` is consumed (Enzyme zeroes the
/// output shadow), and each adjoint buffer accumulates its gradient.
#[allow(clippy::too_many_arguments)]
pub fn reverse(
    grid: &Grid,
    prev: &[f64],
    dprev: &mut [f64],
    u: &[f64],
    du: &mut [f64],
    c: &[f64],
    dc: &mut [f64],
    footprint: &[f64],
    sigma: &[f64],
    den: &[f64],
    factor: f64,
    out: &mut [f64],
    dout: &mut [f64],
) {
    let n = grid.n();
    assert_eq!(prev.len(), n);
    assert_eq!(dprev.len(), n);
    assert_eq!(u.len(), n);
    assert_eq!(du.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(dc.len(), n);
    assert_eq!(footprint.len(), n);
    assert_eq!(sigma.len(), n);
    assert_eq!(den.len(), n);
    assert_eq!(out.len(), n);
    assert_eq!(dout.len(), n);
    unsafe {
        kernel_reverse(
            prev.as_ptr(),
            dprev.as_mut_ptr(),
            u.as_ptr(),
            du.as_mut_ptr(),
            c.as_ptr(),
            dc.as_mut_ptr(),
            footprint.as_ptr(),
            sigma.as_ptr(),
            den.as_ptr(),
            grid.nx,
            grid.nz,
            grid.invdx2,
            grid.dt,
            factor,
            out.as_mut_ptr(),
            dout.as_mut_ptr(),
        );
    }
}
