//! The periodic `l x l` lattice of `+1`/`-1` spins.

/// A square `l x l` lattice of spins, stored row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    l: usize,
    spins: Vec<i8>,
}

impl Lattice {
    /// A lattice with every spin up.
    pub fn all_up(l: usize) -> Self {
        todo!()
    }

    /// A lattice from an explicit row-major spin list.
    pub fn from_spins(l: usize, spins: Vec<i8>) -> Result<Self, String> {
        todo!()
    }

    pub fn l(&self) -> usize {
        todo!()
    }

    pub fn spins(&self) -> &[i8] {
        todo!()
    }

    /// Row-major index of `(row, col)`; both coordinates are reduced modulo `l`.
    pub fn index(&self, row: usize, col: usize) -> usize {
        todo!()
    }

    pub fn spin(&self, row: usize, col: usize) -> i8 {
        todo!()
    }

    /// Sum of the four wrapped neighbours of `(row, col)`.
    pub fn neighbor_sum(&self, row: usize, col: usize) -> i32 {
        todo!()
    }

    /// Energy change of flipping `(row, col)`, one of `-8, -4, 0, 4, 8`.
    pub fn delta_energy(&self, row: usize, col: usize) -> i32 {
        todo!()
    }

    /// Total energy `-sum_<ij> s_i s_j` over the `2 l^2` bonds.
    pub fn energy(&self) -> i64 {
        todo!()
    }

    pub fn energy_per_site(&self) -> f64 {
        todo!()
    }

    /// Signed mean spin.
    pub fn magnetization(&self) -> f64 {
        todo!()
    }

    pub fn abs_magnetization(&self) -> f64 {
        todo!()
    }

    pub fn flip(&mut self, row: usize, col: usize) {
        todo!()
    }
}
