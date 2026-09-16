//! The periodic `l x l` lattice of `+1`/`-1` spins.

/// A square `l x l` lattice of spins, stored row-major.
///
/// Every coordinate is reduced modulo `l`, so the four edges of the square are
/// glued to the opposite ones and no site sits on a boundary. For `l = 2` the
/// "up" and "down" neighbours of a site are the same site and the bond sum
/// counts it twice, which is what the `2 l^2` bonds of that torus need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    l: usize,
    spins: Vec<i8>,
}

impl Lattice {
    /// A lattice with every spin up.
    ///
    /// # Panics
    ///
    /// Panics when `l < 2`; use [`Lattice::from_spins`] for a fallible build.
    pub fn all_up(l: usize) -> Self {
        Self::from_spins(l, vec![1; l * l]).expect("lattice side is at least 2")
    }

    /// A lattice from an explicit row-major spin list.
    pub fn from_spins(l: usize, spins: Vec<i8>) -> Result<Self, String> {
        if l < 2 {
            return Err(format!("lattice side must be at least 2, got {l}"));
        }
        if spins.len() != l * l {
            return Err(format!(
                "expected {} spins for l = {l}, got {}",
                l * l,
                spins.len()
            ));
        }
        if let Some(bad) = spins.iter().find(|spin| **spin != 1 && **spin != -1) {
            return Err(format!("spins must be +1 or -1, got {bad}"));
        }
        Ok(Self { l, spins })
    }

    pub fn l(&self) -> usize {
        self.l
    }

    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// Row-major index of `(row, col)`; both coordinates are reduced modulo `l`.
    pub fn index(&self, row: usize, col: usize) -> usize {
        (row % self.l) * self.l + (col % self.l)
    }

    pub fn spin(&self, row: usize, col: usize) -> i8 {
        self.spins[self.index(row, col)]
    }

    /// Sum of the four wrapped neighbours of `(row, col)`.
    pub fn neighbor_sum(&self, row: usize, col: usize) -> i32 {
        let l = self.l;
        let up = self.spin((row + l - 1) % l, col) as i32;
        let down = self.spin((row + 1) % l, col) as i32;
        let left = self.spin(row, (col + l - 1) % l) as i32;
        let right = self.spin(row, (col + 1) % l) as i32;
        up + down + left + right
    }

    /// Energy change of flipping `(row, col)`, one of `-8, -4, 0, 4, 8`.
    ///
    /// Flipping a site replaces each of its four bonds `-s_i s_j` by `+s_i s_j`,
    /// so `dE = 2 s_i (s_1 + s_2 + s_3 + s_4)`.
    pub fn delta_energy(&self, row: usize, col: usize) -> i32 {
        2 * self.spin(row, col) as i32 * self.neighbor_sum(row, col)
    }

    /// Total energy `-sum_<ij> s_i s_j` over the `2 l^2` bonds.
    pub fn energy(&self) -> i64 {
        let mut energy = 0i64;
        for row in 0..self.l {
            for col in 0..self.l {
                let here = self.spins[row * self.l + col] as i64;
                let right = self.spins[row * self.l + (col + 1) % self.l] as i64;
                let down = self.spins[((row + 1) % self.l) * self.l + col] as i64;
                energy -= here * right;
                energy -= here * down;
            }
        }
        energy
    }

    pub fn energy_per_site(&self) -> f64 {
        self.energy() as f64 / (self.l * self.l) as f64
    }

    /// Signed mean spin.
    pub fn magnetization(&self) -> f64 {
        self.spins.iter().map(|spin| *spin as i64).sum::<i64>() as f64 / (self.l * self.l) as f64
    }

    pub fn abs_magnetization(&self) -> f64 {
        self.magnetization().abs()
    }

    pub fn flip(&mut self, row: usize, col: usize) {
        let index = self.index(row, col);
        self.spins[index] = -self.spins[index];
    }
}
