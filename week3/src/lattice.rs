use rand::Rng;

/// A periodic square Ising lattice stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    l: usize,
    spins: Vec<i8>,
}

impl Lattice {
    pub fn all_up(l: usize) -> Self {
        assert!(l > 1, "lattice side must exceed one");
        Self {
            l,
            spins: vec![1; l * l],
        }
    }

    pub fn from_spins(l: usize, spins: Vec<i8>) -> Self {
        assert!(l > 1, "lattice side must exceed one");
        assert_eq!(spins.len(), l * l, "spin count must equal l squared");
        assert!(spins.iter().all(|&spin| spin == -1 || spin == 1));
        Self { l, spins }
    }

    pub fn random<R: Rng + ?Sized>(l: usize, rng: &mut R) -> Self {
        let spins = (0..l * l)
            .map(|_| if rng.random::<bool>() { 1 } else { -1 })
            .collect();
        Self::from_spins(l, spins)
    }

    pub fn side(&self) -> usize {
        self.l
    }

    pub fn site_count(&self) -> usize {
        self.spins.len()
    }

    pub fn spin(&self, site: usize) -> i8 {
        self.spins[site]
    }

    fn neighbour_sum(&self, site: usize) -> i32 {
        let row = site / self.l;
        let col = site % self.l;
        let up = ((row + self.l - 1) % self.l) * self.l + col;
        let down = ((row + 1) % self.l) * self.l + col;
        let left = row * self.l + (col + self.l - 1) % self.l;
        let right = row * self.l + (col + 1) % self.l;
        [up, down, left, right]
            .into_iter()
            .map(|index| i32::from(self.spins[index]))
            .sum()
    }

    pub fn delta_energy_at(&self, site: usize) -> i32 {
        2 * i32::from(self.spins[site]) * self.neighbour_sum(site)
    }

    pub fn flip(&mut self, site: usize) {
        self.spins[site] = -self.spins[site];
    }

    pub fn flipped(&self, site: usize) -> Self {
        let mut result = self.clone();
        result.flip(site);
        result
    }

    pub fn energy(&self) -> i32 {
        let mut energy = 0;
        for row in 0..self.l {
            for col in 0..self.l {
                let site = row * self.l + col;
                let right = row * self.l + (col + 1) % self.l;
                let down = ((row + 1) % self.l) * self.l + col;
                energy -= i32::from(self.spins[site])
                    * (i32::from(self.spins[right]) + i32::from(self.spins[down]));
            }
        }
        energy
    }

    pub fn magnetization(&self) -> f64 {
        let total: i32 = self.spins.iter().map(|&spin| i32::from(spin)).sum();
        f64::from(total) / self.site_count() as f64
    }

    pub fn render(&self) -> String {
        let mut output = String::with_capacity(self.site_count() + self.l - 1);
        for row in 0..self.l {
            if row > 0 {
                output.push('\n');
            }
            for col in 0..self.l {
                output.push(if self.spins[row * self.l + col] == 1 {
                    '█'
                } else {
                    '·'
                });
            }
        }
        output
    }
}

