use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::Lattice;

#[derive(Clone, Debug)]
pub struct AcceptanceTable([f64; 5]);

impl AcceptanceTable {
    pub fn new(temperature: f64) -> Self {
        assert!(temperature.is_finite() && temperature > 0.0);
        Self([
            1.0,
            1.0,
            1.0,
            (-4.0 / temperature).exp(),
            (-8.0 / temperature).exp(),
        ])
    }

    fn probability(&self, delta_energy: i32) -> f64 {
        debug_assert!(matches!(delta_energy, -8 | -4 | 0 | 4 | 8));
        self.0[((delta_energy + 8) / 4) as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepStats {
    pub proposals: usize,
    pub accepted: usize,
}

pub fn sweep<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    table: &AcceptanceTable,
    rng: &mut R,
) -> SweepStats {
    let proposals = lattice.site_count();
    let mut accepted = 0;
    for _ in 0..proposals {
        let site = rng.random_range(0..lattice.site_count());
        let probability = table.probability(lattice.delta_energy_at(site));
        if probability >= 1.0 || rng.random::<f64>() < probability {
            lattice.flip(site);
            accepted += 1;
        }
    }
    SweepStats {
        proposals,
        accepted,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelaxConfig {
    pub l: usize,
    pub temperature: f64,
    pub equilibration_sweeps: usize,
    pub measurement_sweeps: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct RelaxResult {
    pub config: RelaxConfig,
    pub mean_abs_m: f64,
    pub acceptance_fraction: f64,
    pub lattice: Lattice,
}

impl RelaxResult {
    pub fn render(&self) -> String {
        format!(
            "L={} T={:.6} sweeps={} measure={} mean_abs_m={:.6} accept={:.6}\n{}\n",
            self.config.l,
            self.config.temperature,
            self.config.equilibration_sweeps,
            self.config.measurement_sweeps,
            self.mean_abs_m,
            self.acceptance_fraction,
            self.lattice.render()
        )
    }
}

pub fn relax(config: RelaxConfig) -> RelaxResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l);
    let table = AcceptanceTable::new(config.temperature);
    let mut totals = SweepStats::default();

    for _ in 0..config.equilibration_sweeps {
        let stats = sweep(&mut lattice, &table, &mut rng);
        totals.proposals += stats.proposals;
        totals.accepted += stats.accepted;
    }

    let mut abs_m_sum = 0.0;
    for _ in 0..config.measurement_sweeps {
        let stats = sweep(&mut lattice, &table, &mut rng);
        totals.proposals += stats.proposals;
        totals.accepted += stats.accepted;
        abs_m_sum += lattice.magnetization().abs();
    }

    RelaxResult {
        config,
        mean_abs_m: abs_m_sum / config.measurement_sweeps as f64,
        acceptance_fraction: totals.accepted as f64 / totals.proposals as f64,
        lattice,
    }
}
