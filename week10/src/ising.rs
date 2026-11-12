pub fn site_tensor(k: f64) -> ([f64; 16], [[f64; 2]; 2], f64) {
    let ch = k.cosh().sqrt();
    let sh = k.sinh().sqrt();
    let w = [[ch, sh], [ch, -sh]];
    let mut a = [0.0; 16];
    for r in 0..2 {
        for u in 0..2 {
            for l in 0..2 {
                for d in 0..2 {
                    let mut value = 0.0;
                    for row in &w {
                        value += row[r] * row[u] * row[l] * row[d];
                    }
                    a[((r * 2 + u) * 2 + l) * 2 + d] = value;
                }
            }
        }
    }
    (a, w, 1.0 / ch)
}

pub fn a_index(r: usize, u: usize, l: usize, d: usize) -> usize {
    ((r * 2 + u) * 2 + l) * 2 + d
}

pub fn exact_log_z(k: f64, l: usize, periodic: bool) -> f64 {
    let n = l * l;
    let mut logs = Vec::with_capacity(1 << n);
    for bits in 0..(1usize << n) {
        let mut energy = 0.0;
        for y in 0..l {
            for x in 0..l {
                let s = spin(bits, x, y, l);
                if x + 1 < l || periodic {
                    let xr = (x + 1) % l;
                    energy += s * spin(bits, xr, y, l);
                }
                if y + 1 < l || periodic {
                    let yu = (y + 1) % l;
                    energy += s * spin(bits, x, yu, l);
                }
            }
        }
        logs.push(k * energy);
    }
    log_sum_exp(&logs)
}

pub fn exact_energy_variance(k: f64, l: usize, periodic: bool) -> f64 {
    let n = l * l;
    let mut energies = Vec::with_capacity(1 << n);
    let mut log_weights = Vec::with_capacity(1 << n);
    for bits in 0..(1usize << n) {
        let mut energy = 0.0;
        for y in 0..l {
            for x in 0..l {
                let s = spin(bits, x, y, l);
                if x + 1 < l || periodic {
                    let xr = (x + 1) % l;
                    energy -= s * spin(bits, xr, y, l);
                }
                if y + 1 < l || periodic {
                    let yu = (y + 1) % l;
                    energy -= s * spin(bits, x, yu, l);
                }
            }
        }
        energies.push(energy);
        log_weights.push(-k * energy);
    }
    let log_z = log_sum_exp(&log_weights);
    let z_inv = (-log_z).exp();
    let mut mean = 0.0;
    let mut mean_sq = 0.0;
    for (&e, &lw) in energies.iter().zip(&log_weights) {
        let p = lw.exp() * z_inv;
        mean += p * e;
        mean_sq += p * e * e;
    }
    mean_sq - mean * mean
}

pub fn exact_specific_heat(k: f64, l: usize, periodic: bool) -> f64 {
    k * k * exact_energy_variance(k, l, periodic) / (l * l) as f64
}

fn spin(bits: usize, x: usize, y: usize, l: usize) -> f64 {
    if (bits >> (y * l + x)) & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}
