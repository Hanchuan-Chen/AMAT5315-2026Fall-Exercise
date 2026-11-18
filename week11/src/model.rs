use rand::Rng;
use serde_json::Value;
use std::fs;

#[derive(Clone, Debug)]
pub struct Factor {
    pub vars: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Network {
    pub nvars: usize,
    pub factors: Vec<Factor>,
    pub edges: Vec<(usize, usize)>,
}

impl Network {
    pub fn from_spin_model(
        nvars: usize,
        edges: Vec<(usize, usize)>,
        couplings: &[f64],
        fields: &[f64],
        energy_sign: f64,
    ) -> Self {
        assert_eq!(edges.len(), couplings.len());
        assert_eq!(fields.len(), nvars);
        let mut factors = Vec::with_capacity(edges.len() + nvars);
        for (&(i, j), &j_ij) in edges.iter().zip(couplings) {
            factors.push(Factor {
                vars: vec![i, j],
                values: vec![
                    energy_sign * j_ij,
                    -energy_sign * j_ij,
                    -energy_sign * j_ij,
                    energy_sign * j_ij,
                ],
            });
        }
        for (i, &h_i) in fields.iter().enumerate() {
            if h_i == 0.0 {
                continue;
            }
            factors.push(Factor {
                vars: vec![i],
                values: vec![-h_i, h_i],
            });
        }
        Self {
            nvars,
            factors,
            edges,
        }
    }

    pub fn from_bundle(path: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        let root: Value = serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))?;
        let data = &root["target"]["data"];
        let n = data["graph"]["num_vertices"]
            .as_u64()
            .ok_or("bundle lacks target.data.graph.num_vertices")? as usize;
        let edges: Vec<(usize, usize)> = data["graph"]["edges"]
            .as_array()
            .ok_or("bundle lacks target.data.graph.edges")?
            .iter()
            .map(|e| {
                let pair = e.as_array().ok_or("edge is not an array")?;
                Ok((
                    pair[0].as_u64().ok_or("bad edge endpoint")? as usize,
                    pair[1].as_u64().ok_or("bad edge endpoint")? as usize,
                ))
            })
            .collect::<Result<_, &str>>()
            .map_err(str::to_string)?;
        let couplings: Vec<f64> = data["couplings"]
            .as_array()
            .ok_or("bundle lacks target.data.couplings")?
            .iter()
            .map(|v| v.as_f64().ok_or("bad coupling"))
            .collect::<Result<_, _>>()
            .map_err(str::to_string)?;
        let fields: Vec<f64> = data["fields"]
            .as_array()
            .ok_or("bundle lacks target.data.fields")?
            .iter()
            .map(|v| v.as_f64().ok_or("bad field"))
            .collect::<Result<_, _>>()
            .map_err(str::to_string)?;
        Ok(Self::from_spin_model(n, edges, &couplings, &fields, 1.0))
    }

    pub fn open_lattice(l: usize, p_af: f64, rng: &mut impl Rng) -> Self {
        let n = l * l;
        let mut edges = Vec::with_capacity(2 * l * (l - 1));
        let mut couplings = Vec::with_capacity(2 * l * (l - 1));
        for r in 0..l {
            for c in 0..l {
                let i = r * l + c;
                if c + 1 < l {
                    edges.push((i, i + 1));
                    couplings.push(if rng.random::<f64>() < p_af {
                        -1.0
                    } else {
                        1.0
                    });
                }
                if r + 1 < l {
                    edges.push((i, i + l));
                    couplings.push(if rng.random::<f64>() < p_af {
                        -1.0
                    } else {
                        1.0
                    });
                }
            }
        }
        Self::from_spin_model(n, edges, &couplings, &vec![0.0; n], -1.0)
    }

    pub fn energy(&self, bits: &[u8]) -> f64 {
        self.factors
            .iter()
            .map(|f| {
                let key = f
                    .vars
                    .iter()
                    .fold(0, |acc, &v| (acc << 1) | bits[v] as usize);
                f.values[key]
            })
            .sum()
    }
}

fn carry_values(vars: Vec<usize>, fixed_in: Option<u8>, fixed_out: Option<u8>) -> Vec<f64> {
    let mut values = Vec::with_capacity(1 << vars.len());
    for key in 0..(1usize << vars.len()) {
        let co = key & 1;
        let x = key >> (vars.len() - 2) & 1;
        let ci = key >> (vars.len() - 1) & 1;
        let allowed = matches!((ci, x, co), (0, 0, 0) | (0, 1, 1) | (1, 0, 1));
        if !allowed
            || fixed_in.is_some_and(|v| ci != v as usize)
            || fixed_out.is_some_and(|v| co != v as usize)
        {
            values.push(0.0);
        } else {
            values.push(1.0);
        }
    }
    values
}

pub fn nqueens_network(n: usize) -> Network {
    let bits = n * n;
    let mut next = bits;
    let mut factors = Vec::new();

    let mut add_line = |cells: &[(usize, usize)], require_one: bool| {
        let mut carries = Vec::new();
        for _ in 0..=cells.len() {
            carries.push(next);
            next += 1;
        }
        for (pos, &(r, c)) in cells.iter().enumerate() {
            let x = r * n + c;
            let fixed_in = (pos == 0).then_some(0);
            let fixed_out = (require_one && pos + 1 == cells.len()).then_some(1);
            let vars = vec![carries[pos], x, carries[pos + 1]];
            factors.push(Factor {
                values: carry_values(vars.clone(), fixed_in, fixed_out),
                vars,
            });
        }
    };

    for r in 0..n {
        let cells: Vec<_> = (0..n).map(|c| (r, c)).collect();
        add_line(&cells, true);
    }
    for c in 0..n {
        let cells: Vec<_> = (0..n).map(|r| (r, c)).collect();
        add_line(&cells, true);
    }
    for d in 0..(2 * n - 1) {
        let cells: Vec<_> = (0..n)
            .filter_map(|r| {
                let c = r as isize + d as isize - (n as isize - 1);
                (0..n as isize).contains(&c).then_some((r, c as usize))
            })
            .collect();
        add_line(&cells, false);
    }
    for s in 0..(2 * n - 1) {
        let cells: Vec<_> = (0..n)
            .filter_map(|r| {
                let c = s as isize - r as isize;
                (0..n as isize).contains(&c).then_some((r, c as usize))
            })
            .collect();
        add_line(&cells, false);
    }

    Network {
        nvars: next,
        factors,
        edges: Vec::new(),
    }
}
