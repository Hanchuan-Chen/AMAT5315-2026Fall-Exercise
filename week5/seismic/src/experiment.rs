//! Parse and hold one seismic experiment file.

use serde_json::{Map, Value};
use std::path::Path;

pub struct Experiment {
    pub path: String,
    pub raw: Map<String, Value>,
    pub name: String,
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub sponge_width: f64,
    pub sponge_strength: f64,
    pub background: Vec<f64>,
    pub perturbation: Vec<f64>,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[i64; 2]>,
    pub initial_state: Option<[Vec<f64>; 2]>,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
}

fn number(value: &Value, key: &str) -> f64 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("experiment field {key} must be a number"))
}

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("experiment is missing {key}"))
}

fn grid(value: &Value, key: &str, nx: usize, nz: usize) -> Vec<f64> {
    let rows = field(value, key)
        .as_array()
        .unwrap_or_else(|| panic!("experiment field {key} must be a [z][x] array"));
    assert_eq!(rows.len(), nz, "{key}: expected {nz} rows");
    let mut out = Vec::with_capacity(nx * nz);
    for row in rows {
        let row = row.as_array().unwrap_or_else(|| panic!("{key}: row must be an array"));
        assert_eq!(row.len(), nx, "{key}: expected {nx} columns");
        for cell in row {
            out.push(cell.as_f64().expect("array entry must be a number"));
        }
    }
    out
}

impl Experiment {
    pub fn load(path: &Path) -> Experiment {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let value: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display()));
        let raw = value.as_object().expect("experiment must be a JSON object").clone();
        let value = Value::Object(raw.clone());
        let nx = number(&value, "nx") as usize;
        let nz = number(&value, "nz") as usize;
        let shots = field(&value, "shots")
            .as_array()
            .expect("shots must be an array")
            .iter()
            .map(|s| {
                let s = s.as_array().expect("each shot must be [x, z]");
                [s[0].as_f64().unwrap(), s[1].as_f64().unwrap()]
            })
            .collect();
        let receivers = field(&value, "receivers")
            .as_array()
            .expect("receivers must be an array")
            .iter()
            .map(|r| {
                let r = r.as_array().expect("each receiver must be [x, z]");
                [r[0].as_f64().unwrap() as i64, r[1].as_f64().unwrap() as i64]
            })
            .collect();
        let initial_state = value.get("initial_state").map(|s| {
            let s = s.as_array().expect("initial_state must be two flattened fields");
            let flatten = |v: &Value| -> Vec<f64> {
                v.as_array()
                    .expect("initial_state field must be an array")
                    .iter()
                    .map(|cell| cell.as_f64().unwrap())
                    .collect()
            };
            [flatten(&s[0]), flatten(&s[1])]
        });
        Experiment {
            path: path.to_string_lossy().into_owned(),
            name: value
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("experiment")
                .to_string(),
            nx,
            nz,
            dx: number(&value, "dx"),
            dt: number(&value, "dt"),
            steps: number(&value, "steps") as usize,
            source_frequency: number(&value, "source_frequency"),
            source_peak_time: number(&value, "source_peak_time"),
            source_amplitude: number(&value, "source_amplitude"),
            sponge_width: number(&value, "sponge_width"),
            sponge_strength: number(&value, "sponge_strength"),
            background: grid(&value, "background", nx, nz),
            perturbation: grid(&value, "perturbation", nx, nz),
            shots,
            receivers,
            initial_state,
            length_unit_m: number(&value, "length_unit_m"),
            time_unit_s: number(&value, "time_unit_s"),
            raw,
        }
    }

    /// Every field except the physics arrays, as the design file defines
    /// `run.json`'s `experiment` block.
    pub fn compact(&self) -> Map<String, Value> {
        const KEYS: [&str; 17] = [
            "schema",
            "name",
            "nx",
            "nz",
            "dx",
            "dt",
            "steps",
            "source_frequency",
            "source_peak_time",
            "source_amplitude",
            "sponge_width",
            "sponge_strength",
            "shots",
            "receivers",
            "length_unit_m",
            "time_unit_s",
            "provenance",
        ];
        let mut out = Map::new();
        for key in KEYS {
            if let Some(value) = self.raw.get(key) {
                out.insert(key.to_string(), value.clone());
            }
        }
        out
    }

    /// The Ricker pulse evaluated at the start of step `step`.
    pub fn ricker(&self, step: usize) -> f64 {
        let a = std::f64::consts::PI
            * self.source_frequency
            * (step as f64 * self.dt - self.source_peak_time);
        (1.0 - 2.0 * a * a) * (-a * a).exp()
    }
}
