use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

pub fn susceptibility(l: usize, temperature: f64, magnetizations: &[f64]) -> f64 {
    let n = magnetizations.len() as f64;
    let mean_abs = magnetizations.iter().map(|value| value.abs()).sum::<f64>() / n;
    let mean_square = magnetizations.iter().map(|value| value * value).sum::<f64>() / n;
    (l * l) as f64 * (mean_square - mean_abs * mean_abs) / temperature
}

fn solve_three(mut matrix: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for column in 0..3 {
        let pivot = (column..3).max_by(|&a, &b| {
            matrix[a][column]
                .abs()
                .total_cmp(&matrix[b][column].abs())
        })?;
        if matrix[pivot][column].abs() < 1e-14 {
            return None;
        }
        matrix.swap(column, pivot);
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column] / matrix[column][column];
            for entry in column..4 {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
        }
    }
    Some([
        matrix[0][3] / matrix[0][0],
        matrix[1][3] / matrix[1][1],
        matrix[2][3] / matrix[2][2],
    ])
}

pub fn quadratic_peak(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 3 {
        return None;
    }
    let mut powers = [0.0; 5];
    let mut rhs = [0.0; 3];
    for &(x, y) in points {
        let mut value = 1.0;
        for power in &mut powers {
            *power += value;
            value *= x;
        }
        rhs[0] += x * x * y;
        rhs[1] += x * y;
        rhs[2] += y;
    }
    let [a, b, _] = solve_three([
        [powers[4], powers[3], powers[2], rhs[0]],
        [powers[3], powers[2], powers[1], rhs[1]],
        [powers[2], powers[1], powers[0], rhs[2]],
    ])?;
    (a < 0.0).then_some(-b / (2.0 * a))
}

pub fn critical_temperature(peak_32: f64, peak_64: f64) -> f64 {
    2.0 * peak_64 - peak_32
}

#[derive(Clone, Debug)]
pub struct ObservablePoint {
    pub temperature: f64,
    pub mean_abs_m: f64,
    pub chi: f64,
    pub magnetizations: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AnalysisSummary {
    pub by_size: BTreeMap<usize, Vec<ObservablePoint>>,
    pub peaks: BTreeMap<usize, f64>,
    pub critical_temperature: Option<f64>,
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct SeriesRow {
    L: usize,
    T: f64,
    M: f64,
}

pub fn load_analysis(directory: &Path) -> Result<AnalysisSummary, Box<dyn Error>> {
    let reader = BufReader::new(File::open(directory.join("series.jsonl"))?);
    let mut grouped: BTreeMap<usize, BTreeMap<i64, (f64, Vec<f64>)>> = BTreeMap::new();
    for line in reader.lines() {
        let row: SeriesRow = serde_json::from_str(&line?)?;
        let key = (row.T * 1_000_000.0).round() as i64;
        grouped
            .entry(row.L)
            .or_default()
            .entry(key)
            .or_insert_with(|| (row.T, Vec::new()))
            .1
            .push(row.M);
    }

    let mut by_size = BTreeMap::new();
    let mut peaks = BTreeMap::new();
    for (l, temperatures) in grouped {
        let points: Vec<_> = temperatures
            .into_values()
            .map(|(temperature, magnetizations)| {
                let mean_abs_m = magnetizations.iter().map(|value| value.abs()).sum::<f64>()
                    / magnetizations.len() as f64;
                let chi = susceptibility(l, temperature, &magnetizations);
                ObservablePoint {
                    temperature,
                    mean_abs_m,
                    chi,
                    magnetizations,
                }
            })
            .collect();
        if points.len() >= 5 {
            let maximum = points
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.chi.total_cmp(&b.chi))
                .map(|(index, _)| index)
                .unwrap();
            let start = maximum.saturating_sub(2).min(points.len() - 5);
            let fit_points: Vec<_> = points[start..start + 5]
                .iter()
                .map(|point| (point.temperature, point.chi))
                .collect();
            if let Some(peak) = quadratic_peak(&fit_points) {
                peaks.insert(l, peak);
            }
        }
        by_size.insert(l, points);
    }
    let tc = peaks
        .get(&32)
        .zip(peaks.get(&64))
        .map(|(&p32, &p64)| critical_temperature(p32, p64));
    Ok(AnalysisSummary {
        by_size,
        peaks,
        critical_temperature: tc,
    })
}

