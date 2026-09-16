//! The three files a run writes, in the exact formats the contract fixes.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::ramp::{RunConfig, Update};

/// Streams `series.jsonl` and `spins.jsonl` for one run.
///
/// The rows are formatted by hand rather than by a JSON float printer, so
/// "six decimals" is a property of this writer: `M` and `E` are always written
/// with exactly six fractional digits and the frame magnetization with four.
/// A Wolff run appends `cluster_size` to every measured row; a Metropolis run
/// writes the five keys the contract lists and nothing else.
pub struct Recorder {
    l: usize,
    update: Update,
    series: BufWriter<File>,
    spins: Option<BufWriter<File>>,
}

impl Recorder {
    /// Open the writers for `out_dir`; `spins.jsonl` only when `every > 0`.
    pub fn create(out_dir: &Path, l: usize, update: Update, every: u64) -> Result<Self, String> {
        let series = open(out_dir, "series.jsonl")?;
        let spins = if every > 0 {
            Some(open(out_dir, "spins.jsonl")?)
        } else {
            None
        };
        Ok(Self {
            l,
            update,
            series,
            spins,
        })
    }

    /// One measured step: `{"L":..,"T":..,"sweep":..,"M":..,"E":..}`,
    /// with `"cluster_size":n` appended for a Wolff move.
    pub fn write_series_row(
        &mut self,
        temperature: f64,
        sweep: u64,
        m: f64,
        energy_per_site: f64,
        cluster_size: u64,
    ) -> Result<(), String> {
        let mut line = format!(
            "{{\"L\":{},\"T\":{},\"sweep\":{sweep},\"M\":{m:.6},\"E\":{energy_per_site:.6}",
            self.l,
            temperature_json(temperature),
        );
        if self.update == Update::Wolff {
            line.push_str(&format!(",\"cluster_size\":{cluster_size}"));
        }
        line.push_str("}\n");
        self.series
            .write_all(line.as_bytes())
            .map_err(|error| format!("write series.jsonl: {error}"))
    }

    /// One recorded frame: `{"L":..,"T":..,"sweep":..,"m":..,"spins":[..]}`.
    ///
    /// `sweep` is the global counter, which includes every discarded step of
    /// the whole ramp, so the history reads as one timeline.
    pub fn write_spin_frame(
        &mut self,
        temperature: f64,
        sweep: u64,
        m: f64,
        spins: &[i8],
    ) -> Result<(), String> {
        let Some(writer) = self.spins.as_mut() else {
            return Ok(());
        };
        let mut line = String::with_capacity(spins.len() * 3 + 64);
        line.push_str(&format!(
            "{{\"L\":{},\"T\":{},\"sweep\":{sweep},\"m\":{m:.4},\"spins\":[",
            self.l,
            temperature_json(temperature),
        ));
        for (index, spin) in spins.iter().enumerate() {
            if index > 0 {
                line.push(',');
            }
            line.push_str(if *spin > 0 { "1" } else { "-1" });
        }
        line.push_str("]}\n");
        writer
            .write_all(line.as_bytes())
            .map_err(|error| format!("write spins.jsonl: {error}"))
    }

    /// Flush both streams.
    pub fn finish(mut self) -> Result<(), String> {
        self.series
            .flush()
            .map_err(|error| format!("flush series.jsonl: {error}"))?;
        if let Some(writer) = self.spins.as_mut() {
            writer
                .flush()
                .map_err(|error| format!("flush spins.jsonl: {error}"))?;
        }
        Ok(())
    }
}

/// `<out>/run.json`, with exactly the fields the contract lists, in order.
pub fn write_run_json(config: &RunConfig, t_grid: &[f64]) -> Result<(), String> {
    #[derive(Serialize)]
    struct RunJson<'a> {
        #[serde(rename = "L")]
        l: usize,
        update: &'a str,
        t_grid: &'a [f64],
        discard: u64,
        measure: u64,
        seed: u64,
        sample_every: u64,
        time_unit: &'a str,
    }

    let document = RunJson {
        l: config.l,
        update: config.update.name(),
        t_grid,
        discard: config.discard,
        measure: config.measure,
        seed: config.seed,
        sample_every: 1,
        time_unit: config.update.time_unit(),
    };
    let text = serde_json::to_string_pretty(&document)
        .map_err(|error| format!("serialize run.json: {error}"))?;
    let path = config.out.join("run.json");
    fs::write(&path, format!("{text}\n"))
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn open(out_dir: &Path, name: &str) -> Result<BufWriter<File>, String> {
    let path = out_dir.join(name);
    File::create(&path)
        .map(BufWriter::new)
        .map_err(|error| format!("create {}: {error}", path.display()))
}

/// A JSON number for a temperature: whole temperatures keep their `.0`.
fn temperature_json(temperature: f64) -> String {
    serde_json::to_string(&temperature).unwrap_or_else(|_| temperature.to_string())
}
