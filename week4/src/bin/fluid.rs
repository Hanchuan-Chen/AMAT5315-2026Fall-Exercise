//! `fluid`: integrate the field on stdin, as `week4/fluid.design.toml` fixes.

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, StdoutLock, Write};
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use week4::solver::Solver;
use week4::{Integrator, Method};

const DECIMALS: usize = 6;

#[derive(Parser)]
#[command(name = "fluid", about = "integrate a velocity field on stdin")]
struct Cli {
    /// time integrator: euler, rk2, or rk4
    #[arg(long)]
    method: String,
    /// kinematic viscosity
    #[arg(long)]
    nu: f64,
    /// time step
    #[arg(long)]
    dt: f64,
    /// final integration time
    #[arg(long = "t-end")]
    t_end: f64,
    /// time between snapshots; save at step 0 and every round(every/dt) steps
    #[arg(long)]
    every: f64,
    /// output folder
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize)]
struct Field {
    case: String,
    n: usize,
    seed: Option<i64>,
    k_band: Option<Vec<f64>>,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn write_array(w: &mut impl Write, values: &[f64]) -> std::io::Result<()> {
    write!(w, "[")?;
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{value:.DECIMALS$}")?;
    }
    write!(w, "]")
}

/// The terminal and file record of one snapshot.
struct Recorder {
    stdout: StdoutLock<'static>,
    frames: BufWriter<File>,
}

impl Recorder {
    /// Flush both streams; `std::process::exit` skips destructors.
    fn flush(&mut self) {
        self.stdout.flush().unwrap();
        self.frames.flush().unwrap();
    }

    fn line(&mut self, t: f64, energy: f64, enstrophy: f64) {
        writeln!(
            self.stdout,
            "{t:.1}\t{energy:.DECIMALS$}\t{enstrophy:.DECIMALS$}"
        )
        .unwrap();
    }

    fn snapshot(&mut self, solver: &Solver, omega: &[f64], t: f64, step: usize) {
        let (u, v) = solver.velocity(omega);
        let (energy, enstrophy) = solver.energy_enstrophy(&u, &v, omega);
        self.line(t, energy, enstrophy);
        write!(self.frames, "{{\"t\":{t:.DECIMALS$},\"step\":{step},\"u\":").unwrap();
        write_array(&mut self.frames, &u).unwrap();
        write!(self.frames, ",\"v\":").unwrap();
        write_array(&mut self.frames, &v).unwrap();
        write!(self.frames, ",\"omega\":").unwrap();
        write_array(&mut self.frames, omega).unwrap();
        writeln!(self.frames, "}}").unwrap();
    }
}

fn main() {
    let args = Cli::parse();
    let method = match Method::parse(&args.method) {
        Ok(method) => method,
        Err(message) => {
            eprintln!("fluid: {message}");
            std::process::exit(2);
        }
    };
    let integrator: Box<dyn Integrator> = method.integrator();
    let text = std::io::read_to_string(std::io::stdin()).expect("fluid: read stdin");
    let field: Field = serde_json::from_str(&text).expect("fluid: parse the field on stdin");
    let n = field.n;
    let solver = Solver::new(n, args.nu);

    create_dir_all(&args.out).expect("fluid: create the output folder");
    // `--every` is a time; a snapshot lands every round(every/dt) steps, so the
    // interval recorded in run.json is the one the run really used.
    let steps_between = if args.every > 0.0 {
        ((args.every / args.dt).round() as usize).max(1)
    } else {
        0
    };
    let snapshot_every = steps_between as f64 * args.dt;
    let run = json!({
        "case": field.case,
        "n": n,
        "seed": field.seed,
        "k_band": field.k_band,
        "method": args.method,
        "nu": args.nu,
        "dt": args.dt,
        "t_end": args.t_end,
        "snapshot_every": snapshot_every,
    });
    std::fs::write(
        args.out.join("run.json"),
        serde_json::to_string_pretty(&run).unwrap(),
    )
    .expect("fluid: write run.json");

    let mut record = Recorder {
        stdout: std::io::stdout().lock(),
        frames: BufWriter::new(
            File::create(args.out.join("fields.jsonl")).expect("fluid: create fields.jsonl"),
        ),
    };
    writeln!(record.stdout, "t\tEnergy E\tEnstrophy Z").unwrap();

    let mut omega = solver.vorticity(&field.u, &field.v);
    let mut t = 0.0;
    let mut step = 0usize;
    if steps_between > 0 {
        record.snapshot(&solver, &omega, t, step);
    }
    while t < args.t_end - 1e-12 {
        omega = solver.step(integrator.as_ref(), &omega, args.dt);
        t += args.dt;
        step += 1;
        let (u, v) = solver.velocity(&omega);
        let (energy, enstrophy) = solver.energy_enstrophy(&u, &v, &omega);
        if !energy.is_finite() || !enstrophy.is_finite() {
            record.line(t, energy, enstrophy);
            record.flush();
            std::process::exit(1);
        }
        if steps_between > 0 && step.is_multiple_of(steps_between) {
            record.snapshot(&solver, &omega, t, step);
        }
    }
    record.flush();
}
