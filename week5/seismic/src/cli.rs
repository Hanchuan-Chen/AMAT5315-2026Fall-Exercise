//! `seismic` command line: parse, run one mode, write the artifacts.

use crate::experiment::Experiment;
use crate::npy;
use crate::sim::Model;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

struct Args {
    experiment: String,
    mode: String,
    out: String,
    data: Option<String>,
    storage: Option<String>,
    checkpoints: Option<usize>,
    every: Option<usize>,
}

fn usage() -> String {
    "usage: seismic --experiment FILE --mode forward|born|adjoint --out DIR \
     [--every N] [--data born_data.npy] [--storage full|treeverse] [--checkpoints N]"
        .into()
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut experiment = None;
    let mut mode = None;
    let mut out = None;
    let mut data = None;
    let mut storage = None;
    let mut checkpoints = None;
    let mut every = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--experiment" => experiment = Some(value()?),
            "--mode" => mode = Some(value()?),
            "--out" => out = Some(value()?),
            "--data" => data = Some(value()?),
            "--storage" => storage = Some(value()?),
            "--checkpoints" => {
                checkpoints = Some(
                    value()?
                        .parse()
                        .map_err(|_| "--checkpoints must be a positive integer".to_string())?,
                )
            }
            "--every" => {
                every = Some(
                    value()?
                        .parse()
                        .map_err(|_| "--every must be a positive integer".to_string())?,
                )
            }
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    let args = Args {
        experiment: experiment.ok_or_else(usage)?,
        mode: mode.ok_or_else(usage)?,
        out: out.ok_or_else(usage)?,
        data,
        storage,
        checkpoints,
        every,
    };
    if let Some(every) = args.every {
        if every < 1 {
            return Err("--every must be a positive integer".into());
        }
        if args.mode == "born" {
            return Err("--every applies to forward or adjoint, not born".into());
        }
    }
    match args.storage.as_deref() {
        None | Some("full") | Some("treeverse") => {}
        Some(other) => return Err(format!("unknown storage {other}")),
    }
    Ok(args)
}

fn grid_result(model: &Model, mode: &str) -> Map<String, Value> {
    let exp = &model.exp;
    let mut result = Map::new();
    result.insert("mode".into(), json!(mode));
    result.insert("nx".into(), json!(exp.nx));
    result.insert("nz".into(), json!(exp.nz));
    result.insert("steps".into(), json!(exp.steps));
    result.insert("dt".into(), json!(exp.dt));
    result.insert("dx".into(), json!(exp.dx));
    result.insert("shots".into(), exp.raw["shots"].clone());
    result.insert("receivers".into(), exp.raw["receivers"].clone());
    result
}

fn write_run_json(
    out: &Path,
    model: &Model,
    every: Option<usize>,
    steps: &[usize],
) {
    let mut run = Map::new();
    run.insert("experiment".into(), Value::Object(model.exp.compact()));
    run.insert("experiment_file".into(), json!(model.exp.path));
    if let Some(every) = every {
        run.insert(
            "recording".into(),
            json!({
                "every": every,
                "steps": steps,
                "times": steps.iter().map(|s| *s as f64 * model.exp.dt).collect::<Vec<_>>(),
            }),
        );
    }
    std::fs::write(
        out.join("run.json"),
        serde_json::to_string_pretty(&Value::Object(run)).unwrap() + "\n",
    )
    .unwrap();
}

fn write_result(out: &Path, result: Map<String, Value>) {
    std::fs::write(
        out.join("result.json"),
        serde_json::to_string(&Value::Object(result)).unwrap() + "\n",
    )
    .unwrap();
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn print_rows(mode: &str, norms: &[f64]) {
    println!("shot\tmode\tdata_l2");
    for (shot, norm) in norms.iter().enumerate() {
        println!("{shot}\t{mode}\t{norm:.9}");
    }
}

fn run_forward(model: &Model, args: &Args, out: &Path) -> i32 {
    let exp = &model.exp;
    let nrec = model.receiver_idx.len();
    let mut traces = Vec::with_capacity(exp.shots.len() * exp.steps * nrec);
    let mut norms = Vec::new();
    let mut frames: Vec<(usize, Vec<f64>)> = Vec::new();
    for shot in 0..exp.shots.len() {
        let run = model.forward(shot, &exp.background, None, false, args.every);
        if shot == 0 {
            frames = run.frames;
        }
        norms.push(l2(&run.traces));
        traces.extend_from_slice(&run.traces);
    }
    npy::write_f64(
        &out.join("traces.npy"),
        &[exp.shots.len(), exp.steps, nrec],
        &traces,
    )
    .unwrap();
    if args.every.is_some() {
        let background = model
            .forward(0, &exp.background, None, false, args.every)
            .frames;
        let mut perturbed = exp.background.clone();
        for (c, m) in perturbed.iter_mut().zip(&exp.perturbation) {
            *c += m;
        }
        let perturbed = model.forward(0, &perturbed, None, false, args.every).frames;
        let echo: Vec<(usize, Vec<f32>)> = background
            .iter()
            .zip(&perturbed)
            .map(|((step, b), (_, p))| {
                (*step, b.iter().zip(p).map(|(x, y)| (y - x) as f32).collect())
            })
            .collect();
        let recorded: Vec<(usize, Vec<f32>)> = frames
            .iter()
            .map(|(step, field)| (*step, field.iter().map(|v| *v as f32).collect()))
            .collect();
        model.write_wavefield(&out.join("wavefield.npy"), &recorded);
        model.write_wavefield(&out.join("echo.npy"), &echo);
    }
    let steps: Vec<usize> = frames.iter().map(|(s, _)| *s).collect();
    write_result(out, grid_result(model, "forward"));
    write_run_json(out, model, args.every, &steps);
    print_rows("forward", &norms);
    0
}

fn run_born(model: &Model, _args: &Args, out: &Path) -> i32 {
    let exp = &model.exp;
    let nrec = model.receiver_idx.len();
    let born = model.born();
    let mut flat = Vec::with_capacity(exp.shots.len() * exp.steps * nrec);
    let norms: Vec<f64> = born.iter().map(|d| l2(d)).collect();
    for data in &born {
        flat.extend_from_slice(data);
    }
    npy::write_f64(
        &out.join("born_data.npy"),
        &[exp.shots.len(), exp.steps, nrec],
        &flat,
    )
    .unwrap();
    write_result(out, grid_result(model, "born"));
    write_run_json(out, model, None, &[]);
    print_rows("born", &norms);
    0
}

fn run_adjoint(model: &Model, args: &Args, out: &Path) -> i32 {
    let exp = &model.exp;
    let nrec = model.receiver_idx.len();
    let data_path = args
        .data
        .as_ref()
        .unwrap_or_else(|| panic!("adjoint requires --data"));
    let (shape, flat) = npy::read_f64(Path::new(data_path)).unwrap();
    let expected = [exp.shots.len(), exp.steps, nrec];
    assert_eq!(shape, expected, "--data has the wrong shape");
    let weights: Vec<Vec<f64>> = (0..exp.shots.len())
        .map(|shot| flat[shot * exp.steps * nrec..(shot + 1) * exp.steps * nrec].to_vec())
        .collect();
    let storage = args.storage.as_deref().unwrap_or("full");
    let migrated = if storage == "treeverse" {
        let delta = args
            .checkpoints
            .unwrap_or_else(|| panic!("--storage treeverse requires --checkpoints"));
        model.migrate_treeverse(&weights, delta, args.every)
    } else {
        model.migrate_full(&weights, args.every)
    };
    if storage == "treeverse" {
        model.write_actions(out, &migrated.actions);
    }
    npy::write_f64(&out.join("image.npy"), &[exp.nz, exp.nx], &migrated.image).unwrap();
    let steps = if args.every.is_some() {
        model.write_wavefield(&out.join("wavefield.npy"), &migrated.frames)
    } else {
        Vec::new()
    };
    let mut result = grid_result(model, "adjoint");
    result.insert("statistics".into(), migrated.stats.to_json());
    write_result(out, result);
    write_run_json(out, model, args.every, &steps);
    print_rows("adjoint", &migrated.norms);
    0
}

pub fn run() -> i32 {
    let args = match parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return 2;
        }
    };
    let model = Model::new(Experiment::load(Path::new(&args.experiment)));
    let out = PathBuf::from(&args.out);
    std::fs::create_dir_all(&out).unwrap();
    let started = std::time::Instant::now();
    let code = match args.mode.as_str() {
        "forward" => run_forward(&model, &args, &out),
        "born" => run_born(&model, &args, &out),
        "adjoint" => run_adjoint(&model, &args, &out),
        other => {
            eprintln!("unknown mode {other}\n{}", usage());
            return 2;
        }
    };
    eprintln!(
        "seismic: {} {} ({:.2} s)",
        model.exp.name,
        args.mode,
        started.elapsed().as_secs_f64()
    );
    code
}
