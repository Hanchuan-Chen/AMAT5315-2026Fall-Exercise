use std::collections::BTreeSet;
use std::fs;
use std::time::Instant;

use branchopt::{
    FEATURES, ModelFile, Policy, SearchConfig, SpinGlass, branch_and_bound, random_instance,
    solve_diet, solve_lp, solve_milp,
};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
struct Args {
    mode: String,
    spins: usize,
    edges: usize,
    count: usize,
    seed: u64,
    out: String,
    policy: Policy,
    export_labels: bool,
    model: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        mode: String::new(),
        spins: 22,
        edges: 52,
        count: 16,
        seed: 2026,
        out: String::new(),
        policy: Policy::MostFrac,
        export_labels: false,
        model: None,
    };
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--mode" => args.mode = next(&mut values, &flag)?,
            "--spins" => args.spins = parse(&mut values, &flag)?,
            "--edges" => args.edges = parse(&mut values, &flag)?,
            "--count" => args.count = parse(&mut values, &flag)?,
            "--seed" => args.seed = parse(&mut values, &flag)?,
            "--out" => args.out = next(&mut values, &flag)?,
            "--policy" => args.policy = Policy::parse(&next(&mut values, &flag)?)?,
            "--model" => args.model = Some(next(&mut values, &flag)?),
            "--export-labels" => args.export_labels = true,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if args.mode.is_empty() || args.out.is_empty() {
        return Err("--mode and --out are required".into());
    }
    Ok(args)
}

fn next(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    values
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse<T: std::str::FromStr>(
    values: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    next(values, flag)?
        .parse()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("branchopt: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.edges > args.spins * (args.spins - 1) / 2 {
        return Err("--edges exceeds the number of possible simple bonds".into());
    }
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    println!("mode\tmetric\tvalue\truntime_ms");
    match args.mode.as_str() {
        "gap" => run_gap(&args, started),
        "bnb" => run_bnb(&args, started),
        "eval" => run_eval(&args, started),
        other => Err(format!("unknown mode {other}")),
    }
}

fn row(mode: &str, metric: &str, value: impl ToString, elapsed: &Instant) {
    println!(
        "{mode}\t{metric}\t{}\t{:.3}",
        value.to_string(),
        elapsed.elapsed().as_secs_f64() * 1000.0
    );
}

fn write_json(path: &str, value: &Value) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_string_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn format_vector(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn run_gap(args: &Args, started: Instant) -> Result<(), String> {
    let square = SpinGlass::square();
    let square_lp = solve_lp(&square, &[0.0; 4], &[1.0; 4])?;
    let (square_bits, square_exact) = solve_milp(&square)?;
    let (_, square_rounded) = square.nearest_rounding(&square_lp.spins);
    let (diet_quantities, diet_cost) = solve_diet()?;

    let mut instances = Vec::new();
    let mut lp_bounds = Vec::new();
    let mut exact_energies = Vec::new();
    let mut gaps = Vec::new();
    let mut fractional_counts = Vec::new();
    let mut rounded_excesses = Vec::new();
    let mut suboptimal_roundings = 0;
    for id in 0..args.count {
        let graph = random_instance(args.spins, args.edges, args.seed, id);
        let lp = solve_lp(&graph, &vec![0.0; args.spins], &vec![1.0; args.spins])?;
        let (exact_bits, exact_energy) = solve_milp(&graph)?;
        let (_, rounded_energy) = graph.nearest_rounding(&lp.spins);
        let gap = exact_energy - lp.energy;
        let rounded_excess = rounded_energy - exact_energy;
        let fractional = SpinGlass::fractional_count(&lp.spins);
        if rounded_energy > exact_energy + 1e-9 {
            suboptimal_roundings += 1;
        }
        lp_bounds.push(lp.energy);
        exact_energies.push(exact_energy);
        gaps.push(gap);
        fractional_counts.push(fractional);
        rounded_excesses.push(rounded_excess);
        instances.push(json!({
            "id": id,
            "lp_bound": lp.energy,
            "lp_spins": lp.spins,
            "exact_energy": exact_energy,
            "exact_bits": exact_bits,
            "integrality_gap": gap,
            "fractional_spins": fractional,
            "rounded_energy": rounded_energy,
            "rounding_excess": rounded_excess,
        }));
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let summary = json!({
        "mean_lp_bound": mean(&lp_bounds),
        "mean_exact_energy": mean(&exact_energies),
        "mean_integrality_gap": mean(&gaps),
        "mean_fractional_spins": mean(&fractional_counts.iter().map(|v| *v as f64).collect::<Vec<_>>()),
        "suboptimal_roundings": suboptimal_roundings,
        "mean_rounding_excess": mean(&rounded_excesses),
    });
    let output = json!({
        "schema": "branchopt-run-v1",
        "mode": "gap",
        "seed": args.seed,
        "parameters": {
            "spins": args.spins, "edges": args.edges, "count": args.count,
            "highs": {"presolve": false, "threads": 1}
        },
        "opening": {
            "square": {
                "lp_bound": square_lp.energy,
                "lp_spins": square_lp.spins,
                "exact_energy": square_exact,
                "exact_bits": square_bits,
                "rounded_energy": square_rounded,
            },
            "diet": {"quantities": diet_quantities, "cost": diet_cost}
        },
        "summary": summary,
        "instances": instances,
    });
    write_json(&format!("{}/run.json", args.out), &output)?;
    row(
        "gap",
        "square_lp_spins",
        format_vector(&square_lp.spins),
        &started,
    );
    row(
        "gap",
        "square_integer_spins",
        format_vector(&square_bits),
        &started,
    );
    row(
        "gap",
        "square_lp_bound",
        format!("{:.6}", square_lp.energy),
        &started,
    );
    row(
        "gap",
        "square_exact_energy",
        format!("{square_exact:.6}"),
        &started,
    );
    row(
        "gap",
        "square_rounded_energy",
        format!("{square_rounded:.6}"),
        &started,
    );
    row(
        "gap",
        "diet_quantities",
        format_vector(&diet_quantities),
        &started,
    );
    row("gap", "diet_cost", format!("{diet_cost:.6}"), &started);
    for key in [
        "mean_lp_bound",
        "mean_exact_energy",
        "mean_integrality_gap",
        "mean_fractional_spins",
        "suboptimal_roundings",
        "mean_rounding_excess",
    ] {
        row("gap", key, &summary[key], &started);
    }
    Ok(())
}

fn run_bnb(args: &Args, started: Instant) -> Result<(), String> {
    let model = match &args.model {
        Some(path) => Some(ModelFile::load(path)?),
        None => None,
    };
    let mut instances = Vec::new();
    let mut nodes = Vec::new();
    let mut solves = Vec::new();
    let mut times = Vec::new();
    let mut errors = Vec::new();
    let mut labels = Vec::new();
    for id in 0..args.count {
        let graph = random_instance(args.spins, args.edges, args.seed, id);
        let (_, optimum) = solve_milp(&graph)?;
        let result = branch_and_bound(
            &graph,
            &SearchConfig {
                policy: args.policy,
                model: model.as_ref(),
                export_labels: args.export_labels,
                trace: false,
            },
        )?;
        let error = (result.energy - optimum).abs();
        nodes.push(result.nodes);
        solves.push(result.lp_solves);
        times.push(result.wall_ms);
        errors.push(error);
        labels.extend(result.labels.clone());
        instances.push(json!({
            "id": id,
            "policy": args.policy.name(),
            "exact_energy": optimum,
            "energy": result.energy,
            "error": error,
            "nodes": result.nodes,
            "lp_solves": result.lp_solves,
            "wall_ms": result.wall_ms,
            "lp_ms": result.lp_ms,
            "first_splits": result.first_splits,
        }));
    }
    let tetra = branch_and_bound(
        &SpinGlass::tetrahedron(),
        &SearchConfig {
            policy: Policy::MostFrac,
            model: None,
            export_labels: false,
            trace: true,
        },
    )?;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
    let summary = json!({
        "policy": args.policy.name(),
        "mean_nodes": mean(&nodes.iter().map(|v| *v as f64).collect::<Vec<_>>()),
        "mean_lp_solves": mean(&solves.iter().map(|v| *v as f64).collect::<Vec<_>>()),
        "mean_wall_ms": mean(&times),
        "largest_highs_error": errors.iter().copied().fold(0.0, f64::max),
    });
    let output = json!({
        "schema": "branchopt-run-v1",
        "mode": "bnb",
        "seed": args.seed,
        "parameters": {
            "spins": args.spins, "edges": args.edges, "count": args.count,
            "policy": args.policy.name(), "export_labels": args.export_labels,
            "highs": {"presolve": false, "threads": 1}
        },
        "summary": summary,
        "instances": instances,
        "tetrahedron": tetra.trace,
    });
    write_json(&format!("{}/run.json", args.out), &output)?;
    row("bnb", "policy", args.policy.name(), &started);
    row("bnb", "mean_nodes", &summary["mean_nodes"], &started);
    row(
        "bnb",
        "mean_lp_solves",
        &summary["mean_lp_solves"],
        &started,
    );
    row("bnb", "mean_wall_ms", &summary["mean_wall_ms"], &started);
    row(
        "bnb",
        "largest_highs_error",
        &summary["largest_highs_error"],
        &started,
    );
    row("bnb", "tetrahedron_nodes", tetra.nodes, &started);
    for node in &tetra.trace {
        row(
            "bnb",
            &format!("tetra_node_{}", node.node),
            format!("{:?} {} {}", node.branch, node.status, node.lp_bound),
            &started,
        );
    }
    if args.export_labels {
        let train: Vec<_> = labels.iter().filter(|node| node.instance < 36).collect();
        let validation: Vec<_> = labels.iter().filter(|node| node.instance >= 36).collect();
        let dataset = json!({
            "features": FEATURES,
            "split": {
                "train_instances": (0..36).collect::<Vec<_>>(),
                "validation_instances": (36..48).collect::<Vec<_>>(),
            },
            "nodes": labels,
        });
        write_json(&format!("{}/dataset.json", args.out), &dataset)?;
        row("bnb", "train_nodes", train.len(), &started);
        row(
            "bnb",
            "train_instances",
            train
                .iter()
                .map(|n| n.instance)
                .collect::<BTreeSet<_>>()
                .len(),
            &started,
        );
        row("bnb", "validation_nodes", validation.len(), &started);
        row(
            "bnb",
            "validation_instances",
            validation
                .iter()
                .map(|n| n.instance)
                .collect::<BTreeSet<_>>()
                .len(),
            &started,
        );
    }
    Ok(())
}

fn run_eval(args: &Args, started: Instant) -> Result<(), String> {
    let model_path = args.model.as_ref().ok_or("--mode eval requires --model")?;
    let model = ModelFile::load(model_path)?;
    let probe_error = model.probe_error()?;
    if probe_error >= 1e-4 {
        return Err(format!(
            "JAX-Rust probe error {probe_error:e} is not below 1e-4"
        ));
    }
    let policies = [
        Policy::MostFrac,
        Policy::Neural,
        Policy::RootNeural,
        Policy::Strong,
    ];
    let mut instances_json = Vec::new();
    let mut largest_error: f64 = 0.0;
    for id in 0..args.count {
        let graph = random_instance(args.spins, args.edges, args.seed, id);
        let (_, optimum) = solve_milp(&graph)?;
        let mut policy_values = serde_json::Map::new();
        for policy in policies {
            let result = branch_and_bound(
                &graph,
                &SearchConfig {
                    policy,
                    model: Some(&model),
                    export_labels: false,
                    trace: false,
                },
            )?;
            let error = (result.energy - optimum).abs();
            largest_error = largest_error.max(error);
            policy_values.insert(
                policy.name().into(),
                json!({
                    "energy": result.energy,
                    "error": error,
                    "nodes": result.nodes,
                    "lp_solves": result.lp_solves,
                    "wall_ms": result.wall_ms,
                    "lp_ms": result.lp_ms,
                    "inference_ms": result.inference_ms,
                    "first_splits": result.first_splits,
                }),
            );
        }
        instances_json.push(json!({
            "id": id,
            "exact_energy": optimum,
            "policies": policy_values,
        }));
    }
    if largest_error >= 1e-6 {
        return Err(format!(
            "largest HiGHS error {largest_error:e} is not below 1e-6"
        ));
    }

    let summary = summarize_eval(&instances_json);
    let most = &summary["most-frac"];
    let neural = &summary["neural"];
    let root_neural = &summary["root-neural"];
    let strong = &summary["strong"];
    if as_f64(&neural["mean_nodes"]) >= as_f64(&most["mean_nodes"])
        || as_f64(&root_neural["mean_nodes"]) >= as_f64(&most["mean_nodes"])
    {
        return Err(
            "neural and root-neural must both explore fewer mean nodes than most-frac".into(),
        );
    }
    if as_f64(&strong["mean_wall_ms"]) <= as_f64(&most["mean_wall_ms"]) {
        return Err("strong branching must be visibly slower than most-frac".into());
    }
    let lp_equal = ["most-frac", "neural", "root-neural"].iter().all(|policy| {
        instances_json.iter().all(|instance| {
            as_f64(&instance["policies"][policy]["lp_solves"])
                == as_f64(&instance["policies"][policy]["nodes"])
        })
    });
    if !lp_equal {
        return Err("a non-Strong policy has LP solves different from nodes".into());
    }
    let largest_saving = instances_json
        .iter()
        .max_by(|a, b| {
            let sa = as_f64(&a["policies"]["most-frac"]["nodes"])
                - as_f64(&a["policies"]["neural"]["nodes"]);
            let sb = as_f64(&b["policies"]["most-frac"]["nodes"])
                - as_f64(&b["policies"]["neural"]["nodes"]);
            sa.partial_cmp(&sb).unwrap()
        })
        .ok_or("no evaluation instances")?;
    let size = format!("{}x{}", args.spins, args.edges);
    let output = json!({
        "schema": "branchopt-run-v1",
        "mode": "eval",
        "seed": args.seed,
        "parameters": {
            "spins": args.spins, "edges": args.edges, "count": args.count,
            "policies": policies.iter().map(|p| p.name()).collect::<Vec<_>>(),
            "model": model_path,
            "highs": {"presolve": false, "threads": 1}
        },
        "probe": {
            "probe_count": model.probes.len(),
            "largest_probability_difference": probe_error
        },
        "training": {
            "train_nodes": model.split.train_nodes,
            "train_instances": model.split.train_instances.len(),
            "validation_nodes": model.split.validation_nodes,
            "validation_instances": model.split.validation_instances.len(),
            "overlap": model.split.overlap.len(),
            "checkpoint_epoch": model.checkpoint_epoch,
            "best_validation_cross_entropy": model.best_validation_cross_entropy
        },
        "lp_solves_equal_nodes": lp_equal,
        "largest_highs_error": largest_error,
        "summary": summary,
        "instances": instances_json,
        "largest_neural_saving": {
            "instance": largest_saving["id"],
            "most_frac": largest_saving["policies"]["most-frac"],
            "neural": largest_saving["policies"]["neural"],
        }
    });
    write_json(&format!("{}/run.json", args.out), &output)?;
    row("eval", "size", size, &started);
    row("eval", "train_nodes", model.split.train_nodes, &started);
    row(
        "eval",
        "train_instances",
        model.split.train_instances.len(),
        &started,
    );
    row(
        "eval",
        "validation_nodes",
        model.split.validation_nodes,
        &started,
    );
    row(
        "eval",
        "validation_instances",
        model.split.validation_instances.len(),
        &started,
    );
    row("eval", "split_overlap", model.split.overlap.len(), &started);
    row("eval", "checkpoint_epoch", model.checkpoint_epoch, &started);
    row(
        "eval",
        "best_validation_cross_entropy",
        model.best_validation_cross_entropy,
        &started,
    );
    row("eval", "probe_count", model.probes.len(), &started);
    row("eval", "largest_probe_difference", probe_error, &started);
    row("eval", "lp_solves_equal_nodes", lp_equal, &started);
    row("eval", "largest_highs_error", largest_error, &started);
    for policy in ["most-frac", "neural", "root-neural", "strong"] {
        for metric in [
            "mean_nodes",
            "mean_wall_ms",
            "mean_lp_ms",
            "mean_inference_ms",
            "mean_lp_solves",
        ] {
            row(
                "eval",
                &format!("{policy}_{metric}"),
                &summary[policy][metric],
                &started,
            );
        }
    }
    row(
        "eval",
        "neural_largest_saving_instance",
        largest_saving["id"].clone(),
        &started,
    );
    for policy in ["most-frac", "neural"] {
        row(
            "eval",
            &format!("{policy}_largest_saving_splits"),
            largest_saving["policies"][policy]["first_splits"].to_string(),
            &started,
        );
        for metric in ["nodes", "wall_ms"] {
            row(
                "eval",
                &format!("{policy}_largest_saving_{metric}"),
                &largest_saving["policies"][policy][metric],
                &started,
            );
        }
    }
    Ok(())
}

fn summarize_eval(instances: &[Value]) -> Value {
    let mut summary = serde_json::Map::new();
    for policy in ["most-frac", "neural", "root-neural", "strong"] {
        let values: Vec<&Value> = instances
            .iter()
            .map(|instance| &instance["policies"][policy])
            .collect();
        let mean = |metric: &str| {
            values
                .iter()
                .map(|value| as_f64(&value[metric]))
                .sum::<f64>()
                / values.len() as f64
        };
        summary.insert(
            policy.into(),
            json!({
                "mean_nodes": mean("nodes"),
                "mean_wall_ms": mean("wall_ms"),
                "mean_lp_ms": mean("lp_ms"),
                "mean_inference_ms": mean("inference_ms"),
                "mean_lp_solves": mean("lp_solves"),
            }),
        );
    }
    Value::Object(summary)
}

fn as_f64(value: &Value) -> f64 {
    value.as_f64().unwrap()
}
