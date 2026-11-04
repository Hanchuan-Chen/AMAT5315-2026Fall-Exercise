use infer::bp;
use infer::checks;
use infer::eval;
use infer::exact::ExactResult;
use infer::plan::{self, Plan};
use infer::uai::{AppliedModel, Model};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

struct Args {
    model: PathBuf,
    evidence: String,
    evidence_file: Option<PathBuf>,
    out: PathBuf,
    order: String,
    order_seen: bool,
    seed: Option<u64>,
    plan: Option<PathBuf>,
    analyze: Option<PathBuf>,
    method: String,
    damping: Option<f64>,
    tolerance: Option<f64>,
    max_sweeps: Option<usize>,
    queries: Option<Vec<usize>>,
}

fn usage() -> &'static str {
    "usage: infer --model FILE --evidence prior|AD|ADX [--evidence-file FILE] --out DIR \
     [--order random|greedy|treesa] [--seed N] [--plan FILE] [--analyze FILE] \
     [--method exact|bp --damping X --tolerance X --max-sweeps N] [--queries 59,102,139]"
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut evidence = None;
    let mut evidence_file = None;
    let mut out = None;
    let mut order = "greedy".to_string();
    let mut order_seen = false;
    let mut seed = None;
    let mut plan_file = None;
    let mut analyze = None;
    let mut method = "exact".to_string();
    let mut damping = None;
    let mut tolerance = None;
    let mut max_sweeps = None;
    let mut queries = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--evidence" => evidence = Some(value()?),
            "--evidence-file" => evidence_file = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--order" => {
                order = value()?;
                order_seen = true;
            }
            "--seed" => seed = Some(value()?.parse().map_err(|_| "--seed must be an integer")?),
            "--plan" => plan_file = Some(PathBuf::from(value()?)),
            "--analyze" => analyze = Some(PathBuf::from(value()?)),
            "--method" => method = value()?,
            "--damping" => {
                damping = Some(value()?.parse().map_err(|_| "--damping must be a number")?)
            }
            "--tolerance" => {
                tolerance = Some(
                    value()?
                        .parse()
                        .map_err(|_| "--tolerance must be a number")?,
                )
            }
            "--max-sweeps" => {
                max_sweeps = Some(
                    value()?
                        .parse()
                        .map_err(|_| "--max-sweeps must be an integer")?,
                )
            }
            "--queries" => {
                queries = Some(
                    value()?
                        .split(',')
                        .map(|x| x.parse::<usize>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| "--queries must be comma-separated integers")?,
                )
            }
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    let args = Args {
        model: model.ok_or_else(|| usage().to_string())?,
        evidence: evidence.ok_or_else(|| usage().to_string())?,
        evidence_file,
        out: out.ok_or_else(|| usage().to_string())?,
        order,
        order_seen,
        seed,
        plan: plan_file,
        analyze,
        method,
        damping,
        tolerance,
        max_sweeps,
        queries,
    };
    if !matches!(args.order.as_str(), "random" | "greedy" | "treesa") {
        return Err(format!("unknown order {}\n{}", args.order, usage()));
    }
    if args.plan.is_some() && args.analyze.is_some() {
        return Err("--plan and --analyze are mutually exclusive".into());
    }
    if args.order_seen && (args.plan.is_some() || args.analyze.is_some()) {
        return Err("--order, --plan and --analyze are mutually exclusive".into());
    }
    if matches!(args.order.as_str(), "random" | "treesa")
        && args.plan.is_none()
        && args.analyze.is_none()
        && args.seed.is_none()
    {
        return Err(format!("--seed is required with --order {}", args.order));
    }
    if args.seed.is_some()
        && (args.plan.is_some()
            || args.analyze.is_some()
            || !matches!(args.order.as_str(), "random" | "treesa"))
    {
        return Err("--seed applies only to --order random or treesa".into());
    }
    if args.method == "bp" {
        if args.damping.is_none() || args.tolerance.is_none() || args.max_sweeps.is_none() {
            return Err("BP requires --damping, --tolerance and --max-sweeps".into());
        }
    } else if args.method != "exact" {
        return Err("--method must be exact or bp".into());
    }
    if args.method == "exact"
        && (args.damping.is_some() || args.tolerance.is_some() || args.max_sweeps.is_some())
    {
        return Err("--damping, --tolerance and --max-sweeps apply only to --method bp".into());
    }
    Ok(args)
}

fn observations(model: &Model, args: &Args) -> Result<BTreeMap<usize, usize>, String> {
    if let Some(path) = &args.evidence_file {
        model.load_evidence(path)
    } else {
        model.asia_observations(&args.evidence)
    }
}

fn input_labels(model: &AppliedModel) -> Vec<Vec<usize>> {
    model.factors.iter().map(|f| f.vars.clone()).collect()
}

fn make_plan(model: &AppliedModel, args: &Args) -> Result<Plan, String> {
    let inputs = input_labels(model);
    let output = Vec::new();
    let evidence = args
        .evidence_file
        .as_ref()
        .map(|p| format!("file:{}", p.display()))
        .unwrap_or_else(|| args.evidence.clone());
    match args.order.as_str() {
        "greedy" => plan::greedy(inputs, output, &model.dims, &evidence),
        "random" => plan::random(inputs, output, &evidence, args.seed.unwrap()),
        "treesa" => plan::treesa(inputs, output, &model.dims, &evidence, args.seed.unwrap()),
        _ => unreachable!(),
    }
}

fn read_plan(path: &Path) -> Result<Plan, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} is not a saved plan: {e}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_string_pretty(value).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn observations_json(model: &Model, obs: &BTreeMap<usize, usize>) -> Value {
    Value::Object(
        obs.iter()
            .map(|(&v, &s)| (model.variable_names()[v].clone(), json!(s)))
            .collect(),
    )
}

fn normalised(results: &BTreeMap<usize, Vec<f64>>) -> bool {
    results
        .values()
        .all(|values| (values.iter().sum::<f64>() - 1.0).abs() < 1e-12)
}

fn default_queries(model: &Model, args: &Args) -> Result<Vec<usize>, String> {
    let queries = if let Some(q) = &args.queries {
        q.clone()
    } else if model.vars == 8 {
        (0..8).collect()
    } else {
        vec![59, 102, 139]
    };
    if let Some(&bad) = queries.iter().find(|&&v| v >= model.vars) {
        return Err(format!("query {bad} is outside the model"));
    }
    Ok(queries)
}

fn run_exact(args: &Args) -> Result<(), String> {
    let started = std::time::Instant::now();
    let model = Model::load(&args.model)?;
    let obs = observations(&model, args)?;
    let exact_model = AppliedModel::with_selectors(&model, &obs)?;
    let expected_inputs = input_labels(&exact_model);
    let expected_output = Vec::new();

    if let Some(plan_path) = &args.plan {
        let plan = read_plan(plan_path)?;
        plan.validate(&expected_inputs, &expected_output)?;
    }
    if let Some(plan_path) = &args.analyze {
        let plan = read_plan(plan_path)?;
        plan.validate(&expected_inputs, &expected_output)?;
        let complexity = plan.complexity(&exact_model.dims)?;
        std::fs::create_dir_all(&args.out)
            .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;
        write_json(
            &args.out.join("complexity.json"),
            &serde_json::to_value(&complexity).unwrap(),
        )?;
        println!(
            "analysis work={} largest_tensor={} steps={}",
            complexity.work,
            complexity.largest_tensor,
            complexity.steps.len()
        );
        return Ok(());
    }

    let tensor_checks = if args.method == "exact" {
        Some(checks::matrix_and_ising()?)
    } else {
        None
    };
    let plan = if let Some(plan_path) = &args.plan {
        read_plan(plan_path)?
    } else {
        make_plan(&exact_model, args)?
    };
    plan.validate(&expected_inputs, &expected_output)?;
    let complexity = plan.complexity(&exact_model.dims)?;
    let queries = default_queries(&model, args)?;
    let joint = if model.vars == 8 { Some((1, 3)) } else { None };
    let contraction = eval::exact(&plan, &exact_model, &queries, joint)?;
    let enumeration = if model.vars == 8 {
        Some(ExactResult::asia_enumeration(&exact_model))
    } else {
        None
    };
    let marginal_err = enumeration.as_ref().map_or(0.0, |e| {
        infer::exact::max_abs_error(&contraction.marginals, &e.marginals)
    });
    let evidence_rel_err = enumeration.as_ref().map_or(0.0, |e| {
        (contraction.evidence_mass - e.evidence_mass).abs() / e.evidence_mass.abs()
    });
    if marginal_err >= 1e-10 || evidence_rel_err >= 1e-10 {
        return Err(format!(
            "exact comparison failed: marginal={marginal_err:e}, evidence={evidence_rel_err:e}"
        ));
    }
    if !normalised(&contraction.marginals) {
        return Err("a node marginal does not sum to one".into());
    }
    if contraction
        .joints
        .values()
        .any(|j| (j.iter().sum::<f64>() - 1.0).abs() >= 1e-12)
    {
        return Err("a joint marginal does not sum to one".into());
    }

    let mut run = json!({
        "schema": "amat5315-week9-infer-v1",
        "model": args.model,
        "evidence": args.evidence,
        "observations": observations_json(&model, &obs),
        "queries": queries,
        "plan": {
            "method": plan.method,
            "seed": plan.seed,
            "work": complexity.work,
            "largest_tensor": complexity.largest_tensor,
            "largest_input": complexity.largest_input,
            "largest_intermediate": complexity.largest_intermediate,
            "steps": complexity.steps.len(),
        },
        "contraction": &contraction,
        "comparison": {
            "max_abs_marginal_error": marginal_err,
            "relative_evidence_mass_error": evidence_rel_err,
            "marginals_normalized": normalised(&contraction.marginals),
            "joints_normalized": contraction.joints.values().all(|j| (j.iter().sum::<f64>() - 1.0).abs() < 1e-12),
        },
    });
    if let Some(enumeration) = &enumeration {
        run["enumeration"] = serde_json::to_value(enumeration).unwrap();
        run["table_check"] = ExactResult::table_check(&model);
    }
    if let Some(check) = &tensor_checks {
        run["tensor_checks"] = serde_json::to_value(check).unwrap();
    }

    let bp_result = if args.method == "bp" {
        let bp_model = AppliedModel::apply(&model, &obs)?;
        Some(bp::run(
            &bp_model,
            args.damping.unwrap(),
            args.tolerance.unwrap(),
            args.max_sweeps.unwrap(),
            &contraction.marginals,
        )?)
    } else {
        None
    };
    if let Some(result) = &bp_result {
        run["bp"] = serde_json::to_value(result).unwrap();
        println!(
            "bp converged={} sweeps={} residual={:.6e} max_marginal_error={:.6e} work={}",
            result.converged,
            result.sweeps_run,
            result.returned_undamped_residual,
            result.max_marginal_error,
            result.cumulative_update_work
        );
    }
    run["runtime_seconds"] = json!(started.elapsed().as_secs_f64());

    std::fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;
    write_json(&args.out.join("run.json"), &run)?;
    write_json(
        &args.out.join("order.json"),
        &serde_json::to_value(&plan).map_err(|e| e.to_string())?,
    )?;
    write_json(
        &args.out.join("complexity.json"),
        &serde_json::to_value(&complexity).map_err(|e| e.to_string())?,
    )?;

    println!(
        "evidence_mass={:.15e} log_evidence={:.15e} plan={} work={} largest_tensor={}",
        contraction.evidence_mass,
        contraction.log_evidence,
        plan.method,
        complexity.work,
        complexity.largest_tensor
    );
    let names = model.variable_names();
    if model.vars == 8 {
        for &var in [1usize, 3, 4].iter() {
            println!("{}={:.12}", names[var], contraction.marginals[&var][0]);
        }
        println!(
            "marginal_error={marginal_err:.3e} evidence_relative_error={evidence_rel_err:.3e}"
        );
        println!(
            "T,L joint minus product max absolute dependence={:.12e}",
            contraction.joints["T,L"]
                .iter()
                .enumerate()
                .map(|(i, joint)| {
                    (joint - contraction.marginals[&1][i / 2] * contraction.marginals[&3][i % 2])
                        .abs()
                })
                .fold(0.0, f64::max)
        );
    } else {
        for &var in &queries {
            println!("P({}=1)={:.12}", var, contraction.marginals[&var][1]);
        }
    }
    Ok(())
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run_exact(&args) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
