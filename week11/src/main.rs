mod model;
mod tensor;

use model::{Network, nqueens_network};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Default)]
struct Args {
    mode: String,
    seed: u64,
    out: String,
    l: Option<usize>,
    p_af: Option<f64>,
    beta: Option<f64>,
    input: Option<String>,
    order: String,
    samples: Option<usize>,
    n: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        order: "treesa".to_string(),
        ..Args::default()
    };
    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--mode" => args.mode = value,
            "--seed" => args.seed = value.parse().map_err(|_| "invalid --seed")?,
            "--out" => args.out = value,
            "--l" => args.l = Some(value.parse().map_err(|_| "invalid --l")?),
            "--p-af" => args.p_af = Some(value.parse().map_err(|_| "invalid --p-af")?),
            "--beta" => args.beta = Some(value.parse().map_err(|_| "invalid --beta")?),
            "--input" => args.input = Some(value),
            "--order" => args.order = value,
            "--samples" => args.samples = Some(value.parse().map_err(|_| "invalid --samples")?),
            "--n" => args.n = Some(value.parse().map_err(|_| "invalid --n")?),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if args.mode.is_empty() {
        return Err("--mode is required".to_string());
    }
    if args.out.is_empty() {
        return Err("--out is required".to_string());
    }
    Ok(args)
}

fn bit(value: u8) -> f64 {
    2.0 * value as f64 - 1.0
}

fn and_energy(a: u8, b: u8, z: u8) -> f64 {
    let (sa, sb, sz) = (bit(a), bit(b), bit(z));
    sa * sb - 2.0 * sa * sz - 2.0 * sb * sz - sa - sb + 2.0 * sz
}

fn or_energy(a: u8, b: u8, z: u8) -> f64 {
    let (sa, sb, sz) = (bit(a), bit(b), bit(z));
    sa * sb - 2.0 * sa * sz - 2.0 * sb * sz + sa + sb - 2.0 * sz
}

fn not_energy(a: u8, z: u8) -> f64 {
    bit(a) * bit(z)
}

fn copy_energy(a: u8, z: u8) -> f64 {
    -bit(a) * bit(z)
}

fn set0_energy(z: u8) -> f64 {
    bit(z)
}

fn set1_energy(z: u8) -> f64 {
    -bit(z)
}

fn xor_energy(a: u8, b: u8, z: u8, w: u8) -> f64 {
    let (sa, sb, sz, sw) = (bit(a), bit(b), bit(z), bit(w));
    sa * sb - sa * sz - 2.0 * sa * sw - sb * sz - 2.0 * sb * sw + 2.0 * sz * sw - sa - sb
        + sz
        + 2.0 * sw
}

fn gadget_rows(name: &str) -> Vec<Value> {
    let mut rows = Vec::new();
    let arity = match name {
        "AND" | "OR" | "XOR" => 3,
        "NOT" | "COPY" => 2,
        "SET0" | "SET1" => 1,
        _ => unreachable!(),
    };
    let floor = match name {
        "AND" | "OR" => -3.0,
        "NOT" | "COPY" | "SET0" | "SET1" => -1.0,
        "XOR" => -4.0,
        _ => unreachable!(),
    };
    for key in 0..(1usize << arity) {
        let inputs = (0..arity)
            .map(|pos| ((key >> (arity - 1 - pos)) & 1) as u8)
            .collect::<Vec<_>>();
        let (energy, helper) = match name {
            "AND" => (and_energy(inputs[0], inputs[1], inputs[2]), None),
            "OR" => (or_energy(inputs[0], inputs[1], inputs[2]), None),
            "NOT" => (not_energy(inputs[0], inputs[1]), None),
            "COPY" => (copy_energy(inputs[0], inputs[1]), None),
            "SET0" => (set0_energy(inputs[0]), None),
            "SET1" => (set1_energy(inputs[0]), None),
            "XOR" => {
                let choices = [0, 1]
                    .map(|w| (xor_energy(inputs[0], inputs[1], inputs[2], w), w))
                    .to_vec();
                let best = choices
                    .iter()
                    .copied()
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .unwrap();
                (best.0, Some(best.1))
            }
            _ => unreachable!(),
        };
        rows.push(json!({
            "bits": inputs,
            "energy": energy,
            "valid": energy == floor,
            "helper_w": helper,
        }));
    }
    rows
}

fn circuit_rows() -> Vec<Value> {
    (0..16)
        .map(|key| {
            let a = ((key >> 3) & 1) as u8;
            let b = ((key >> 2) & 1) as u8;
            let t = ((key >> 1) & 1) as u8;
            let out = (key & 1) as u8;
            let energy = and_energy(a, b, t) + not_energy(t, out) + set1_energy(out);
            json!({"bits": [a, b, t, out], "energy": energy, "valid": energy == -5.0})
        })
        .collect()
}

fn run_gadgets(args: &Args, start: Instant) -> Result<(), String> {
    let names = ["AND", "OR", "NOT", "COPY", "SET0", "SET1", "XOR"];
    let mut gates = BTreeMap::new();
    for name in names {
        gates.insert(name, gadget_rows(name));
    }
    let circuit = circuit_rows();
    let floors = BTreeMap::from([
        ("AND", -3),
        ("OR", -3),
        ("NOT", -1),
        ("COPY", -1),
        ("SET0", -1),
        ("SET1", -1),
        ("XOR", -4),
        ("AND-NOT-SET1", -5),
    ]);
    let valid_count = circuit.iter().filter(|row| row["valid"] == true).count();
    if valid_count != 3 {
        return Err(format!(
            "AND-NOT-SET1 has {valid_count} ground states, expected 3"
        ));
    }
    let runtime_ms = start.elapsed().as_millis() as u64;
    let evidence = json!({
        "schema_version": 1,
        "mode": "gadgets",
        "seed": args.seed,
        "run_parameters": {"seed": args.seed},
        "floors": floors,
        "gadgets": gates,
        "circuit": {
            "name": "AND-NOT-SET1",
            "variables": ["a", "b", "t", "out"],
            "rows": circuit,
            "ground_state_count": valid_count,
        },
        "runtime_ms": runtime_ms,
    });
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    fs::write(
        Path::new(&args.out).join("run.json"),
        serde_json::to_string_pretty(&evidence).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for (name, rows) in evidence["gadgets"].as_object().unwrap() {
        for row in rows.as_array().unwrap() {
            let bits = row["bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap().to_string())
                .collect::<String>();
            println!("gadgets\t{name}[{bits}]\t{}\t{runtime_ms}", row["energy"]);
        }
    }
    for row in evidence["circuit"]["rows"].as_array().unwrap() {
        let bits = row["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap().to_string())
            .collect::<String>();
        println!(
            "gadgets\tAND-NOT-SET1[{bits}]\t{}\t{runtime_ms}",
            row["energy"]
        );
    }
    Ok(())
}

fn spin_network(args: &Args) -> Result<Network, String> {
    if let Some(path) = &args.input {
        return Network::from_bundle(path);
    }
    if let Some(l) = args.l {
        let p_af = args.p_af.ok_or("--p-af is required with --l")?;
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
        return Ok(Network::open_lattice(l, p_af, &mut rng));
    }
    Err("one of --input or --l is required".to_string())
}

fn run_plan(args: &Args, start: Instant) -> Result<(), String> {
    let network = if let Some(n) = args.n {
        nqueens_network(n)
    } else {
        spin_network(args)?
    };
    let (_, tc, sc) = tensor::plan(&network, &args.order)?;
    let runtime_ms = start.elapsed().as_millis() as u64;
    let evidence = json!({
        "schema_version": 1,
        "mode": "plan",
        "seed": args.seed,
        "run_parameters": {
            "order": args.order,
            "input": args.input,
            "l": args.l,
            "p_af": args.p_af,
            "n": args.n,
        },
        "network": {
            "variables": network.nvars,
            "tensors": network.factors.len(),
            "edges": network.edges.len(),
        },
        "tc": tc,
        "sc": sc,
        "runtime_ms": runtime_ms,
    });
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    fs::write(
        Path::new(&args.out).join("run.json"),
        serde_json::to_string_pretty(&evidence).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("mode\tmetric\tvalue\truntime_ms");
    println!("plan\tvariables\t{}\t{runtime_ms}", network.nvars);
    println!("plan\ttc\t{tc:.12}\t{runtime_ms}");
    println!("plan\tsc\t{sc:.12}\t{runtime_ms}");
    Ok(())
}

fn output_evidence(out: &str, evidence: &Value) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    fs::write(
        Path::new(out).join("run.json"),
        serde_json::to_string_pretty(evidence).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn run_count(args: &Args, start: Instant) -> Result<(), String> {
    if let Some(n) = args.n {
        let network = nqueens_network(n);
        let order = &args.order;
        let (tree, tc, sc) = tensor::plan_fast(&network, order)?;
        let (count, _, _) = tensor::count_sparse(&network, &tree)?;
        let runtime_ms = start.elapsed().as_millis() as u64;
        let evidence = json!({
            "schema_version": 1,
            "mode": "count",
            "seed": args.seed,
            "run_parameters": {"n": n, "order": order},
            "network": {"variables": network.nvars, "tensors": network.factors.len()},
            "tc": tc,
            "sc": sc,
            "count": count,
            "runtime_ms": runtime_ms,
        });
        output_evidence(&args.out, &evidence)?;
        println!("mode\tmetric\tvalue\truntime_ms");
        println!("count\tN\t{n}\t{runtime_ms}");
        println!("count\tvalid_boards\t{count}\t{runtime_ms}");
        println!("count\tsc\t{sc:.12}\t{runtime_ms}");
        return Ok(());
    }

    let beta = args.beta.ok_or("--beta is required for spin-glass count")?;
    let network = spin_network(args)?;
    let (tree, tc, sc) = tensor::plan(&network, &args.order)?;
    let (log_z, _, _) = tensor::count_log(&network, &tree, beta)?;
    let (e_min, g0) = tensor::tropical(&network, &tree);
    let runtime_ms = start.elapsed().as_millis() as u64;
    let evidence = json!({
        "schema_version": 1,
        "mode": "count",
        "seed": args.seed,
        "run_parameters": {
            "input": args.input,
            "l": args.l,
            "p_af": args.p_af,
            "beta": beta,
            "order": args.order,
        },
        "network": {
            "variables": network.nvars,
            "tensors": network.factors.len(),
            "edges": network.edges.len(),
        },
        "tc": tc,
        "sc": sc,
        "log_z": log_z,
        "e_min": e_min,
        "g0": g0,
        "runtime_ms": runtime_ms,
    });
    output_evidence(&args.out, &evidence)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    println!("count\tlogZ\t{log_z:.15}\t{runtime_ms}");
    println!("count\tE_min\t{e_min:.15}\t{runtime_ms}");
    println!("count\tg0\t{g0}\t{runtime_ms}");
    println!("count\tsc\t{sc:.12}\t{runtime_ms}");
    println!("count\ttc\t{tc:.12}\t{runtime_ms}");
    Ok(())
}

fn valid_queens(board: &[u8]) -> bool {
    let n = board.len();
    let mut rows = vec![0; n];
    let mut cols = vec![0; n];
    let mut diag_a = vec![0; 2 * n - 1];
    let mut diag_b = vec![0; 2 * n - 1];
    for (r, &c) in board.iter().enumerate() {
        if cols[c as usize] != 0
            || diag_a[r + c as usize] != 0
            || diag_b[r + n - 1 - c as usize] != 0
        {
            return false;
        }
        rows[r] = 1;
        cols[c as usize] = 1;
        diag_a[r + c as usize] = 1;
        diag_b[r + n - 1 - c as usize] = 1;
    }
    rows.iter().all(|&x| x == 1)
}

fn run_sample(args: &Args, start: Instant) -> Result<(), String> {
    let samples = args.samples.ok_or("--samples is required")?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    if let Some(n) = args.n {
        let network = nqueens_network(n);
        let order = &args.order;
        let (tree, tc, sc) = tensor::plan_fast(&network, order)?;
        let (count, tensors, _) = tensor::count_sparse(&network, &tree)?;
        let mut boards = Vec::with_capacity(samples);
        let mut frequencies = BTreeMap::<String, usize>::new();
        for _ in 0..samples {
            let bits = tensor::sample_count(&network, &tree, &tensors, &mut rng);
            let board: Vec<u8> = (0..n)
                .map(|r| (0..n).find(|&c| bits[r * n + c] == 1).unwrap_or(0) as u8)
                .collect();
            if !valid_queens(&board) {
                return Err(format!("sampled invalid {n}-queens board: {board:?}"));
            }
            *frequencies.entry(format!("{board:?}")).or_default() += 1;
            boards.push(board);
        }
        let distinct = frequencies.len();
        let runtime_ms = start.elapsed().as_millis() as u64;
        let evidence = json!({
            "schema_version": 1,
            "mode": "sample",
            "seed": args.seed,
            "run_parameters": {"n": n, "order": order, "samples": samples},
            "network": {"variables": network.nvars, "tensors": network.factors.len()},
            "tc": tc,
            "sc": sc,
            "valid_board_count": count,
            "sample_count": samples,
            "distinct_boards": distinct,
            "frequencies": frequencies,
            "boards": boards,
            "runtime_ms": runtime_ms,
        });
        output_evidence(&args.out, &evidence)?;
        println!("mode\tmetric\tvalue\truntime_ms");
        println!("sample\tN\t{n}\t{runtime_ms}");
        println!("sample\tsamples\t{samples}\t{runtime_ms}");
        println!("sample\tdistinct_boards\t{distinct}\t{runtime_ms}");
        return Ok(());
    }

    let beta = args
        .beta
        .ok_or("--beta is required for spin-glass sample")?;
    let network = spin_network(args)?;
    let (tree, tc, sc) = tensor::plan(&network, &args.order)?;
    let (log_z, tensors, _) = tensor::count_log(&network, &tree, beta)?;
    let (e_min, g0) = tensor::tropical(&network, &tree);
    let mut histogram = BTreeMap::<String, usize>::new();
    let mut states = Vec::with_capacity(samples);
    for _ in 0..samples {
        let bits = tensor::sample_log(&network, &tree, &tensors, &mut rng);
        let energy = network.energy(&bits);
        *histogram.entry(format!("{energy:.0}")).or_default() += 1;
        states.push(json!({"bits": bits, "energy": energy}));
    }
    let predicted_ground = samples as f64 * g0 as f64 * (-beta * e_min - log_z).exp();
    let runtime_ms = start.elapsed().as_millis() as u64;
    let evidence = json!({
        "schema_version": 1,
        "mode": "sample",
        "seed": args.seed,
        "run_parameters": {
            "input": args.input,
            "l": args.l,
            "p_af": args.p_af,
            "beta": beta,
            "order": args.order,
            "samples": samples,
        },
        "network": {
            "variables": network.nvars,
            "tensors": network.factors.len(),
            "edges": network.edges.len(),
        },
        "tc": tc,
        "sc": sc,
        "log_z": log_z,
        "e_min": e_min,
        "g0": g0,
        "predicted_ground_count": predicted_ground,
        "energy_histogram": histogram,
        "samples": states,
        "runtime_ms": runtime_ms,
    });
    output_evidence(&args.out, &evidence)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    println!("sample\tsamples\t{samples}\t{runtime_ms}");
    println!("sample\tE_min\t{e_min:.15}\t{runtime_ms}");
    println!("sample\tg0\t{g0}\t{runtime_ms}");
    println!("sample\tpredicted_ground\t{predicted_ground:.15}\t{runtime_ms}");
    Ok(())
}

fn main() {
    let start = Instant::now();
    let result = parse_args().and_then(|args| match args.mode.as_str() {
        "gadgets" => run_gadgets(&args, start),
        "plan" => run_plan(&args, start),
        "count" => run_count(&args, start),
        "sample" => run_sample(&args, start),
        other => Err(format!("unsupported mode {other}")),
    });
    if let Err(error) = result {
        eprintln!("tnsample: {error}");
        std::process::exit(1);
    }
}
