use std::fs;
use std::path::PathBuf;

use serde_json::json;

fn main() {
    if let Err(error) = run() {
        eprintln!("thermo: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut raw_args = std::env::args().skip(1);
    let first = raw_args.next().ok_or_else(usage)?;
    let command = if first == "--method" {
        raw_args.next().ok_or_else(usage)?
    } else {
        first
    };
    let mut options = Vec::new();
    let mut iter = raw_args;
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        options.push((flag, value));
    }
    let get = |name: &str| {
        options
            .iter()
            .find(|(flag, _)| flag == name)
            .map(|(_, value)| value.as_str())
            .ok_or_else(|| format!("missing {name}"))
    };
    let l =
        || get("--l").and_then(|value| value.parse::<usize>().map_err(|_| "--l invalid".into()));
    let k = || get("--k").and_then(|value| value.parse::<f64>().map_err(|_| "--k invalid".into()));
    let chi = || {
        get("--chi").and_then(|value| value.parse::<usize>().map_err(|_| "--chi invalid".into()))
    };
    let out = || get("--out").map(PathBuf::from);

    match command.as_str() {
        "boundary" => {
            let evidence = thermo::boundary::solve(l()?, k()?, chi()?);
            write_json(out()?, &evidence)?;
            println!(
                "boundary L={} K={} chi={} g={:.15} time_ms={:.3}",
                evidence.l, evidence.k, evidence.chi, evidence.g, evidence.solver_time_ms
            );
        }
        "trg" => {
            let evidence = thermo::trg::solve(l()?, k()?, chi()?);
            write_json(out()?, &evidence)?;
            println!(
                "trg L={} K={} chi={} g={:.15} time_ms={:.3}",
                evidence.l, evidence.k, evidence.chi, evidence.g, evidence.solver_time_ms
            );
        }
        "svd" => {
            let size = get("--size")
                .and_then(|value| value.parse::<usize>().map_err(|_| "--size invalid".into()))?;
            let seed = get("--seed")
                .and_then(|value| value.parse::<u64>().map_err(|_| "--seed invalid".into()))?;
            let evidence = thermo::svd::generate(size, seed, &[16, 64, 128, 256, 512]);
            write_json(out()?, &evidence)?;
            println!("svd size={size} seed={seed}");
        }
        "exact" => {
            let l = l()?;
            let k = k()?;
            let periodic = get("--boundary")? == "periodic";
            let value = json!({
                "model": "zero-field square-lattice Ising; H=-sum_edges s_i*s_j",
                "geometry": "square",
                "boundary": if periodic {"periodic"} else {"open"},
                "l": l,
                "k": k,
                "log_z": thermo::ising::exact_log_z(k, l, periodic),
                "g": thermo::ising::exact_log_z(k, l, periodic) / (l * l) as f64,
                "energy_variance": thermo::ising::exact_energy_variance(k, l, periodic),
                "specific_heat": thermo::ising::exact_specific_heat(k, l, periodic)
            });
            write_json(out()?, &value)?;
            println!("exact L={l} K={k}");
        }
        _ => return Err(usage()),
    }
    Ok(())
}

fn write_json(path: PathBuf, value: &impl serde::Serialize) -> Result<(), String> {
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    let target = path.join("evidence.json");
    fs::write(
        &target,
        serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn usage() -> String {
    "usage:\n  thermo boundary --l L --k K --chi CHI --out DIR\n  thermo trg --l L --k K --chi CHI --out DIR\n  thermo svd --size N --seed SEED --out DIR\n  thermo exact --l L --k K --boundary open|periodic --out DIR".into()
}
