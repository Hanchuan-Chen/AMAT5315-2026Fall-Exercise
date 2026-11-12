#!/usr/bin/env python3
"""Run the Week 10 calculations and then the independent analysis."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
COARSE_K = [round(0.40 + 0.0025 * i, 8) for i in range(41)]
FINE_H = 0.00125


def load_analyzer():
    spec = importlib.util.spec_from_file_location("w10_analyze", ROOT / "scripts" / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def default_thermo() -> str:
    explicit = os.environ.get("THERMO")
    if explicit:
        return explicit
    installed = shutil.which("thermo")
    if installed:
        return installed
    local = ROOT / "target" / "release" / "thermo"
    if local.exists():
        return str(local)
    raise SystemExit("thermo not found; run cargo build --release or set THERMO")


def execute(thermo: str, args: list[str], output: Path, force: bool) -> str:
    evidence = output / "evidence.json"
    if evidence.exists() and not force:
        return f"skip {output.relative_to(ROOT)}"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [thermo, *args, "--out", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return f"done {output.relative_to(ROOT)}"


def run_parallel(thermo: str, jobs: list[tuple[list[str], Path]], force: bool) -> None:
    unique_jobs = []
    seen = set()
    for command, output in jobs:
        key = str(output)
        if key not in seen:
            seen.add(key)
            unique_jobs.append((command, output))
    jobs = unique_jobs
    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(execute, thermo, args, output, force): output
            for args, output in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{index}/{len(jobs)}] {future.result()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermo", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    thermo = args.thermo or default_thermo()
    analyzer = load_analyzer()

    execute(
        thermo,
        ["svd", "--size", "512", "--seed", "2026"],
        RUNS / "svd",
        args.force,
    )

    checks = []
    for k in (0.0, 0.44, 0.7):
        checks.append((["exact", "--l", "4", "--k", str(k), "--boundary", "open"],
                       RUNS / "checks" / "exact" / "open4" / f"K{k:.5f}"))
        checks.append((["boundary", "--l", "4", "--k", str(k), "--chi", "8"],
                       RUNS / "checks" / "boundary" / "open4" / f"K{k:.5f}"))
    checks.append((["exact", "--l", "2", "--k", "0.44", "--boundary", "periodic"],
                   RUNS / "checks" / "exact" / "periodic2" / "K0.44000"))
    checks.append((["exact", "--l", "4", "--k", "0.44", "--boundary", "periodic"],
                   RUNS / "checks" / "exact" / "periodic4" / "K0.44000"))
    for k in (0.4375, 0.44, 0.4425):
        checks.append((["boundary", "--l", "4", "--k", str(k), "--chi", "8"],
                       RUNS / "checks" / "boundary" / "open4" / f"K{k:.5f}"))
        checks.append((["trg", "--l", "2", "--k", str(k), "--chi", "16"],
                       RUNS / "checks" / "trg" / "periodic2" / f"K{k:.5f}"))
    checks.append((["trg", "--l", "2", "--k", "0.44", "--chi", "16"],
                   RUNS / "checks" / "trg" / "periodic2" / "K0.44000"))
    checks.append((["trg", "--l", "4", "--k", "0.44", "--chi", "8"],
                   RUNS / "checks" / "trg" / "periodic4" / "K0.44000"))
    run_parallel(thermo, checks, args.force)

    boundary_coarse = [
        (["boundary", "--l", str(length), "--k", str(k), "--chi", "16"],
         analyzer.run_dir("boundary", length, 16, k))
        for length in (16, 32, 64)
        for k in COARSE_K
    ]
    boundary_coarse.append(
        (["boundary", "--l", "64", "--k", "0.45", "--chi", "16"],
         RUNS / "boundary64")
    )
    run_parallel(thermo, boundary_coarse, args.force)

    trg_coarse = [
        (["trg", "--l", str(length), "--k", str(k), "--chi", "16"],
         analyzer.run_dir("trg", length, 16, k))
        for length in (64, 256, 1024)
        for k in COARSE_K
    ]
    trg_coarse.append(
        (["trg", "--l", "1024", "--k", "0.44", "--chi", "16"],
         RUNS / "trg1024")
    )
    run_parallel(thermo, trg_coarse, args.force)

    refine_jobs = []
    for method, lengths in (("boundary", (16, 32, 64)), ("trg", (64, 256, 1024))):
        for length in lengths:
            center = analyzer.coarse_peak(method, length)
            for k in analyzer.refinement_k(center):
                for chi in (16, 32):
                    refine_jobs.append(
                        ([method, "--l", str(length), "--k", str(float(k)),
                          "--chi", str(chi)],
                         analyzer.run_dir(method, length, chi, float(k)))
                    )
    run_parallel(thermo, refine_jobs, args.force)

    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py"), "--check"],
                   check=True)


if __name__ == "__main__":
    main()

