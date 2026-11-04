import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_evidence_files_and_summary_contracts():
    evidence = ROOT / "evidence"
    for name in [
        "graphs.svg",
        "posteriors.png",
        "joint.png",
        "orders.png",
        "bp-traces.png",
        "promedus-order-costs.png",
        "promedus-bp.png",
        "summary.json",
    ]:
        path = evidence / name
        assert path.is_file() and path.stat().st_size > 1000
    summary = json.loads((evidence / "summary.json").read_text())
    assert summary["orders"]["analysis_matches_saved"]
    assert summary["orders"]["independently_recounted"]
    numpy_checks = summary["numpy_checks"]
    a = np.asarray(numpy_checks["a"], dtype=np.float64)
    b = np.asarray(numpy_checks["b"], dtype=np.float64)
    ab = a @ b
    assert np.max(np.abs(ab - np.asarray(numpy_checks["tensor_ab"]))) < 1e-12
    assert abs(float(np.trace(ab)) - numpy_checks["tensor_trace"]) < 1e-12
    assert numpy_checks["ising_error"] < 1e-12
    assert summary["bp_asia"]["ADX"]["converged"]
    assert summary["bp_asia"]["ADX"]["max_marginal_error"] > 0.01
    assert summary["promedus"]["exact_plan_log10_difference"] < 1e-10
    assert summary["promedus"]["published_PR_error"] < 1e-4
    assert summary["promedus"]["published_MAR_max_error_exact"] < 1e-5
    assert not summary["promedus"]["bp_converged"]
