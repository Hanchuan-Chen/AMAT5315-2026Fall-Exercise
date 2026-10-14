"""Part 3's sweep: the unstable-bit fraction and recall against the load.

For every ``N`` in 100, 200, 400, 800 and every load ``alpha = P/N`` from 0.05
to 0.25 in steps of 0.01, draws 20 independent pattern sets, measures the
unstable-bit fraction over all stored patterns, and recalls the first 10 stored
patterns of each set (all of them when ``P < 10``), recording the final overlap.
The raw run goes to the git-ignored ``artifacts/hopfield/``.
"""

from __future__ import annotations

import json
import time

import jax
import numpy as np

import hopfield_lib

OUT_DIR = "artifacts/hopfield"
NS = (100, 200, 400, 800)
ALPHAS = np.round(np.arange(0.05, 0.2501, 0.01), 2)
SETS = 20
SEED_ROOT = 2026


def make_patterns(n: int, p: int, alpha_index: int, set_index: int) -> np.ndarray:
    """One independent fair +-1 pattern set; its seed is recorded in run.json."""
    seed = [SEED_ROOT, n, alpha_index, set_index]
    rng = np.random.default_rng(seed)
    return (2 * rng.integers(0, 2, size=(n, p)) - 1).astype(np.float32)


def one_point(n: int, alpha_index: int, p: int) -> dict:
    patterns = [
        make_patterns(n, p, alpha_index, s) for s in range(SETS)
    ]
    unstable = np.array(
        [hopfield_lib.unstable_fraction(x)[0] for x in patterns]
    )

    targets = min(p, 10)
    x_batch = np.repeat(np.stack(patterns), targets, axis=0)
    starts = np.concatenate(
        [x[:, :targets].T for x in patterns]
    ).astype(np.float32)
    key = jax.random.key(SEED_ROOT * 10_000 + n * 10 + alpha_index)
    final, sweeps = hopfield_lib.recall(x_batch, starts, key)

    final = np.asarray(final)
    per_recall = np.sum(final * starts, axis=1) / n
    per_set = per_recall.reshape(SETS, targets).mean(axis=1)
    return {
        "unstable_mean": float(unstable.mean()),
        "unstable_se": float(unstable.std(ddof=1) / np.sqrt(SETS)),
        "overlap_mean": float(per_recall.mean()),
        "overlap_se": float(per_set.std(ddof=1) / np.sqrt(SETS)),
        "overlap_fraction_ge_0.9": float(np.mean(per_recall >= 0.9)),
        "sweeps_mean": float(np.mean(np.asarray(sweeps))),
    }


def main() -> None:
    started = time.time()
    record = {
        "seed_root": SEED_ROOT,
        "pattern_set_seeds": "default_rng([2026, N, alpha_index, set_index])",
        "recall_seed": "jax.random.key(2026 * 10_000 + N * 10 + alpha_index)",
        "sets": SETS,
        "max_sweeps": 200,
        "alphas": ALPHAS.tolist(),
        "n": list(NS),
        "points": {},
        "seconds": {},
    }
    arrays: dict[str, list] = {}
    for n in NS:
        rows = []
        n_started = time.time()
        for alpha_index, alpha in enumerate(ALPHAS):
            p = int(round(alpha * n))
            rows.append(one_point(n, alpha_index, p))
            print(
                f"N={n:4d} alpha={alpha:.2f} P={p:3d}: unstable "
                f"{rows[-1]['unstable_mean']:.3e}+/-{rows[-1]['unstable_se']:.1e}, "
                f"mean overlap {rows[-1]['overlap_mean']:.4f}, "
                f"fraction >= 0.9 {rows[-1]['overlap_fraction_ge_0.9']:.2f}, "
                f"sweeps {rows[-1]['sweeps_mean']:.1f}"
            )
        record["seconds"][str(n)] = time.time() - n_started
        record["points"][str(n)] = rows
        keys = rows[0].keys()
        arrays[f"n{n}"] = np.array([[row[k] for k in keys] for row in rows])
    record["total_seconds"] = time.time() - started

    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        f"{OUT_DIR}/scan.npz",
        alphas=ALPHAS,
        ns=np.array(NS),
        columns=np.array(["unstable_mean", "unstable_se", "overlap_mean",
                          "overlap_se", "overlap_fraction_ge_0.9", "sweeps_mean"]),
        **{name: value for name, value in arrays.items()},
    )
    with open(f"{OUT_DIR}/run.json", "w") as handle:
        json.dump(record, handle, indent=2)
    per_n = ", ".join(
        "N=%d: %.1f s" % (n, record["seconds"][str(n)]) for n in NS
    )
    print(
        f"wrote {OUT_DIR}/scan.npz and run.json in "
        f"{record['total_seconds']:.1f} s ({per_n})"
    )


if __name__ == "__main__":
    main()
