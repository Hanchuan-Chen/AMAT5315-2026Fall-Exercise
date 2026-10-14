"""Part 4's scan: retrieval success against P for classical and modern recall.

Stores the first P MNIST training images in stored order as binarized spin
patterns, queries 100 of them (cycling through the stored ones) with the lower
half replaced by random spins, retrieves with one modern update at each beta
and with Part 3's classical recall on the same patterns, and writes the raw run
to the git-ignored ``artifacts/modern/``.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import modern_lib as M

OUT_DIR = "artifacts/modern"
QUERY_SEED = 6608
RECALL_SEED = 6609
WEIGHT_BETAS = (0.003, 0.03, 0.3)
SAMPLE_QUERIES = 6


def sample_indices(p: int) -> np.ndarray:
    """Query q targets stored pattern q mod P, cycling through the stored ones."""
    return np.arange(M.QUERIES) % p


def main() -> None:
    started = time.time()
    record = {
        "queries": M.QUERIES,
        "ps": list(M.PS),
        "betas": list(M.BETAS),
        "query_seed": QUERY_SEED,
        "recall_seed": RECALL_SEED,
        "success_rule": "overlap >= 0.95 with the target and the target is the "
                        "nearest stored digit to the signed result",
        "seconds": {},
    }
    keys = {
        "queries": jax.random.key(QUERY_SEED),
        "recall": jax.random.key(RECALL_SEED),
    }

    success_modern = np.zeros((len(M.PS), len(M.BETAS)))
    success_classical = np.zeros(len(M.PS))
    extra = {}
    max_diff = 0.0
    for i, p in enumerate(M.PS):
        p_started = time.time()
        x = jnp.asarray(M.stored_patterns(p))
        targets = jnp.asarray(sample_indices(p))
        queries = M.corrupted_queries(x, targets, keys["queries"])

        for j, beta in enumerate(M.BETAS):
            update = M.modern_update(x, queries, beta)
            attention = M.attention_update(x, queries, beta)
            max_diff = max(max_diff, float(jnp.max(jnp.abs(update - attention))))
            success_modern[i, j] = M.retrieval_success(
                x, jnp.sign(update), targets
            ).mean()

        classical = M.classical_recall(x, queries, keys["recall"])
        success_classical[i] = M.retrieval_success(
            x, jnp.sign(classical), targets
        ).mean()

        if p == 100:
            extra["weights"] = np.stack(
                [M.softmax_weights(x, queries, beta)[:, 0]
                 for beta in WEIGHT_BETAS], axis=1
            )
            extra["weight_query"] = 0
            extra["weight_target"] = int(targets[0])
            pick = np.arange(SAMPLE_QUERIES)
            extra["sample_targets"] = np.asarray(targets[pick])
            extra["sample_patterns"] = np.asarray(x[:, targets[pick]])
            extra["sample_queries"] = np.asarray(queries[:, pick])
            extra["sample_classical"] = np.asarray(classical[:, pick])
            extra["sample_modern"] = np.stack(
                [np.asarray(M.modern_update(x, queries, beta))[:, pick]
                 for beta in M.BETAS], axis=2
            )

        record["seconds"][str(p)] = time.time() - p_started
        print(
            f"P={p:5d}: modern success "
            + " ".join(f"b={beta:g}:{success_modern[i, j]:.2f}"
                       for j, beta in enumerate(M.BETAS))
            + f"; classical {success_classical[i]:.2f}; {record['seconds'][str(p)]:.1f} s"
        )

    record["attention_max_abs_difference"] = max_diff
    record["total_seconds"] = time.time() - started
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        f"{OUT_DIR}/scan.npz",
        ps=np.array(M.PS),
        betas=np.array(M.BETAS),
        success_modern=success_modern,
        success_classical=success_classical,
        weight_betas=np.array(WEIGHT_BETAS),
        **extra,
    )
    with open(f"{OUT_DIR}/run.json", "w") as handle:
        json.dump(record, handle, indent=2)
    print(
        f"largest |modern update - softmax attention| over all queries, P and "
        f"beta: {max_diff:.2e} (required below 1e-5)"
    )
    print(
        f"wrote {OUT_DIR}/scan.npz and run.json in {record['total_seconds']:.1f} s"
    )


if __name__ == "__main__":
    main()
