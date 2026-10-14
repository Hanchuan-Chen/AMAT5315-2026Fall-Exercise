"""Part 2's training run: CD-1 and CD-10 on the full MNIST training split.

Writes the raw run into the git-ignored ``artifacts/rbm/``: the two parameter
files, the settings record (the TOML plus resolved values and wall-clock time),
and the histories of the CD surrogate and the held-out one-step reconstruction
error.  The two models share the initialization seed, the data order and the
evaluation randomness, so they differ only in their chain length.
"""

from __future__ import annotations

import os
import time

import numpy as np

import datasets
import rbm_lib

OUT_DIR = "artifacts/rbm"
CHAIN_LENGTHS = {"cd1": 1, "cd10": 10}


def main() -> None:
    cfg = rbm_lib.load_config()
    x_train, _, x_test, _ = datasets.read_mnist()
    v_train = rbm_lib.binarize(x_train)
    n_eval = int(cfg["evaluation"]["test_images"])
    v_test = rbm_lib.binarize(x_test[:n_eval])
    print(
        f"binarized {v_train.shape[0]} training and {v_test.shape[0]} held-out "
        f"images; visible layer {rbm_lib.N_VISIBLE}, hidden layer {rbm_lib.N_HIDDEN}"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    runs = {}
    started = time.time()
    for label, chain_length in CHAIN_LENGTHS.items():
        run = rbm_lib.train(cfg, v_train, v_test, chain_length, label)
        run["chain_length"] = chain_length
        np.savez(
            f"{OUT_DIR}/{label}-params.npz", **rbm_lib.params_to_numpy(run["params"])
        )
        runs[label] = run
    rbm_lib.write_settings(f"{OUT_DIR}/run.json", cfg, runs)
    print(f"wrote {OUT_DIR}/ and {OUT_DIR}/run.json in {time.time() - started:.1f} s")

    for label, run in runs.items():
        history = run["history"]
        print(
            f"{label}: surrogate {history['surrogate'][1]:.4f} at epoch 1 -> "
            f"{history['surrogate'][-1]:.4f} at epoch 20; "
            f"reconstruction error {history['reconstruction_error'][-1]:.5f}"
        )


if __name__ == "__main__":
    main()
