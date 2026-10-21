"""Train Part 1's noise predictor and persist all raw results."""

from __future__ import annotations

import shutil

import numpy as np

from diffusion_lib import ROOT, load_toml, save_params, train_noise, write_json


def main() -> None:
    cfg = load_toml("diffusion.toml")
    out = ROOT / "artifacts" / "diffusion"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "diffusion.toml", out / "diffusion.toml")
    _, losses, checkpoints, run = train_noise(cfg)
    np.save(out / "loss.npy", losses)
    for update, params in checkpoints.items():
        save_params(out / "checkpoints" / f"{int(update):06d}.npz", params)
    write_json(out / "run.json", run)
    print(
        f"trained {run['updates']} updates in {run['elapsed_seconds']:.1f}s; "
        f"final loss {run['final_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
