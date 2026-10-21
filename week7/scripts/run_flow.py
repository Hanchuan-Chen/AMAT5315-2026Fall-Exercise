"""Train Part 3's flow-matching velocity network."""

from __future__ import annotations

import shutil

import numpy as np

from diffusion_lib import ROOT, load_toml, save_params, train_flow, write_json


def main() -> None:
    cfg = load_toml("flow.toml")
    out = ROOT / "artifacts" / "flow"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "flow.toml", out / "flow.toml")
    params, losses, run = train_flow(cfg)
    save_params(out / "params.npz", params)
    np.save(out / "loss.npy", losses)
    write_json(out / "run.json", run)
    print(
        f"trained {run['updates']} updates in {run['elapsed_seconds']:.1f}s; "
        f"final loss {run['final_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
