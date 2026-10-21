"""Generate four CIFAR-10 photographs with the fixed pretrained U-Net."""

from __future__ import annotations

import hashlib
import importlib.util
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import ALPHA_BARS, BETAS, ROOT, T, write_json
from unet_jax import load_unet_params, unet_forward

MODEL_DIR = ROOT / "inputs" / "cifar-model"
WEEK6_DIR = ROOT.parent / "week6"
OUT = ROOT / "artifacts" / "cifar"
EVIDENCE = ROOT / "evidence"
REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def cifar_real_images() -> np.ndarray:
    source = WEEK6_DIR / "scripts" / "datasets.py"
    archive = WEEK6_DIR / "inputs" / "cifar10" / "cifar-10-binary.tar.gz"
    if not source.is_file():
        raise RuntimeError(f"Week 6 image-gallery reader is missing: {source}")
    if not archive.is_file():
        raise FileNotFoundError(
            f"Week 6 CIFAR-10 input is missing: {archive}; "
            "run week7/scripts/download_inputs.sh"
        )
    spec = importlib.util.spec_from_file_location("week6_datasets", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Week 6 image-gallery reader: {source}")
    datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datasets)
    datasets.CIFAR_TAR = str(archive)
    x_train, _, _, _ = datasets.read_cifar10()
    return x_train[:4].astype(np.float32) / 255.0


def generate(params) -> tuple[np.ndarray, np.ndarray, float]:
    key_noise, key_steps = jax.random.split(jax.random.key(7200))
    starts = jax.random.normal(key_noise, (4, 32, 32, 3), jnp.float32)
    levels = jnp.arange(T, -1, -1, dtype=jnp.int32)
    noise = jax.random.normal(key_steps, (T, 4, 32, 32, 3), jnp.float32)

    @jax.jit
    def sample(x0, zseq):
        def body(x, step):
            t, tp, z = step
            eps = unet_forward(params, x, jnp.full((4,), t - 1, jnp.int32))
            ab_t, ab_tp = ALPHA_BARS[t], ALPHA_BARS[tp]
            a = ab_t / ab_tp
            mean = (x - (1.0 - a) / jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(a)
            variance = jnp.where(tp > 0, 1.0 - a, 0.0)
            next_x = mean + jnp.sqrt(variance) * z
            return next_x, next_x

        x, path = jax.lax.scan(body, x0, (levels[:-1], levels[1:], zseq))
        return x, path

    started = time.perf_counter()
    samples, path = sample(starts, noise)
    samples.block_until_ready()
    elapsed = time.perf_counter() - started
    full_path = np.concatenate([np.asarray(starts)[None], np.asarray(path)], axis=0)
    return np.asarray(samples), full_path, elapsed


def figure(samples: np.ndarray, path: np.ndarray) -> None:
    shown_t = [1000, 500, 200, 100, 50, 0]
    columns = shown_t
    fig, axes = plt.subplots(5, len(columns), figsize=(10.5, 8.5), constrained_layout=True)
    for row in range(4):
        for column, level in enumerate(shown_t):
            state = path[T - level, row]
            image = np.clip((state + 1.0) / 2.0, 0.0, 1.0)
            axes[row, column].imshow(image, interpolation="none")
            axes[row, column].set_axis_off()
        axes[row, 0].set_ylabel(f"generated {row + 1}", rotation=90)
    real = cifar_real_images()
    for column in range(len(columns)):
        axes[4, column].set_axis_off()
    for index in range(4):
        axes[4, index].imshow(real[index], interpolation="none")
    for column in range(4, len(columns)):
        axes[4, column].set_visible(False)
    axes[4, 0].text(
        0,
        1.2,
        "real CIFAR-10 examples",
        transform=axes[4, 0].transAxes,
        fontsize=10,
    )
    for column, level in enumerate(shown_t):
        axes[0, column].set_title(f"t={level}")
    fig.suptitle("fixed downloaded predictor + JAX 1000-step DDPM")
    fig.savefig(EVIDENCE / "cifar-generation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    params, config = load_unet_params(MODEL_DIR)
    samples, path, elapsed = generate(params)
    np.save(OUT / "samples.npy", samples)
    np.save(OUT / "reverse-path.npy", path)
    figure(samples, path)
    payload = {
        "model": "google/ddpm-cifar10-32",
        "revision": REVISION,
        "model_sha256": digest(MODEL_DIR / "diffusion_pytorch_model.safetensors"),
        "config_sha256": digest(MODEL_DIR / "config.json"),
        "config": config,
        "sampler": "DDPM",
        "mathematical_levels": [T, 0],
        "checkpoint_index": "t-1",
        "network_evaluations": T,
        "samples": 4,
        "dtype": "float32",
        "elapsed_seconds": elapsed,
        "device": str(jax.devices()[0]),
        "checkpoint_tensors": len(params),
    }
    write_json(OUT / "run.json", payload)
    print(f"generated four images with {T} evaluations in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
