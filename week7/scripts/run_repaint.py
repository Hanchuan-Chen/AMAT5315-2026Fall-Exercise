"""Run RePaint on the four supplied CIFAR-10 test images."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import ALPHA_BARS, ROOT, write_json
from unet_jax import load_unet_params, unet_forward

MODEL_DIR = ROOT / "inputs" / "cifar-model"
OUT = ROOT / "artifacts" / "repaint"
EVIDENCE = ROOT / "evidence"
SEEDS = [7201, 7202, 7203, 7204]
LEVELS = np.arange(1000, -1, -20, dtype=np.int32)


def masks_and_observation(images_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(images_nhwc.shape, np.float32)
    mask[:, 6:26, 6:26, :] = 0.0
    return mask, images_nhwc * mask


def proposal_mean_variance(
    current: jax.Array, clean: jax.Array, t: jax.Array, tp: jax.Array
) -> tuple[jax.Array, jax.Array]:
    ab_t, ab_tp = ALPHA_BARS[t], ALPHA_BARS[tp]
    a = ab_t / ab_tp
    mean = (
        jnp.sqrt(ab_tp) * (1.0 - a) / (1.0 - ab_t) * clean
        + jnp.sqrt(a) * (1.0 - ab_tp) / (1.0 - ab_t) * current
    )
    variance = (1.0 - ab_tp) * (1.0 - a) / (1.0 - ab_t)
    return mean, variance


def steps() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, next_times, retries = [], [], []
    for index in range(len(LEVELS) - 1):
        attempts = 2 if index + 1 < len(LEVELS) - 1 else 1
        for attempt in range(attempts):
            times.append(LEVELS[index])
            next_times.append(LEVELS[index + 1])
            retries.append(attempt > 0)
    return (
        np.asarray(times, np.int32),
        np.asarray(next_times, np.int32),
        np.asarray(retries, bool),
    )


def repaint(params, observation, mask, key: jax.Array) -> tuple[np.ndarray, int]:
    times, next_times, retries = steps()
    key_initial, key_noise = jax.random.split(key)
    initial = jax.random.normal(key_initial, observation.shape, jnp.float32)
    noise = jax.random.normal(
        key_noise, (len(times), 3) + observation.shape, jnp.float32
    )
    obs = jnp.asarray(observation)
    keep = jnp.asarray(mask)

    @jax.jit
    def run(x, zseq):
        def body(current, record):
            t, tp, retry, z = record
            f, xi, eta = z[0], z[1], z[2]
            ab_t, ab_tp = ALPHA_BARS[t], ALPHA_BARS[tp]
            a = ab_t / ab_tp
            current = jnp.where(
                retry,
                jnp.sqrt(a) * current + jnp.sqrt(1.0 - a) * f,
                current,
            )
            eps = unet_forward(params, current, jnp.full((4,), t - 1, jnp.int32))
            clean = (current - jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(ab_t)
            clean = jnp.clip(clean, -1.0, 1.0)
            mean, variance = proposal_mean_variance(current, clean, t, tp)
            proposal = mean + jnp.sqrt(variance) * xi
            observed_at_tp = jnp.sqrt(ab_tp) * obs + jnp.sqrt(1.0 - ab_tp) * eta
            merged = keep * observed_at_tp + (1.0 - keep) * proposal
            return merged, None

        return jax.lax.scan(body, x, (times, next_times, retries, zseq))[0]

    return np.asarray(run(initial, noise)), len(times)


def display(image: np.ndarray) -> np.ndarray:
    return np.clip((image + 1.0) / 2.0, 0.0, 1.0)


def draw(
    truth: np.ndarray,
    observation: np.ndarray,
    completions: np.ndarray,
    metrics: dict,
) -> None:
    variance = np.var(completions, axis=1).mean(axis=-1)
    fig, axes = plt.subplots(
        len(truth), 8, figsize=(18, 9.5), constrained_layout=True
    )
    for image_index in range(len(truth)):
        axes[image_index, 0].imshow(display(truth[image_index]), interpolation="none")
        axes[image_index, 0].set_title("truth")
        axes[image_index, 1].imshow(display(observation[image_index]), interpolation="none")
        axes[image_index, 1].set_title("observation")
        for seed_index, seed in enumerate(SEEDS):
            axis = axes[image_index, 2 + seed_index]
            axis.imshow(display(completions[image_index, seed_index]), interpolation="none")
            mse = metrics["hidden_mse"][image_index][seed_index]
            axis.set_title(f"seed {seed}\nMSE {mse:.3f}")
        axis = axes[image_index, 6]
        image = axis.imshow(variance[image_index], cmap="magma", interpolation="none")
        axis.set_title("pixelwise variance")
        fig.colorbar(image, ax=axis, fraction=0.046)
        axis = axes[image_index, 7]
        axis.axis("off")
        observed_max = metrics["observed_max_difference"][image_index]
        axis.text(
            0.0,
            0.5,
            "observed max change\n"
            + "\n".join(f"{value:.2e}" for value in observed_max),
            fontsize=9,
            va="center",
        )
        for axis in axes[image_index, :7]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("RePaint: 49 jumps, two attempts except the final, 99 evaluations")
    fig.savefig(EVIDENCE / "repaint.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    data = np.load(ROOT / "inputs" / "images.npz")
    truth = data["test"].transpose(0, 2, 3, 1).astype(np.float32)
    mask_np, observation_np = masks_and_observation(truth)
    params, _ = load_unet_params(MODEL_DIR)
    completions = []
    evaluations = []
    started = time.perf_counter()
    for seed in SEEDS:
        run, count = repaint(
            params, observation_np, mask_np, jax.random.key(seed)
        )
        completions.append(run)
        evaluations.append(count)
    completions = np.stack(completions, axis=1)
    assert evaluations == [99] * len(SEEDS)
    hidden_mse = []
    observed_max_difference = []
    for image_index in range(4):
        hidden_mse.append([])
        observed_max_difference.append([])
        for seed_index in range(len(SEEDS)):
            completion = completions[image_index, seed_index]
            hidden_mse[-1].append(
                float(np.mean((completion[6:26, 6:26] - truth[image_index, 6:26, 6:26]) ** 2))
            )
            observed_max_difference[-1].append(
                float(np.max(np.abs(completion[mask_np[image_index] == 1] - truth[image_index][mask_np[image_index] == 1])))
            )
    metrics = {
        "hidden_mse": hidden_mse,
        "observed_max_difference": observed_max_difference,
        "mean_hidden_mse": float(np.mean(hidden_mse)),
        "maximum_observed_difference": float(np.max(observed_max_difference)),
        "hidden_variance_mean": float(
            np.mean(np.var(completions, axis=1)[:, 6:26, 6:26, :])
        ),
        "network_evaluations": evaluations,
        "seeds": SEEDS,
        "levels": LEVELS.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    assert metrics["maximum_observed_difference"] < 1e-5
    assert metrics["hidden_variance_mean"] > 0.0
    for image_index in range(4):
        for seed_index, seed in enumerate(SEEDS):
            print(
                f"image {image_index + 1} seed {seed}: hidden MSE "
                f"{hidden_mse[image_index][seed_index]:.6f}; "
                f"observed max change "
                f"{observed_max_difference[image_index][seed_index]:.3e}"
            )
    np.savez(
        OUT / "completions.npz",
        completions=completions,
        truth=truth,
        observation=observation_np,
        mask=mask_np,
    )
    write_json(OUT / "metrics.json", metrics)
    draw(truth, observation_np, completions, metrics)
    print(
        f"mean hidden MSE {metrics['mean_hidden_mse']:.6f}; "
        f"max observed difference {metrics['maximum_observed_difference']:.3e}; "
        f"mean hidden variance {metrics['hidden_variance_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
