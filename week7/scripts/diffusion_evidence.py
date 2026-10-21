"""Draw Part 1 evidence from trained checkpoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from diffusion_lib import (
    ALPHAS,
    BETAS,
    CENTERS,
    ROOT,
    T,
    ddpm_sample,
    load_params,
    sample_target,
    summarize,
    write_json,
)

ART = ROOT / "artifacts" / "diffusion"
EVIDENCE = ROOT / "evidence"
CHECKPOINTS = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]


def frame_image(fig) -> Image.Image:
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def style(axis) -> None:
    axis.set_xlim(-4.5, 4.5)
    axis.set_ylim(-4.5, 4.5)
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)


def forward_movie() -> None:
    key_data, key_noise = jax.random.split(jax.random.key(7001))
    x0 = sample_target(key_data, 2000)
    noise = jax.random.normal(key_noise, (T,) + x0.shape, jnp.float32)

    @jax.jit
    def forward(x, noise_sequence):
        def body(state, values):
            step_noise, alpha, beta = values
            next_state = jnp.sqrt(alpha) * state + jnp.sqrt(beta) * step_noise
            return next_state, next_state

        return jax.lax.scan(body, x, (noise_sequence, ALPHAS, BETAS))[1]

    path = np.asarray(forward(x0, noise))
    levels = np.unique(np.rint(np.linspace(0, T, 51)).astype(int))
    frames = []
    metrics = {}
    for level in levels:
        xt = np.asarray(x0) if level == 0 else path[int(level) - 1]
        fig, axis = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
        axis.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.35, color="#287271")
        style(axis)
        axis.set_title(f"Forward diffusion, t = {level}")
        if level == 0:
            axis.scatter(
                np.asarray(CENTERS)[:, 0],
                np.asarray(CENTERS)[:, 1],
                marker="+",
                s=80,
                color="#c73e1d",
            )
        frames.append(frame_image(fig))
        plt.close(fig)
        if level == T:
            metrics = {
                "mean": np.mean(xt, axis=0).tolist(),
                "variance": np.var(xt, axis=0).tolist(),
            }
    frames[0].save(
        EVIDENCE / "forward-diffusion.gif",
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        disposal=2,
        optimize=False,
    )
    assert np.max(np.abs(metrics["mean"])) <= 0.1
    assert np.max(np.abs(np.asarray(metrics["variance"]) - 1.0)) <= 0.1
    write_json(ART / "forward-metrics.json", metrics)
    print(
        f"t=1000 coordinate mean: x0={metrics['mean'][0]:.6f}, "
        f"x1={metrics['mean'][1]:.6f}; variance: "
        f"x0={metrics['variance'][0]:.6f}, x1={metrics['variance'][1]:.6f}"
    )


def training_movie() -> None:
    losses = np.load(ART / "loss.npy")
    key_start, key_noise = jax.random.split(jax.random.key(7002))
    starts = sample_target(key_start, 10)
    levels = np.arange(T, -1, -1)
    frames = []
    for update in CHECKPOINTS:
        params = load_params(ART / "checkpoints" / f"{update:06d}.npz")
        paths = np.asarray(ddpm_sample(params, starts, levels, key_noise, return_path=True)[1])
        paths = np.concatenate([np.asarray(starts)[None], paths], axis=0)
        fig, (axis, loss_axis) = plt.subplots(
            1, 2, figsize=(9.2, 4.4), gridspec_kw={"width_ratios": [1, 1.25]}
        )
        for path in np.moveaxis(paths, 0, 1):
            axis.plot(path[:, 0], path[:, 1], linewidth=0.7, alpha=0.75)
            axis.scatter(path[0, 0], path[0, 1], s=8, color="black")
            axis.scatter(path[-1, 0], path[-1, 1], s=12)
        for center in np.asarray(CENTERS):
            axis.add_patch(plt.Circle(center, 0.6, fill=False, linestyle="--", alpha=0.45))
        style(axis)
        axis.set_title(f"Shared paths at update {update}")
        if update:
            visible = losses[:update]
            loss_axis.plot(np.arange(1, update + 1), visible, linewidth=0.8)
            loss_axis.set_xlim(0, 25000)
            loss_axis.set_ylim(
                max(0.05, float(np.min(visible)) * 0.8),
                max(1.2, float(np.percentile(visible, 99)) * 1.1),
            )
            loss_axis.set_yscale("log")
            loss_axis.set_xlabel("update")
            loss_axis.set_ylabel("noise-prediction loss")
        else:
            loss_axis.text(0.5, 0.5, "initialization: no updates yet", ha="center", va="center")
            loss_axis.set_axis_off()
        loss_axis.set_title("training loss so far")
        frames.append(frame_image(fig))
        plt.close(fig)
    frames[0].save(
        EVIDENCE / "training-progress.gif",
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        disposal=2,
        optimize=False,
    )


def final_distribution() -> None:
    params = load_params(ART / "checkpoints" / "025000.npz")
    key_start, key_target = jax.random.split(jax.random.key(7003))
    starts = jax.random.normal(key_start, (2000, 2), jnp.float32)
    levels = np.arange(T, -1, -1)
    samples, path = ddpm_sample(params, starts, levels, jax.random.key(7004), return_path=True)
    exact = sample_target(key_target, 2000)
    samples = np.asarray(samples)
    summary = summarize(samples)
    assert summary["max_population_error"] <= 0.04
    assert summary["max_width_error"] <= 0.03
    assert summary["max_centre_error"] <= 0.05
    write_json(ART / "final-distribution.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    axes[0].scatter(np.asarray(exact)[:, 0], np.asarray(exact)[:, 1], s=3, alpha=0.35)
    axes[0].set_title("exact target, 2000 samples")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.35, color="#287271")
    axes[1].set_title("DDPM, 1000 steps, 2000 samples")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(-4.5, 4.5)
        axis.set_ylim(-4.5, 4.5)
        axis.grid(alpha=0.15)
    fig.savefig(EVIDENCE / "ddpm-final.png", dpi=180)
    plt.close(fig)

    levels_shown = np.rint(np.linspace(T, 0, 50)).astype(int)
    path_np = np.concatenate([np.asarray(starts)[None], np.asarray(path)], axis=0)
    frames = []
    for level in levels_shown:
        state = path_np[T - level]
        fig, axis = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
        axis.scatter(state[:, 0], state[:, 1], s=2, alpha=0.3, color="#287271")
        style(axis)
        axis.set_title(f"DDPM reverse process, t = {level}")
        frames.append(frame_image(fig))
        plt.close(fig)
    frames[0].save(
        EVIDENCE / "ddpm-reverse.gif",
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(
        "population error", summary["max_population_error"],
        "centre error", summary["max_centre_error"],
        "width error", summary["max_width_error"],
    )


def main() -> None:
    EVIDENCE.mkdir(exist_ok=True)
    forward_movie()
    training_movie()
    final_distribution()


if __name__ == "__main__":
    main()
