"""Produce Part 3 velocity and step-count comparisons."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import (
    ROOT,
    ddpm_sample,
    ddim_sample,
    energy_distance,
    exact_velocity,
    flow_sample,
    full_grid,
    load_params,
    mlp_forward,
    sample_target,
    summarize,
    write_json,
)

ART_DIFFUSION = ROOT / "artifacts" / "diffusion"
ART_FLOW = ROOT / "artifacts" / "flow"
EVIDENCE = ROOT / "evidence"
STEPS = [1, 2, 5, 10, 20, 50, 100]


def relative_field_error(params, s: float, points) -> float:
    x = jnp.asarray(points, jnp.float32)
    learned = mlp_forward(
        params, x, jnp.full((len(points),), s, jnp.float32), flow=True
    )
    exact = exact_velocity(x, s)
    return float(
        jnp.sqrt(jnp.sum((learned - exact) ** 2) / jnp.sum(exact**2))
    )


def velocity_figure(params) -> dict:
    key_x0, key_z = jax.random.split(jax.random.key(7101))
    x0 = sample_target(key_x0, 4000)
    z = jax.random.normal(key_z, x0.shape, jnp.float32)
    times = [0.1, 0.5, 0.9]
    errors = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, s in zip(axes, times):
        xs = (1.0 - s) * x0 + s * z
        errors[str(s)] = relative_field_error(params, s, xs)
        grid = np.linspace(-3.5, 3.5, 17)
        gx, gy = np.meshgrid(grid, grid)
        points = np.stack([gx.ravel(), gy.ravel()], axis=1)
        exact = np.asarray(exact_velocity(jnp.asarray(points), s))
        learned = np.asarray(
            mlp_forward(
                params,
                jnp.asarray(points),
                jnp.full((len(points),), s, jnp.float32),
                flow=True,
            )
        )
        scale = max(np.linalg.norm(exact, axis=1).max(), np.linalg.norm(learned, axis=1).max())
        axis.quiver(gx, gy, exact[:, 0], exact[:, 1], color="#287271", alpha=0.72)
        axis.quiver(gx, gy, learned[:, 0], learned[:, 1], color="#c73e1d", alpha=0.62)
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)
        axis.set_title(f"s={s}: relative RMS {errors[str(s)]:.4f}")
    fig.suptitle("exact velocity (teal) and learned velocity (red)")
    fig.savefig(EVIDENCE / "flow-velocity-fields.png", dpi=180)
    plt.close(fig)
    assert max(errors.values()) < 0.1
    return errors


def sampler_scan() -> dict:
    diffusion = load_params(ART_DIFFUSION / "checkpoints" / "025000.npz")
    flow = load_params(ART_FLOW / "params.npz")
    key_target, key_start, key_ddpm = jax.random.split(jax.random.key(7102), 3)
    exact = np.asarray(sample_target(key_target, 4000))
    starts = jax.random.normal(key_start, (4000, 2), jnp.float32)
    records = {"ddpm": {}, "ddim": {}, "flow": {}}
    for index, steps in enumerate(STEPS):
        key_step = jax.random.fold_in(key_ddpm, index)
        grid = full_grid(steps)
        for name, samples in (
            ("ddpm", ddpm_sample(diffusion, starts, grid, key_step)),
            ("ddim", ddim_sample(diffusion, starts, grid)),
            ("flow", flow_sample(flow, starts, steps)),
        ):
            array = np.asarray(samples)
            summary = summarize(array)
            summary["energy_distance"] = energy_distance(array, exact)
            summary["energy_distance_plot"] = max(summary["energy_distance"], 1e-4)
            summary["steps"] = steps
            records[name][str(steps)] = summary
    for index, steps in enumerate((1000,), start=20):
        grid = full_grid(steps)
        key_step = jax.random.fold_in(key_ddpm, index)
        for name, samples in (
            ("ddpm", ddpm_sample(diffusion, starts, grid, key_step)),
            ("ddim", ddim_sample(diffusion, starts, grid)),
        ):
            array = np.asarray(samples)
            summary = summarize(array)
            summary["energy_distance"] = energy_distance(array, exact)
            summary["energy_distance_plot"] = max(summary["energy_distance"], 1e-4)
            summary["steps"] = steps
            records[name][str(steps)] = summary

    null_distances = []
    for pair in range(20):
        key_a, key_b = jax.random.split(jax.random.fold_in(jax.random.key(7103), pair))
        null_distances.append(
            energy_distance(
                np.asarray(sample_target(key_a, 4000)),
                np.asarray(sample_target(key_b, 4000)),
            )
        )
    null_quantile = float(np.quantile(null_distances, 0.95))
    fraction_band = float(2 * np.sqrt(0.25 * 0.75 / 4000))
    payload = {
        "records": records,
        "null_distances": null_distances,
        "null_95_quantile": null_quantile,
        "fraction_two_se": fraction_band,
    }
    write_json(ART_FLOW / "sampler-comparison.json", payload)

    flow20 = records["flow"]["20"]
    assert flow20["energy_distance"] < 0.005
    assert flow20["max_width_error"] < 0.03

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    colors = {"ddpm": "#c73e1d", "ddim": "#287271", "flow": "#7a5195"}
    for name in ("ddpm", "ddim", "flow"):
        ordered = sorted(records[name].values(), key=lambda item: item["steps"])
        steps = [item["steps"] for item in ordered]
        axes[0].plot(steps, [item["energy_distance_plot"] for item in ordered], marker="o", label=name.upper(), color=colors[name])
        axes[1].plot(steps, [item["max_population_error"] for item in ordered], marker="o", label=name.upper(), color=colors[name])
        axes[2].plot(steps, [item["max_width_error"] for item in ordered], marker="o", label=name.upper(), color=colors[name])
    axes[0].axhspan(0, null_quantile, color="0.75", alpha=0.35, label="null 95%")
    axes[1].axhspan(0, fraction_band, color="0.75", alpha=0.35, label="2 binomial SE")
    axes[2].axhspan(0, 0.03, color="0.75", alpha=0.35, label="0.03 tolerance")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("energy distance")
    axes[1].set_ylabel("largest cluster-fraction error")
    axes[2].set_ylabel("largest width error")
    for axis in axes:
        axis.axvline(20, color="black", linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel("network evaluations")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    fig.suptitle("step count versus distribution error")
    fig.savefig(EVIDENCE / "sampler-comparison.png", dpi=180)
    plt.close(fig)
    print(
        f"null95={null_quantile:.6f}; flow20 ED={flow20['energy_distance']:.6f}, "
        f"fraction={flow20['max_population_error']:.6f}, "
        f"width={flow20['max_width_error']:.6f}"
    )
    return payload


def main() -> None:
    EVIDENCE.mkdir(exist_ok=True)
    flow = load_params(ART_FLOW / "params.npz")
    errors = velocity_figure(flow)
    write_json(ART_FLOW / "velocity-errors.json", errors)
    print("relative RMS errors", errors)
    sampler_scan()


if __name__ == "__main__":
    main()
