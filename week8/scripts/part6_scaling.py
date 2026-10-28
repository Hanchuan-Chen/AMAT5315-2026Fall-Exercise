from __future__ import annotations

import csv
import io
import json
import math

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA, EVIDENCE, load_settings, write_json

PARAMETER_NAMES = ("E", "A", "B", "alpha", "beta")


def read_runs() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    text = (DATA / "chinchilla-runs.csv").read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    n_values, compute_values, losses = [], [], []
    for row in reader:
        n_values.append(float(row["Model Size"]))
        compute_values.append(float(row["Training FLOP"]))
        losses.append(float(row["loss"]))
    n = np.asarray(n_values)
    compute = np.asarray(compute_values)
    loss = np.asarray(losses)
    downloaded_rows = len(loss)
    keep = np.argsort(loss)[:-5]
    keep.sort()
    return n[keep], compute[keep], loss[keep], downloaded_rows


def law(parameters: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    e, a, b, alpha, beta = parameters
    return e + a * n ** (-alpha) + b * d ** (-beta)


def objective_and_gradient_log(
    parameters: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    log_loss: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Huber fit in the replication's log(A), log(B), log(E) variables."""
    log_a, log_b, log_e, alpha, beta = parameters
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a = np.exp(log_a)
        b = np.exp(log_b)
        e = np.exp(log_e)
        n_term = a * n ** (-alpha)
        d_term = b * d ** (-beta)
        predicted = e + n_term + d_term
        if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
            return 1e100, np.full(5, 1e50)
        log_predicted = np.log(predicted)
        residual = log_predicted - log_loss
        absolute = np.abs(residual)
        loss = float(
            np.sum(
                np.where(
                    absolute <= delta,
                    0.5 * residual**2,
                    delta * (absolute - 0.5 * delta),
                )
            )
        )
        factor = (
            np.where(absolute <= delta, residual, delta * np.sign(residual))
            / predicted
        )
        gradient = np.stack(
            [
                n_term,
                d_term,
                np.full_like(n, e),
                -n_term * np.log(n),
                -d_term * np.log(d),
            ],
            axis=1,
        ).T @ factor
    return loss, gradient


def fit(
    n: np.ndarray,
    d: np.ndarray,
    loss: np.ndarray,
    delta: float,
    starts_log: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    y = np.log(loss)
    best_x = None
    best_loss = float("inf")
    for start in starts_log:
        result = minimize(
            lambda x: objective_and_gradient_log(x, n, d, y, delta),
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12},
        )
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_x = np.asarray(result.x)
    assert best_x is not None
    physical = np.asarray(
        [
            np.exp(best_x[2]),
            np.exp(best_x[0]),
            np.exp(best_x[1]),
            best_x[3],
            best_x[4],
        ]
    )
    return physical, best_loss


def grid_starts_log() -> list[np.ndarray]:
    starts = []
    for alpha in np.linspace(0.0, 2.0, 10):
        for e in np.linspace(0.0, 2.0, 10):
            for a in np.linspace(6.0, 30.0, 10):
                starts.append(np.asarray([a, a, e, alpha, alpha]))
    return starts


def optimal(parameters: np.ndarray, compute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, a, b, alpha, beta = parameters
    g = (alpha * a / (beta * b)) ** (1.0 / (alpha + beta))
    n_star = g * (compute / 6.0) ** (beta / (alpha + beta))
    d_star = (1.0 / g) * (compute / 6.0) ** (alpha / (alpha + beta))
    return n_star, d_star


def bootstrap(
    n: np.ndarray,
    d: np.ndarray,
    loss: np.ndarray,
    best: np.ndarray,
    delta: float,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    results = np.empty((count, 5), dtype=np.float64)
    for index in range(count):
        sample = rng.integers(0, len(n), size=len(n))
        start_log = np.asarray(
            [np.log(best[1]), np.log(best[2]), np.log(best[0]), best[3], best[4]]
        )
        fitted, _ = fit(n[sample], d[sample], loss[sample], delta, [start_log])
        results[index] = fitted
    return results


def figure_runs(
    n: np.ndarray,
    compute: np.ndarray,
    d: np.ndarray,
    loss: np.ndarray,
    fitted: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    budgets = (1e19, 1e20, 1e21)
    colors = ("#2878b5", "#e67e22", "#2a9d8f")
    for budget, color in zip(budgets, colors):
        within = np.abs(compute / budget - 1.0) <= 0.15
        axes[0].scatter(n[within], loss[within], s=15, color=color, alpha=0.75, label=f"C≈{budget:.0e}")
        n_curve = np.logspace(math.log10(n.min()), math.log10(n.max()), 300)
        d_curve = budget / (6.0 * n_curve)
        axes[0].plot(n_curve, law(fitted, n_curve, d_curve), color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("model parameters N")
    axes[0].set_ylabel("final loss L")
    axes[0].set_title("Chinchilla runs and fitted laws at fixed compute")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    predicted = law(fitted, n, d)
    scatter = axes[1].scatter(
        predicted, loss, c=np.log10(compute), s=18, cmap="viridis"
    )
    limits = [min(predicted.min(), loss.min()), max(predicted.max(), loss.max())]
    axes[1].plot(limits, limits, color="black", lw=1)
    axes[1].set_xlabel("fitted loss")
    axes[1].set_ylabel("measured loss")
    axes[1].set_title("Measured against fitted loss")
    fig.colorbar(scatter, ax=axes[1], label="log10 compute")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part6-loss-and-fit.png", dpi=180)
    plt.close(fig)


def plot_residuals(
    n: np.ndarray,
    d: np.ndarray,
    loss: np.ndarray,
    fitted: np.ndarray,
    printed: np.ndarray,
) -> tuple[float, float]:
    refit = np.log(law(fitted, n, d)) - np.log(loss)
    paper = np.log(law(printed, n, d)) - np.log(loss)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(n, refit, s=13, color="#2878b5", alpha=0.75, label="refit")
    axes[0].scatter(n, paper, s=13, color="#e67e22", alpha=0.65, label="printed")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("model parameters N")
    axes[0].set_ylabel("predicted minus measured log loss")
    axes[0].set_title("Residual against model size")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    bins = np.linspace(min(refit.min(), paper.min()), max(refit.max(), paper.max()), 26)
    axes[1].hist(refit, bins=bins, alpha=0.55, color="#2878b5", label="refit")
    axes[1].hist(paper, bins=bins, alpha=0.5, color="#e67e22", label="printed")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("log-loss residual")
    axes[1].set_ylabel("run count")
    axes[1].set_title("Residual distributions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part6-residuals.png", dpi=180)
    plt.close(fig)
    return float(refit.mean()), float(paper.mean())


def figure_allocations(
    fitted: np.ndarray,
    printed: np.ndarray,
    compute_budget: float,
    chinchilla_n: float,
    chinchilla_d: float,
) -> dict[str, float]:
    compute = np.logspace(18, 25, 300)
    refit_n, refit_d = optimal(fitted, compute)
    paper_n, paper_d = optimal(printed, compute)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.loglog(compute, refit_n, color="#2878b5", label="refit N*")
    ax.loglog(compute, refit_d, color="#2a9d8f", label="refit D*")
    ax.loglog(compute, paper_n, "--", color="#2878b5", alpha=0.7, label="printed N*")
    ax.loglog(compute, paper_d, "--", color="#2a9d8f", alpha=0.7, label="printed D*")
    chinchilla_compute = 6 * chinchilla_n * chinchilla_d
    ax.scatter(
        [chinchilla_compute, chinchilla_compute],
        [chinchilla_n, chinchilla_d],
        marker="*",
        s=180,
        color="#c0392b",
        zorder=5,
        label="Chinchilla N* and D*",
    )
    ax.set_xlabel("training compute C (FLOP)")
    ax.set_ylabel("compute-optimal N or D")
    ax.set_title("Compute-optimal allocation")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part6-compute-optimal.png", dpi=180)
    plt.close(fig)
    n_at, d_at = optimal(fitted, np.asarray([compute_budget]))
    paper_n_at, paper_d_at = optimal(printed, np.asarray([compute_budget]))
    return {
        "N_star": float(n_at[0]),
        "D_star": float(d_at[0]),
        "tokens_per_parameter": float(d_at[0] / n_at[0]),
        "printed_tokens_per_parameter": float(paper_d_at[0] / paper_n_at[0]),
        "printed_N_star": float(paper_n_at[0]),
        "N_slope": float(np.polyfit(np.log(compute), np.log(refit_n), 1)[0]),
        "D_slope": float(np.polyfit(np.log(compute), np.log(refit_d), 1)[0]),
    }


def main() -> None:
    settings = load_settings()
    p6 = settings["part6"]
    n, compute, loss, downloaded_rows = read_runs()
    d = compute / (6.0 * n)
    best, best_objective = fit(
        n,
        d,
        loss,
        float(p6["huber_delta"]),
        grid_starts_log(),
    )
    bootstrap_fits = bootstrap(
        n,
        d,
        loss,
        best,
        float(p6["huber_delta"]),
        int(p6["bootstrap"]),
        seed=5315,
    )
    standard_errors = bootstrap_fits.std(axis=0, ddof=1)
    np.savetxt(
        "artifacts/part6-bootstrap.csv",
        bootstrap_fits,
        delimiter=",",
        header=",".join(PARAMETER_NAMES),
        comments="",
    )

    printed = np.asarray(
        [
            float(p6["printed"]["E"]),
            float(p6["printed"]["A"]),
            float(p6["printed"]["B"]),
            float(p6["printed"]["alpha"]),
            float(p6["printed"]["beta"]),
        ]
    )
    figure_runs(n, compute, d, loss, best)
    refit_mean, printed_mean = plot_residuals(n, d, loss, best, printed)
    allocation = figure_allocations(
        best,
        printed,
        float(p6["compute_budget"]),
        float(p6["chinchilla_n"]),
        float(p6["chinchilla_d"]),
    )

    budget_compute = float(p6["compute_budget"])
    bootstrap_n, bootstrap_d = optimal(bootstrap_fits.T, np.full(len(bootstrap_fits), budget_compute))
    ratio = bootstrap_d / bootstrap_n
    ratio_interval = np.quantile(ratio, [0.025, 0.975]).tolist()
    parameter_interval = {
        name: np.quantile(bootstrap_fits[:, index], [0.025, 0.975]).tolist()
        for index, name in enumerate(PARAMETER_NAMES)
    }
    fitted_parameters = {name: float(value) for name, value in zip(PARAMETER_NAMES, best)}
    checks = {
        "exponent_alpha": {
            "pass": 0.32 <= best[3] <= 0.38,
            "actual": float(best[3]),
            "reference": 0.347,
        },
        "exponent_beta": {
            "pass": 0.32 <= best[4] <= 0.42,
            "actual": float(best[4]),
            "reference": 0.367,
        },
        "floor": {
            "pass": 1.75 <= best[0] <= 1.90,
            "actual": float(best[0]),
            "reference": 1.817,
        },
        "refit_residual_mean": {
            "pass": abs(refit_mean) <= 0.005,
            "actual": refit_mean,
            "reference": -0.0007,
        },
        "printed_residual_mean_above_0.01": {
            "pass": printed_mean > 0.01,
            "actual": printed_mean,
            "reference": 0.017,
        },
        "tokens_per_parameter": {
            "pass": 10.0 <= allocation["tokens_per_parameter"] <= 30.0,
            "actual": allocation["tokens_per_parameter"],
            "reference": 17.9,
        },
        "bootstrap_ci_contains_20": {
            "pass": bool(ratio_interval[0] <= 20.0 <= ratio_interval[1]),
            "actual_interval": ratio_interval,
        },
        "allocation_slopes": {
            "N_near_0.51": {"pass": abs(allocation["N_slope"] - 0.51) <= 0.05, "actual": allocation["N_slope"]},
            "D_near_0.49": {"pass": abs(allocation["D_slope"] - 0.49) <= 0.05, "actual": allocation["D_slope"]},
        },
    }

    def leaves(node: object) -> list[bool]:
        if isinstance(node, dict):
            if isinstance(node.get("pass"), bool):
                return [node["pass"]]
            return [item for value in node.values() for item in leaves(value)]
        return []

    metrics = {
        "downloaded_rows": downloaded_rows,
        "fitted_rows_after_dropping_five_highest_loss": len(n),
        "huber_delta": float(p6["huber_delta"]),
        "best_objective": best_objective,
        "parameters": fitted_parameters,
        "bootstrap_standard_errors": dict(zip(PARAMETER_NAMES, standard_errors.tolist())),
        "bootstrap_95_intervals": parameter_interval,
        "bootstrap_count": int(p6["bootstrap"]),
        "residual_means": {"refit": refit_mean, "printed": printed_mean},
        "compute_allocation": allocation,
        "tokens_per_parameter_95_interval": ratio_interval,
        "checks": checks,
        "all_checks_pass": all(leaves(checks)),
    }
    write_json("part6-metrics.json", metrics)
    print(
        json.dumps(
            metrics,
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else item,
        )
    )
    if not metrics["all_checks_pass"]:
        raise SystemExit("Part 6 acceptance check failed")


if __name__ == "__main__":
    main()

