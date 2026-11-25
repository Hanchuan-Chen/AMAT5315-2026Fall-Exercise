import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
plt.rcParams.update({"font.size": 8})

POLICY_LABEL = {
    "most-frac": "most-frac",
    "neural": "neural",
    "root-neural": "root-neural",
    "strong": "strong",
}
POLICY_COLOR = {
    "most-frac": "#d95f02",
    "neural": "#e69f00",
    "root-neural": "#0072b2",
    "strong": "#1b9e77",
}
EVIDENCE_COLOR = {"bound": "#2878b5", "exact": "#d95f02", "gap": "#2878b5", "round": "#e69f00"}


def load(path):
    return json.loads(path.read_text())


def panel_label(ax, label):
    ax.set_title(label, fontsize=8.7, loc="left")


def style(ax):
    ax.grid(color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_gap():
    run = load(ARTIFACTS / "gap" / "run.json")
    data = run["instances"]
    x = np.arange(len(data))
    lp = np.array([row["lp_bound"] for row in data])
    exact = np.array([row["exact_energy"] for row in data])
    gap = np.array([row["integrality_gap"] for row in data])
    rounded = np.array([row["rounding_excess"] for row in data])
    ids = np.array([row["id"] for row in data])

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), constrained_layout=True)
    ax = axes[0]
    axis_low = min(lp) - 2
    for value, color in ((lp, EVIDENCE_COLOR["bound"]), (exact, EVIDENCE_COLOR["exact"])):
        ax.vlines(ids, value, axis_low, color=color, alpha=0.28, linewidth=1.4)
        ax.scatter(ids, value, color=color, s=16, zorder=3)
    ax.set_ylim(axis_low, max(exact) + 3)
    ax.set_xlabel("Seeded Spin Glass ID")
    ax.set_ylabel("Energy H(s)")
    panel_label(ax, "(a) Spin Glass LP vs. ILP energy (22 spins, 52 bonds)")
    ax.legend(
        [
            Line2D([], [], color=EVIDENCE_COLOR["bound"], marker="o", markersize=4, linewidth=0.8),
            Line2D([], [], color=EVIDENCE_COLOR["exact"], marker="o", markersize=4, linewidth=0.8),
        ],
        ["LP lower bound", "Exact ILP ground state"],
        frameon=False,
        loc="upper right",
    )
    ax.text(
        0.03,
        0.96,
        f"Mean gap = {np.mean(gap):.2f} ({100*np.mean(gap/np.abs(exact)):.1f}%)\n"
        f"Suboptimal round: {run['summary']['suboptimal_roundings']}/{len(data)}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fff7e6", "ec": "#c89f6a", "lw": 0.7},
    )
    style(ax)

    ax = axes[1]
    width = 0.38
    ax.bar(ids - width / 2, gap, width, color=EVIDENCE_COLOR["gap"], edgecolor="#1f4e6b", linewidth=0.4)
    ax.bar(ids + width / 2, rounded, width, color=EVIDENCE_COLOR["round"], edgecolor="#9a6200", linewidth=0.4)
    ax.set_xlabel("Seeded Spin Glass ID")
    ax.set_ylabel("Energy difference")
    panel_label(ax, "(b) Integrality gap vs. naive rounding excess")
    ax.legend(["Integrality gap ($H^* - z_{LP}$)", "Rounded excess ($H_{round} - H^*$)"], frameon=False)
    style(ax)
    fig.savefig(ARTIFACTS / "gap.png", dpi=220)
    plt.close(fig)


def plot_bnb():
    runs = {policy: load(ARTIFACTS / f"bnb_{suffix}" / "run.json")
            for policy, suffix in (("most-frac", "most"), ("strong", "strong"))}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    for ax, key, ylabel, label in (
        (axes[0], "nodes", "B&B nodes explored", "(a) B&B nodes on 48 Spin Glasses (22, 52)"),
        (axes[1], "lp_solves", "LP relaxations solved per instance", "(b) LP solves per instance (lookahead cost)"),
    ):
        for policy in ("most-frac", "strong"):
            values = np.sort([row[key] for row in runs[policy]["instances"]])
            rank = np.arange(len(values))
            mean = runs[policy]["summary"][f"mean_{key}"]
            ax.plot(
                rank,
                values,
                color=POLICY_COLOR[policy],
                marker="o",
                markersize=2.4,
                linewidth=1.4,
                label=f"{POLICY_LABEL[policy]} (mean {mean:.1f})",
            )
        ax.set_xlabel("Sorted instance rank")
        ax.set_ylabel(ylabel)
        panel_label(ax, label)
        ax.legend(frameon=False, loc="upper left")
        style(ax)
    ratio = runs["most-frac"]["summary"]["mean_nodes"] / runs["strong"]["summary"]["mean_nodes"]
    slow = runs["strong"]["summary"]["mean_wall_ms"] / runs["most-frac"]["summary"]["mean_wall_ms"]
    per_node = runs["strong"]["summary"]["mean_lp_solves"] / runs["strong"]["summary"]["mean_nodes"]
    mismatches = sum(
        row["error"] > 1e-6
        for policy in ("most-frac", "strong")
        for row in runs[policy]["instances"]
    )
    axes[0].text(
        0.06,
        0.58,
        f"Node cut: {ratio:.2f}x smaller tree\n"
        f"Exact opt: {mismatches}/{len(runs['strong']['instances'])} mismatches",
        transform=axes[0].transAxes,
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#eaf7ef", "ec": "#7bbf91", "lw": 0.7},
    )
    axes[1].text(
        0.04,
        0.42,
        f"SB cost: {per_node:.1f} LP solves / node\nWall time: {slow:.1f}x slower",
        transform=axes[1].transAxes,
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fff7e6", "ec": "#c89f6a", "lw": 0.7},
    )
    fig.savefig(ARTIFACTS / "bnb.png", dpi=220)
    plt.close(fig)


def plot_neural_bnb():
    model = load(ARTIFACTS / "model.json")
    dataset = load(ARTIFACTS / "bnb_strong" / "dataset.json")
    run = load(ARTIFACTS / "eval_test" / "run.json")
    train_nodes = sum(node["instance"] < 36 for node in dataset["nodes"])
    validation_nodes = len(dataset["nodes"]) - train_nodes
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    ax = axes[0]
    epochs = np.arange(1, len(model["history"]["train"]) + 1)
    ax.plot(epochs, model["history"]["train"], color=EVIDENCE_COLOR["bound"], linewidth=1.5, label="Train CE")
    ax.plot(epochs, model["history"]["validation"], color=EVIDENCE_COLOR["exact"], linewidth=1.5,
            linestyle="--", label="Val CE (disjoint instances)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    panel_label(ax, "(a) Masked softmax cross-entropy loss")
    ax.legend(frameon=False)
    ax.text(
        0.42,
        0.52,
        f"Train: {train_nodes} nodes ({len(dataset['split']['train_instances'])} insts)\n"
        f"Val: {validation_nodes} nodes ({len(dataset['split']['validation_instances'])} insts)\n"
        f"Checkpoint val CE: {model['best_validation_cross_entropy']:.3f}",
        transform=ax.transAxes,
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#aaaaaa", "lw": 0.6},
    )
    style(ax)

    ax = axes[1]
    for policy in ("most-frac", "neural", "strong"):
        values = np.sort([row["policies"][policy]["nodes"] for row in run["instances"]])
        mean = run["summary"][policy]["mean_nodes"]
        ax.plot(
            np.arange(len(values)),
            values,
            marker="o",
            markersize=2.8,
            linewidth=1.4,
            color=POLICY_COLOR[policy],
            label=f"{POLICY_LABEL[policy]} (mean {mean:.1f})",
        )
    ax.set_xlabel("Sorted instance rank")
    ax.set_ylabel("B&B nodes explored")
    panel_label(ax, "(b) B&B nodes on 16 unseen Spin Glasses")
    ax.legend(frameon=False, loc="upper left")
    cut = 1 - run["summary"]["neural"]["mean_nodes"] / run["summary"]["most-frac"]["mean_nodes"]
    ax.text(
        0.04,
        0.52,
        f"Node cut: {100*cut:.0f}% vs most-frac\n"
        f"Exact opt: {run['largest_highs_error'] < 1e-6 and len(run['instances'])}/{len(run['instances'])} matches",
        transform=ax.transAxes,
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#eaf7ef", "ec": "#7bbf91", "lw": 0.7},
    )
    style(ax)
    fig.savefig(ARTIFACTS / "neural_bnb.png", dpi=220)
    plt.close(fig)


def plot_eval():
    test = {
        "22 spins, 52 bonds": load(ARTIFACTS / "eval_test" / "run.json"),
        "30 spins, 72 bonds": load(ARTIFACTS / "eval_transfer" / "run.json"),
    }
    policies = ("most-frac", "neural", "root-neural", "strong")
    labels = ("Most-frac", "Online Neural", "Root Neural", "Strong Branching")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    ax = axes[0]
    width = 0.36
    x = np.arange(len(policies))
    for offset, (size, color, run) in zip(
        (-width / 2, width / 2),
        (("22 spins, 52 bonds", "#3c78c5", test["22 spins, 52 bonds"]),
         ("30 spins, 72 bonds", "#d98500", test["30 spins, 72 bonds"])),
    ):
        values = [run["summary"][policy]["mean_nodes"] for policy in policies]
        bars = ax.bar(x + offset, values, width, color=color, edgecolor="#444444", linewidth=0.5, label=size)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Mean B&B nodes")
    panel_label(ax, "(a) Rust B&B nodes across policies")
    ax.legend(frameon=False)
    style(ax)

    ax = axes[1]
    run = test["30 spins, 72 bonds"]
    totals = np.array([run["summary"][policy]["mean_wall_ms"] for policy in policies])
    inference = np.array([run["summary"][policy]["mean_inference_ms"] for policy in policies])
    search = np.maximum(totals - inference, 0)
    ax.bar(x, search, color="#3c78c5", edgecolor="#444444", linewidth=0.5, label="HiGHS LP + search ms")
    ax.bar(x, inference, bottom=search, color="#e15759", edgecolor="#444444", linewidth=0.5,
           label="Neural infer ms")
    for index, total in enumerate(totals):
        ax.text(index, total, f"{total:.1f} ms", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Mean CPU wall time (ms / instance)")
    panel_label(ax, "(b) CPU time on 30-spin, 72-bond transfer")
    ax.set_ylim(0, totals.max() * 1.16)
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.savefig(ARTIFACTS / "eval.png", dpi=220)
    plt.close(fig)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    plot_gap()
    plot_bnb()
    plot_neural_bnb()
    plot_eval()


if __name__ == "__main__":
    main()
