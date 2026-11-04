#!/usr/bin/env python3
"""Draw the Week 9 evidence and check the numerical contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)

BLUE = "#2b6a94"
ORANGE = "#b85b17"
GREEN = "#2f7656"
INK = "#26323a"


def read_json(path: Path):
    return json.loads(path.read_text())


def case(name: str):
    return read_json(ART / name / "run.json")


def asia_dims():
    tokens = (ROOT / "asia.uai").read_text().split()
    assert tokens[0] == "MARKOV"
    count = int(tokens[1])
    return [int(x) for x in tokens[2 : 2 + count]]


def recount_plan(plan, dims):
    def entries(labels):
        return math.prod(dims[label] for label in labels)

    steps = []

    def visit(node):
        if node["kind"] == "leaf":
            return plan["input_labels"][node["tensor"]]
        left = visit(node["left"])
        right = visit(node["right"])
        loop_labels = sorted(set(left) | set(right))
        retained = node["retained"]
        steps.append(
            {
                "index": len(steps),
                "inputs": [left, right],
                "retained": retained,
                "loop_labels": loop_labels,
                "cost": entries(loop_labels),
                "result_entries": entries(retained),
            }
        )
        return retained

    visit(plan["tree"])
    largest_input = max(entries(labels) for labels in plan["input_labels"])
    largest_intermediate = max((step["result_entries"] for step in steps), default=1)
    return {
        "work": sum(step["cost"] for step in steps),
        "largest_tensor": max(largest_input, largest_intermediate),
        "largest_input": largest_input,
        "largest_intermediate": largest_intermediate,
        "steps": steps,
    }


def numpy_tensor_checks():
    checks = case("prior")["tensor_checks"]
    a = np.asarray(checks["a"], dtype=np.float64)
    b = np.asarray(checks["b"], dtype=np.float64)
    ab = a @ b
    trace = np.trace(ab)
    beta = 0.5
    coupling = 1.0
    z = 0.0
    for bits in np.ndindex(2, 2, 2):
        spins = np.asarray(bits, dtype=np.float64) * 2.0 - 1.0
        z += np.exp(beta * coupling * (spins[0] * spins[1] + spins[1] * spins[2] + spins[2] * spins[0]))
    ab_error = float(np.max(np.abs(np.asarray(checks["tensor_ab"]) - ab)))
    trace_error = abs(float(checks["tensor_trace"]) - float(trace))
    z_error = abs(float(checks["tensor_ising_z"]) - z)
    maximum = max(ab_error, trace_error, z_error)
    print(
        "numpy direct checks:",
        f"AB={ab.tolist()}",
        f"tr(AB)={trace:.17g}",
        f"Z={z:.17g}",
        f"max_abs_error={maximum:.3e}",
    )
    assert maximum < 1e-12
    return {
        "a": a.tolist(),
        "b": b.tolist(),
        "numpy_ab": ab.tolist(),
        "numpy_trace": float(trace),
        "numpy_ising_z": float(z),
        "tensor_ab": checks["tensor_ab"],
        "tensor_trace": checks["tensor_trace"],
        "tensor_ising_z": checks["tensor_ising_z"],
        "ab_error": ab_error,
        "trace_error": trace_error,
        "ising_error": z_error,
        "max_abs_error": maximum,
    }


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#78848c",
            "axes.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.hashsalt": "amat5315-week9",
        }
    )


def draw_query_diagram():
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title(
        "ASIA query tensor network: A = 0, D = 0 (AD); T remains open",
        fontsize=16,
        pad=12,
    )

    factors = {
        "p(A)": ((2.7, 7.6), "A"),
        "p(T|A)": ((5.25, 7.2), "T, A"),
        "p(S)": ((7.9, 7.95), "S"),
        "p(L|S)": ((10.4, 7.18), "L, S"),
        "p(B|S)": ((13.0, 7.15), "B, S"),
        "p(E|T,L)": ((7.0, 4.45), "E, T, L"),
        "p(X|E)": ((9.55, 2.85), "X, E"),
        "p(D|E,B)": ((13.0, 4.2), "D, E, B"),
    }
    variables = {
        "A": (3.35, 5.7),
        "T": (5.65, 5.15),
        "S": (7.7, 5.85),
        "L": (9.35, 5.2),
        "B": (11.85, 5.2),
        "E": (8.35, 3.7),
        "X": (10.0, 2.15),
        "D": (13.35, 2.8),
    }
    scopes = {
        "p(A)": ["A"],
        "p(T|A)": ["T", "A"],
        "p(S)": ["S"],
        "p(L|S)": ["L", "S"],
        "p(B|S)": ["B", "S"],
        "p(E|T,L)": ["E", "T", "L"],
        "p(X|E)": ["X", "E"],
        "p(D|E,B)": ["D", "E", "B"],
    }

    for name, (center, axes) in factors.items():
        x, y = center
        color = "#fff4e7" if name in {"p(E|T,L)", "p(D|E,B)"} else "#edf5fa"
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.85, y - 0.43),
                1.7,
                0.86,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor=BLUE,
                linewidth=1.3,
            )
        )
        ax.text(x, y + 0.12, name, ha="center", va="center", weight="bold")
        ax.text(x, y - 0.19, f"axes: {axes}", ha="center", va="center", fontsize=8.2)

    for name, (center, _) in factors.items():
        fx, fy = center
        for var in scopes[name]:
            vx, vy = variables[var]
            ax.plot([fx, vx], [fy - 0.38, vy + 0.14], color="#6c7c88", lw=1.0, zorder=0)

    for var, (x, y) in variables.items():
        color = "#dceff8" if var == "T" else ("#e6f2e9" if var == "S" else "white")
        edge = BLUE if var == "T" else (GREEN if var == "S" else "#6c7c88")
        ax.add_patch(Circle((x, y), 0.19, facecolor=color, edgecolor=edge, lw=1.4, zorder=3))
        ax.text(x, y, var, ha="center", va="center", weight="bold", zorder=4)

    ax.plot([3.35, 2.25], [5.7, 4.92], color=ORANGE, lw=1.4)
    ax.plot([13.35, 14.2], [2.8, 2.15], color=ORANGE, lw=1.4)
    for x, y, text in [(2.0, 4.58, "A* = 0"), (14.35, 1.78, "D* = 0")]:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            color=ORANGE,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#fff7ef", "ec": ORANGE, "lw": 1.2},
        )

    ax.annotate(
        "T is the only open query axis",
        xy=(5.65, 5.15),
        xytext=(2.7, 2.9),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.4},
        color=BLUE,
        weight="bold",
    )
    ax.text(
        7.7,
        4.95,
        "S occurs in p(S), p(L|S), p(B|S)",
        ha="center",
        color=GREEN,
        weight="bold",
    )
    ax.text(
        8.5,
        0.55,
        "Every box lists the exact UAI scope order; selector caps condition A and D.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(EVIDENCE / "graphs.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_posteriors():
    runs = {name: case(name)["contraction"] for name in ["prior", "AD", "ADX"]}
    enum = {name: case(name)["enumeration"] for name in ["prior", "AD", "ADX"]}
    labels = ["Tuberculosis", "Lung cancer", "Bronchitis"]
    ids = [1, 3, 4]
    titles = ["Prior", "Visit + dyspnoea", "Visit + dyspnoea\n+ positive X-ray"]
    colors = ["#294e61", "#3f7897", "#6ea6c2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
    for ax, name, title in zip(axes, runs, titles):
        values = np.array([runs[name]["marginals"][str(v)][0] for v in ids])
        enum_values = np.array([enum[name]["marginals"][str(v)][0] for v in ids])
        y = np.arange(3)
        ax.barh(y, values, color=colors, height=0.72)
        ax.plot(enum_values, y, "k|", ms=14, mew=2.2, label="enumeration")
        for i, value in enumerate(values):
            ax.text(min(value + 0.018, 0.96), i, f"{value:.3f}", va="center", fontsize=9)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.grid(axis="x", color="#dfe5e8", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("")
    fig.text(0.5, 0.055, "Probability of disease being present", ha="center", fontsize=10)
    fig.text(0.5, 0.012, "Bars: tensor contraction. Black marks: enumeration.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(EVIDENCE / "posteriors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_joint():
    names = ["prior", "AD", "ADX"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for col, name in enumerate(names):
        result = case(name)["contraction"]
        joint = np.array(result["joints"]["T,L"]).reshape(2, 2)
        marg_t = np.array(result["marginals"]["1"])
        marg_l = np.array(result["marginals"]["3"])
        product = np.outer(marg_t, marg_l)
        dependence = joint - product
        im0 = axes[0, col].imshow(joint, cmap="Blues", vmin=0, vmax=max(0.5, joint.max()))
        vmax = max(1e-6, float(np.abs(dependence).max()))
        im1 = axes[1, col].imshow(
            dependence, cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        for row in range(2):
            for c in range(2):
                axes[0, col].text(
                    c, row, f"{joint[row, c]:.4f}", ha="center", va="center",
                    color="white" if joint[row, c] > 0.4 else INK, weight="bold",
                )
                axes[1, col].text(
                    c, row, f"{dependence[row, c]:+.4f}", ha="center", va="center",
                    color="white" if abs(dependence[row, c]) > 0.55 * vmax else INK,
                )
        for row in range(2):
            axes[row, col].set_xticks([0, 1], ["L=0", "L=1"])
            axes[row, col].set_yticks([0, 1], ["T=0", "T=1"])
            axes[row, col].set_title(["Prior", "AD", "ADX"][col] + (" joint p(T,L)" if row == 0 else " joint minus product"))
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
    fig.suptitle("Dependence is the nonzero joint-minus-product field", fontsize=14)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "joint.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


FACTOR_NAMES = [
    "p(A)", "p(T|A)", "p(S)", "p(L|S)", "p(B|S)",
    "p(E|T,L)", "p(X|E)", "p(D|E,B)", "A*=0", "D*=0",
]
VAR_NAMES = ["A", "T", "S", "L", "B", "E", "X", "D"]


def tree_layout(tree):
    positions = {}
    leaves = 0

    def visit(node, depth=0):
        nonlocal leaves
        if node["kind"] == "leaf":
            x = leaves
            leaves += 1
            positions[id(node)] = (x, -depth, node)
            return x
        left = visit(node["left"], depth + 1)
        right = visit(node["right"], depth + 1)
        x = (left + right) / 2
        positions[id(node)] = (x, -depth, node)
        return x

    visit(tree)
    return positions


def draw_plan(ax, run):
    plan = run["_order"]
    complexity = run["_complexity"]
    pos = tree_layout(plan["tree"])
    largest = complexity["largest_tensor"]
    for (x, y, node) in pos.values():
        if node["kind"] == "leaf":
            continue
        for child in (node["left"], node["right"]):
            cx, cy, _ = pos[id(child)]
            ax.plot([x, cx], [y - 0.12, cy + 0.16], color="#75848d", lw=0.9, zorder=0)

    for (x, y, node) in pos.values():
        if node["kind"] == "leaf":
            name = FACTOR_NAMES[node["tensor"]]
            color = "#fff0df" if node["tensor"] >= 8 else "#eef6fa"
            ax.text(
                x, y, name, ha="center", va="center", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.27", "fc": color, "ec": BLUE, "lw": 1},
            )
        else:
            retained = node["retained"]
            entries = 2 ** len(retained)
            label = ("".join(VAR_NAMES[v] for v in retained) or "Scalar") + f" · {entries}"
            hot = entries == largest and largest > complexity["largest_input"]
            ax.text(
                x, y, label, ha="center", va="center", fontsize=7.8,
                bbox={"boxstyle": "round,pad=0.25",
                      "fc": "#fff4e6" if hot else "white",
                      "ec": ORANGE if hot else "#8a99a2",
                      "lw": 1.25 if hot else 0.8},
            )
    ax.set_xlim(-0.7, max(x for x, _, _ in pos.values()) + 0.7)
    ax.set_ylim(min(y for _, y, _ in pos.values()) - 0.5, 0.55)
    ax.axis("off")


def draw_orders():
    random_run = case("random")
    greedy_run = case("greedy")
    dims = asia_dims()
    for run in (random_run, greedy_run):
        run["_order"] = read_json(ART / run["plan"]["method"] / "order.json")
        run["_complexity"] = recount_plan(run["_order"], dims)
    fig = plt.figure(figsize=(16, 12))
    tree_random = fig.add_axes([0.04, 0.53, 0.62, 0.40])
    tree_greedy = fig.add_axes([0.04, 0.07, 0.62, 0.40])
    bars_work = fig.add_axes([0.72, 0.57, 0.24, 0.30])
    bars_space = fig.add_axes([0.72, 0.14, 0.24, 0.30])
    for ax, run, title in [
        (tree_random, random_run, f"Random seed {random_run['plan']['seed']} · actual saved tree"),
        (tree_greedy, greedy_run, "Greedy · actual saved tree"),
    ]:
        draw_plan(ax, run)
        ax.set_title(title, loc="left", color=ORANGE if "Random" in title else BLUE)
    methods = ["Random", "Greedy"]
    works = [random_run["plan"]["work"], greedy_run["plan"]["work"]]
    spaces = [random_run["plan"]["largest_tensor"], greedy_run["plan"]["largest_tensor"]]
    for ax, values, label in [(bars_work, works, "Multiply-accumulate terms"), (bars_space, spaces, "Elements, including inputs")]:
        ax.barh(methods, values, color=[ORANGE, BLUE], height=0.55)
        for y, value in enumerate(values):
            ax.text(value * 1.02, y, f"{value:,}", va="center", fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.25)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    bars_work.set_title("Recounted plan cost")
    bars_space.set_title("Recounted largest tensor")
    fig.text(
        0.72,
        0.47,
        "Random: largest intermediate ATSLBE · 64 (orange outline).\n"
        "Greedy: largest tensor is input p(E|T,L) · 8.\n"
        "Saved plans replay exactly; analysis recounts without tensor evaluation.",
        fontsize=9,
    )
    fig.suptitle("Same ASIA AD evidence query; two contraction trees", fontsize=15)
    fig.savefig(EVIDENCE / "orders.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_bp_traces():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, name, title in zip(axes, ["bp-prior", "bp-AD", "bp-ADX"], ["Prior", "AD", "ADX"]):
        bp = case(name)["bp"]
        sweeps = bp["sweeps"]
        x = [s["sweep"] for s in sweeps]
        residual = [max(s["undamped_residual"], 1e-16) for s in sweeps]
        error = [max(s["max_marginal_error"], 1e-16) for s in sweeps]
        ax.semilogy(x, residual, color=BLUE, lw=1.5, label="undamped message residual")
        ax.semilogy(x, error, color=ORANGE, lw=1.5, label="largest marginal error")
        ax.axhline(1e-10, color="#8b949a", ls="--", lw=0.9)
        ax.set_title(title)
        ax.set_xlabel("Sweeps")
        ax.grid(which="major", color="#e3e8eb", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.04, 0.05,
            f"final residual {bp['returned_undamped_residual']:.2e}\nmax error {bp['max_marginal_error']:.5f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#c7d0d5"},
        )
    axes[0].set_ylabel("Absolute difference (log scale)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Loopy BP: convergence is not accuracy", fontsize=14)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "bp-traces.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_mar(path: Path):
    tokens = path.read_text().split()
    assert tokens[0] == "MAR"
    count = int(tokens[1])
    at = 2
    result = {}
    for var in range(count):
        states = int(tokens[at])
        at += 1
        result[var] = [float(tokens[at + i]) for i in range(states)]
        at += states
    return result


def draw_promedus():
    greedy = case("promedus-greedy")
    treesa = case("promedus-treesa")
    bprun = case("promedus-bp")
    published_pr = float((ROOT / "promedus/Promedus_14.uai.PR").read_text().split()[1])
    mar = parse_mar(ROOT / "promedus/Promedus_14.uai.MAR")
    ids = [59, 102, 139]
    exact_g = [greedy["contraction"]["marginals"][str(v)][1] for v in ids]
    exact_t = [treesa["contraction"]["marginals"][str(v)][1] for v in ids]
    bp_values = [bprun["bp"]["final_marginals"][str(v)][1] for v in ids]
    published = [mar[v][1] for v in ids]
    log10_g = greedy["contraction"]["log_evidence"] / math.log(10)
    log10_t = treesa["contraction"]["log_evidence"] / math.log(10)
    exact_delta = abs(log10_g - log10_t)
    mar_plan_delta = max(abs(a - b) for a, b in zip(exact_g, exact_t))
    pr_error = abs(log10_g - published_pr)
    mar_exact_error = max(abs(a - b) for a, b in zip(exact_g, published))
    mar_bp_error = max(abs(a - b) for a, b in zip(bp_values, published))

    assert exact_delta < 1e-10
    assert mar_plan_delta < 1e-10
    assert pr_error < 1e-4
    assert mar_exact_error < 1e-5
    assert not bprun["bp"]["converged"]

    fig = plt.figure(figsize=(16, 9.5))
    ax1 = fig.add_axes([0.05, 0.62, 0.22, 0.25])
    ax2 = fig.add_axes([0.31, 0.62, 0.22, 0.25])
    ax3 = fig.add_axes([0.60, 0.56, 0.36, 0.34])
    methods = ["Greedy", "TreeSA"]
    works = [greedy["plan"]["work"], treesa["plan"]["work"]]
    spaces = [greedy["plan"]["largest_tensor"], treesa["plan"]["largest_tensor"]]
    for ax, values, refs, label, title in [
        (ax1, works, [667111481, 10635220], "Work terms", "Contraction work"),
        (ax2, spaces, [4194304, 131072], "Largest tensor entries", "Largest generated tensor"),
    ]:
        ax.bar(methods, values, color=[BLUE, ORANGE], width=0.55)
        for x, value in enumerate(values):
            ax.text(x, value * 1.02, f"{value:,}", ha="center", va="bottom", fontsize=9)
        for x, ref in enumerate(refs):
            ax.hlines(ref, x - 0.28, x + 0.28, color="#4f5b63", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylim(min(values) * 0.6, max([*values, *refs]) * 2.4)
        ax.set_ylabel(label + " (log scale)")
        ax.set_title(title + "\n(dashed: sheet reference)")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    x_ids = np.arange(len(ids))
    width = 0.25
    for offset, values, label, color in [
        (-width, exact_g, "exact", BLUE),
        (0.0, bp_values, "BP", ORANGE),
        (width, published, "published", GREEN),
    ]:
        ax3.bar(x_ids + offset, values, width=width, label=label, color=color)
        for x, value in zip(x_ids + offset, values):
            ax3.text(x, min(value + 0.025, 1.02), f"{value:.3f}", ha="center", fontsize=7.8)
    ax3.set_xticks(x_ids, [str(v) for v in ids])
    ax3.set_ylim(0, 1.12)
    ax3.set_xlabel("Variable ID")
    ax3.set_ylabel("p(state 1 | evidence)")
    ax3.set_title("Final selected MAR: exact, BP, published")
    ax3.legend(frameon=False, ncol=3, loc="upper center")
    ax3.grid(axis="y", color="#e4e9ec")
    ax3.set_axisbelow(True)
    ax3.spines[["top", "right"]].set_visible(False)

    ax_text = fig.add_axes([0.04, 0.06, 0.92, 0.42])
    ax_text.axis("off")
    ratio_work = works[0] / works[1]
    ratio_space = spaces[0] / spaces[1]
    lines = [
        f"log10 Z exact: greedy {log10_g:.15f}   TreeSA {log10_t:.15f}   |difference| {exact_delta:.2e}",
        f"Published PR {published_pr:.15f}   |exact PR error| {pr_error:.2e}",
        "Exact MAR errors for IDs 59,102,139: " + ", ".join(f"{abs(a-b):.2e}" for a,b in zip(exact_g,published)),
        f"TreeSA / saved greedy: {ratio_work:.2f}× less work, {ratio_space:.1f}× smaller largest tensor",
        "BP selected state-1 beliefs: " + ", ".join(f"{v:.6f}" for v in bp_values),
        f"BP MAR max error {mar_bp_error:.6f}; 1000 sweeps; residual {bprun['bp']['returned_undamped_residual']:.3e}",
        "BP is UNCONVERGED at tolerance 1e-10 and INACCURATE for the selected queries.",
    ]
    for i, line in enumerate(lines):
        color = "#a73522" if "BP is" in line else INK
        weight = "bold" if "BP is" in line else "normal"
        ax_text.text(0.0, 0.94 - i * 0.125, line, fontsize=10.2, color=color, weight=weight)
    fig.suptitle("Promedus_14: exact order costs, published answers, and loopy BP", fontsize=15)
    fig.savefig(EVIDENCE / "promedus-order-costs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    sweeps = bprun["bp"]["sweeps"]
    x = [s["sweep"] for s in sweeps]
    axes[0].semilogy(x, [max(s["undamped_residual"], 1e-16) for s in sweeps], color=BLUE)
    axes[0].axhline(1e-10, color="#7c8991", ls="--", lw=0.9)
    axes[0].set_ylabel("Undamped residual")
    axes[0].set_title("No convergence after 1000 sweeps")
    for var, color in zip(ids, [BLUE, ORANGE, GREEN]):
        axes[1].semilogy(
            x,
            [max(s["query_errors"][str(var)], 1e-16) for s in sweeps],
            color=color,
            lw=1.3,
            label=f"p({var}=1)",
        )
    axes[1].set_ylabel("Absolute marginal error")
    axes[1].set_title("Selected-query error stays large")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("Sweeps")
        ax.grid(color="#e3e8eb")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Promedus_14 BP residual and query accuracy", fontsize=14)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "promedus-bp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "greedy_work": works[0],
        "treesa_work": works[1],
        "greedy_space": spaces[0],
        "treesa_space": spaces[1],
        "work_reduction": ratio_work,
        "space_reduction": ratio_space,
        "log10_Z_greedy": log10_g,
        "log10_Z_treesa": log10_t,
        "exact_plan_log10_difference": exact_delta,
        "exact_plan_marginal_difference": mar_plan_delta,
        "published_log10_Z": published_pr,
        "published_PR_error": pr_error,
        "published_exact_MAR_values": published,
        "exact_state1_values": exact_g,
        "bp_state1_values": bp_values,
        "published_MAR_max_error_exact": mar_exact_error,
        "published_MAR_max_error_bp": mar_bp_error,
        "bp_converged": bprun["bp"]["converged"],
        "bp_residual": bprun["bp"]["returned_undamped_residual"],
        "bp_sweeps": bprun["bp"]["sweeps_run"],
        "bp_update_work": bprun["bp"]["cumulative_update_work"],
        "runtime_seconds": {k: case(k).get("runtime_seconds") for k in ["promedus-greedy", "promedus-treesa", "promedus-bp"]},
    }


def verify_and_summary():
    for name in ["prior", "AD", "ADX"]:
        run = case(name)
        comp = run["comparison"]
        assert comp["max_abs_marginal_error"] < 1e-10
        assert comp["relative_evidence_mass_error"] < 1e-10
        assert comp["marginals_normalized"] and comp["joints_normalized"]
    random_complexity = read_json(ART / "random/complexity.json")
    analysis = read_json(ART / "analysis/complexity.json")
    assert random_complexity == analysis
    dims = asia_dims()
    for name in ["random", "greedy"]:
        saved = read_json(ART / name / "complexity.json")
        recounted = recount_plan(read_json(ART / name / "order.json"), dims)
        assert recounted["work"] == saved["work"]
        assert recounted["largest_tensor"] == saved["largest_tensor"]
        assert recounted["largest_input"] == saved["largest_input"]
        assert recounted["largest_intermediate"] == saved["largest_intermediate"]
        assert recounted["steps"] == saved["steps"]
    replay = case("replay")
    original = case("greedy")
    assert replay["comparison"]["max_abs_marginal_error"] < 1e-10
    assert abs(replay["contraction"]["evidence_mass"] - original["contraction"]["evidence_mass"]) / original["contraction"]["evidence_mass"] < 1e-10
    bp_adx = case("bp-ADX")["bp"]
    assert bp_adx["converged"] and bp_adx["returned_undamped_residual"] < 1e-10
    return {
        "numpy_checks": numpy_tensor_checks(),
        "asia": {
            name: {
                "evidence_mass": case(name)["contraction"]["evidence_mass"],
                "plan_work": case(name)["plan"]["work"],
                "largest_tensor": case(name)["plan"]["largest_tensor"],
                "max_marginal_error": case(name)["comparison"]["max_abs_marginal_error"],
                "relative_evidence_error": case(name)["comparison"]["relative_evidence_mass_error"],
            }
            for name in ["prior", "AD", "ADX"]
        },
        "orders": {
            "random": {k: case("random")["plan"][k] for k in ["seed", "work", "largest_tensor"]},
            "greedy": {k: case("greedy")["plan"][k] for k in ["work", "largest_tensor"]},
            "analysis_matches_saved": random_complexity == analysis,
            "independently_recounted": True,
        },
        "bp_asia": {
            name: {
                "converged": case("bp-" + name)["bp"]["converged"],
                "sweeps": case("bp-" + name)["bp"]["sweeps_run"],
                "residual": case("bp-" + name)["bp"]["returned_undamped_residual"],
                "max_marginal_error": case("bp-" + name)["bp"]["max_marginal_error"],
            }
            for name in ["prior", "AD", "ADX"]
        },
    }


def main():
    style()
    summary = verify_and_summary()
    draw_query_diagram()
    draw_posteriors()
    draw_joint()
    draw_orders()
    draw_bp_traces()
    summary["promedus"] = draw_promedus()
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
