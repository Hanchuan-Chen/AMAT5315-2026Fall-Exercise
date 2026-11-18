#!/usr/bin/env python3
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "artifacts"
BLUE = "#2d6794"
ORANGE = "#b45f12"
GOLD = "#c88a21"


def load(*parts):
    with (RUNS.joinpath(*parts) / "run.json").open() as stream:
        return json.load(stream)


def validate():
    gates = load("gadgets")
    expected_floors = {
        "AND": -3,
        "OR": -3,
        "NOT": -1,
        "COPY": -1,
        "SET0": -1,
        "SET1": -1,
        "XOR": -4,
    }
    for name, floor in expected_floors.items():
        energies = [row["energy"] for row in gates["gadgets"][name]]
        assert min(energies) == floor
        assert all(e == floor or e >= floor + 2 for e in energies)
        assert sum(e == floor for e in energies) > 0
    circuit = gates["circuit"]["rows"]
    assert gates["circuit"]["ground_state_count"] == 3
    assert sum(row["energy"] == -5 for row in circuit) == 3

    p0 = load("count_p0")
    assert p0["e_min"] == -180 and p0["g0"] == 2
    f15 = load("count_f15")
    assert f15["e_min"] == -299 and f15["g0"] == 1
    assert f15["sc"] == 5 and abs(f15["log_z"] - 598.3179) < 1e-4
    sampled = load("sample_f15")
    assert sampled["run_parameters"]["samples"] == 2000
    assert len(sampled["samples"]) == 2000
    assert sum(sampled["energy_histogram"].values()) == 2000
    ground = sampled["energy_histogram"][str(int(sampled["e_min"]))]
    assert abs(ground - sampled["predicted_ground_count"]) <= 25
    extract = json.loads((RUNS / "pred" / "factor_15.extract.txt").read_text())
    assert extract["solution"] == [1, 1, 1, 0, 1]
    assert extract["evaluation"] == "Min(0)"

    tree_sc = []
    for k in range(2, 9):
        plan = load(f"plan_k{k}_treesa")
        tree_sc.append(plan["sc"])
    assert tree_sc == [4, 7, 9, 11, 13, 15, 17]

    known = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}
    for n, count in known.items():
        assert load(f"count_nq{n}")["count"] == count
    for n in (6, 8):
        sample = load(f"sample_nq{n}")
        assert sample["sample_count"] == 200
        assert sample["distinct_boards"] == len(sample["frequencies"])
        for board in sample["boards"]:
            assert len(board) == n and sorted(board) == list(range(n))
            assert len({r + c for r, c in enumerate(board)}) == n
            assert len({r - c for r, c in enumerate(board)}) == n
    assert load("sample_nq6")["distinct_boards"] == 4
    assert load("sample_nq8")["distinct_boards"] >= 75


def plot_gates():
    data = load("gadgets")
    names = ["AND", "OR", "NOT", "COPY", "SET0", "SET1", "XOR"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.36})
    ax = axes[0]
    for x, name in enumerate(names):
        rows = data["gadgets"][name]
        valid = [row["energy"] for row in rows if row["valid"]]
        invalid = [row["energy"] for row in rows if not row["valid"]]
        ax.scatter(np.full(len(valid), x), valid, s=16, color=BLUE, marker="o", linewidths=0)
        ax.scatter(np.full(len(invalid), x), invalid, s=24, color=ORANGE, marker="x", linewidths=1.3)
    ax.scatter([], [], s=16, color=BLUE, label="valid row")
    ax.scatter([], [], s=24, color=ORANGE, marker="x", linewidths=1.3, label="invalid row")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks([-4, -2, 0, 2, 4, 6, 8])
    ax.set_ylim(-4.5, 9.4)
    ax.set_ylabel("energy of a truth-table row")
    ax.set_title("(a) Gate energies", fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.02), fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    rows = data["circuit"]["rows"]
    for x, row in enumerate(rows):
        color = BLUE if row["valid"] else ORANGE
        marker = "o" if row["valid"] else "x"
        size = 16 if row["valid"] else 24
        ax.scatter(x, row["energy"], s=size, color=color, marker=marker, linewidths=1.2)
    ax.set_xticks([1.5, 5.5, 9.5, 13.5], ["00", "01", "10", "11"])
    ax.set_xlim(-0.6, 15.6)
    ax.set_yticks([-5, -3, -1, 1, 3, 9])
    ax.set_xlabel("a b =", labelpad=2)
    ax.set_ylabel("energy of a state")
    ax.set_title("(b) Three gates joined", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01, "the 16 states of a, b, t, out", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.89)
    fig.savefig(OUT / "gates.png", dpi=240)
    plt.close(fig)


def plot_factoring():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    ks = list(range(2, 9))
    greedy = [load(f"plan_k{k}_greedy")["sc"] for k in ks]
    treesa = [load(f"plan_k{k}_treesa")["sc"] for k in ks]
    ax.plot(ks, greedy, color=ORANGE, marker="o", ms=3.2, lw=1.2, label="greedy")
    ax.plot(ks, treesa, color=BLUE, marker="o", ms=3.2, lw=1.2, label="treesa")
    ax.axhline(30, color="0.55", lw=0.8, ls=":")
    ax.text(2.15, 30.7, "sc = 30 (~8 GB RAM)", fontsize=7, color="0.35")
    ax.annotate(f"sc = {treesa[0]:.0f}", (2, treesa[0]), xytext=(0, -13), textcoords="offset points", fontsize=7, color=BLUE)
    ax.annotate(f"sc = {treesa[-1]:.0f}", (8, treesa[-1]), xytext=(-8, 8), textcoords="offset points", fontsize=7, color=BLUE)
    ax.set_xticks(ks)
    ax.set_xlabel("bits per factor k")
    ax.set_ylabel("space complexity sc (log$_2$ entries)")
    ax.set_title("(a) Contraction memory cost", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    sample = load("sample_f15")
    energies = list(range(int(sample["e_min"]), -287, 2))
    counts = [sample["energy_histogram"].get(str(energy), 0) for energy in energies]
    colors = [BLUE if e == sample["e_min"] else ORANGE for e in energies]
    ax.bar(range(len(energies)), counts, color=colors, width=0.72)
    for i, count in enumerate(counts):
        ax.text(i, count + 30, str(count), ha="center", va="bottom", fontsize=7)
    predicted = sample["predicted_ground_count"]
    ax.axhline(predicted, color="0.45", lw=0.9, ls="--")
    ax.text(len(energies) - 0.55, predicted + 35, f"predicted from Z: {predicted:.0f}", ha="right", fontsize=7, color="0.3")
    ax.set_xticks(range(len(energies)), [str(e) for e in energies])
    ax.set_xlabel("spin-glass energy E")
    ax.set_ylabel("samples (out of 2000)")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title("(b) N = 15 sampled states ($\\beta = 2$)", fontsize=10)
    ax.annotate(
        "ground state\n(decodes to 3 x 5)",
        xy=(0, counts[0] * 0.88),
        xytext=(1.2, counts[0] * 0.84),
        fontsize=7,
        color=BLUE,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.8},
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.88)
    fig.savefig(OUT / "factoring.png", dpi=240)
    plt.close(fig)


def draw_board(ax, board):
    n = len(board)
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor="#f4f4f4" if (r + c) % 2 else "white", edgecolor="0.82", lw=0.25))
    for r, c in enumerate(board):
        ax.add_patch(Rectangle((c + 0.18, n - 1 - r + 0.18), 0.64, 0.64, facecolor=BLUE, edgecolor="none"))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_nqueens():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"wspace": 0.37})
    ax = axes[0]
    ns = list(range(4, 11))
    contraction = [load(f"count_nq{n}")["count"] for n in ns]
    known = [2, 10, 4, 40, 92, 352, 724]
    ax.plot(ns, contraction, color=BLUE, marker="o", ms=3.5, lw=1.2, label="contraction")
    ax.scatter(ns, known, color=ORANGE, marker="x", s=30, linewidth=1.3, label="known (OEIS A000170)")
    for n, value in zip(ns, contraction):
        ax.annotate(str(value), (n, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_yticks(10.0 ** np.arange(0, 4))
    ax.set_xlabel("board size N")
    ax.set_ylabel("valid boards (log scale)")
    ax.set_title("(a) Exact N-queens counts", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    sample = load("sample_nq6")
    items = sorted(sample["frequencies"].items(), key=lambda item: json.loads(item[0]))
    counts = [value for _, value in items]
    positions = np.arange(len(items))
    ax.bar(positions, counts, color=BLUE, width=0.68)
    for x, count in zip(positions, counts):
        ax.text(x, count + 1.5, str(count), ha="center", fontsize=7)
    ax.axhline(50, color="0.4", lw=0.9, ls="--")
    ax.text(3.35, 60.0, "uniform", ha="right", fontsize=7, color="0.3")
    ax.set_xticks([])
    ax.set_ylim(0, 70)
    ax.set_xlabel("the 4 solutions")
    ax.set_ylabel("times drawn")
    ax.set_title("(b) 200 sampled 6-queens boards", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (key, _) in enumerate(items):
        inset = ax.inset_axes([0.04 + i * 0.25, -0.37, 0.15, 0.27], transform=ax.transAxes)
        draw_board(inset, json.loads(key))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.88)
    fig.savefig(OUT / "nqueens.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    validate()
    plot_gates()
    plot_factoring()
    plot_nqueens()
    print("PASS: all saved runs satisfy the Week 11 checks")
    print("wrote artifacts/gates.png, artifacts/factoring.png, artifacts/nqueens.png")
