#!/usr/bin/env python3
"""Part 4, checks (1) and (2): schedule plot and the storage/recomputation curve."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK = Path(__file__).resolve().parents[1]
ART = WEEK / "artifacts"


def main():
    budgets = []
    for delta in (1, 3, 5, 10):
        stats = json.loads((ART / f"checkpoint-{delta}" / "result.json").read_text())["statistics"]
        budgets.append(
            dict(
                delta=delta,
                peak_saved_states=stats["peak_saved_states"],
                peak_saved_bytes=stats["peak_saved_bytes"],
                scheduler_forward_calls=stats["per_shot"][0]["scheduler_forward_calls"],
            )
        )
    full = json.loads((ART / "adjoint" / "result.json").read_text())["statistics"]
    full_row = dict(
        delta=0,
        peak_saved_states=full["peak_saved_states"],
        peak_saved_bytes=full["peak_saved_bytes"],
        scheduler_forward_calls=full["per_shot"][0]["scheduler_forward_calls"],
    )
    actions = json.loads((ART / "checkpoint-5" / "actions-0.json").read_text())

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.2, 3.2), layout="constrained")
    ax.plot(np.arange(len(actions)), [event["step"] for event in actions], color="#cccccc", lw=0.4, zorder=0)
    for action, color, label in (
        ("call", "#24557a", "Call"),
        ("store", "#34845a", "Store"),
        ("restore", "#8b55a1", "Restore"),
        ("grad", "#ba4b37", "Grad"),
        ("fetch", "#c08a21", "Fetch"),
    ):
        points = np.array([(index, event["step"]) for index, event in enumerate(actions) if event["action"] == action])
        size = 4 if action == "call" else 7
        ax.scatter(points[:, 0], points[:, 1], s=size, color=color, label=label, rasterized=True)
    ax.set(xlabel="Operation index", ylabel="Time step", title="Treeverse schedule; reflector shot 0, budget 5")
    ax.legend(ncol=5, fontsize=8, loc="upper right", markerscale=2)
    fig.savefig(ART / "checkpoint-actions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
    delta = [row["delta"] for row in budgets]
    axes[0].semilogy(delta, [row["scheduler_forward_calls"] for row in budgets], "o-", color="#24557a", label="Treeverse")
    axes[0].axhline(full_row["scheduler_forward_calls"], color="#888888", ls=":", lw=1, label="Full history")
    axes[0].set(
        xlabel="Additional checkpoint slots $\\delta$",
        ylabel="Forward steps of the schedule per shot",
        title="Recomputation cost",
    )
    axes[0].legend(fontsize=7)
    axes[1].plot(delta, [row["peak_saved_bytes"] for row in budgets], "o-", color="#24557a")
    for row in budgets:
        axes[1].annotate(
            f"{row['peak_saved_states']} states",
            (row["delta"], row["peak_saved_bytes"]),
            textcoords="offset points",
            xytext=(4, -9),
            fontsize=7,
            color="#555555",
        )
    axes[1].set(
        xlabel="Additional checkpoint slots $\\delta$",
        ylabel="Peak saved-state bytes",
        title="Two wavefields per saved state",
        ylim=(0, max(row["peak_saved_bytes"] for row in budgets) * 1.3),
    )
    axes[1].text(
        0.04,
        0.9,
        f"Full history: {full_row['peak_saved_bytes']:,} bytes ({full_row['peak_saved_states']} states)",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    fig.savefig(ART / "checkpoint-work.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    full_image = np.load(ART / "adjoint" / "image.npy")
    for row in budgets:
        image = np.load(ART / f"checkpoint-{row['delta']}" / "image.npy")
        relative = float(np.linalg.norm(image - full_image) / np.linalg.norm(full_image))
        print(
            f"checkpoint-{row['delta']}: relative L2 error {relative:.1e}, peak {row['peak_saved_states']} states, "
            f"{row['peak_saved_bytes']:,} bytes, {row['scheduler_forward_calls']} forward steps per shot"
        )
        assert relative < 1e-9
    print(
        f"full history: {full_row['scheduler_forward_calls']} forward steps per shot, "
        f"{full_row['peak_saved_states']} states, {full_row['peak_saved_bytes']:,} bytes"
    )


if __name__ == "__main__":
    main()
