from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

from common import ARTIFACTS, EVIDENCE, INPUTS, ROOT, load_settings, write_json


def relation_scores(table: np.ndarray, vocab: dict[str, int], words: list[str]) -> np.ndarray:
    ids = [vocab[word] for word in words]
    rows = table[ids].astype(np.float64)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    offsets = rows[1::2] - rows[0::2]
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    scores = np.empty(len(offsets), dtype=np.float64)
    for i in range(len(offsets)):
        mean_other = offsets[np.arange(len(offsets)) != i].mean(axis=0)
        mean_other /= np.linalg.norm(mean_other)
        scores[i] = float(offsets[i] @ mean_other)
    return scores


def unit_arrow(cosine: float) -> tuple[float, float]:
    """Return the upper-half unit vector with the requested horizontal endpoint."""
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return cosine, float(np.sqrt(max(0.0, 1.0 - cosine**2)))


def make_figure(
    relations: list[list[str]],
    names: list[str],
    found: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    output: "ARTIFACTS | None" = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), sharey=True)
    for ax, words, name in zip(axes, relations, names):
        pairs = [f"{words[i]}→{words[i + 1]}" for i in range(0, len(words), 2)]
        xs = np.arange(len(pairs), dtype=np.float64) * 2.4
        for base, scores, direction, color in (
            (0.65, found[name], 1.0, "#2878b5"),
            (-0.65, control[name], -1.0, "#8a8a8a"),
        ):
            for i, score in enumerate(scores):
                horizontal, vertical = unit_arrow(score)
                ax.arrow(
                    xs[i],
                    base,
                    horizontal,
                    direction * vertical,
                    width=0.006,
                    head_width=0.045,
                    head_length=0.08,
                    length_includes_head=True,
                    color=color,
                    alpha=0.82,
                )
            ax.plot([], [], color=color, label=f"mean = {scores.mean():.3f}")
        ax.axhline(0, color="black", lw=1.1)
        ax.vlines(xs, -0.05, 0.05, color="black", lw=0.7, ls="--", alpha=0.65)
        ax.set_title(name)
        ax.set_xticks(xs, pairs, rotation=55, ha="right", fontsize=8)
        ax.set_xlim(-1.2, xs[-1] + 1.2)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([-0.65, 0.65], ["shuffled control", "embedding table"])
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        ax.grid(axis="x", alpha=0.12)
    axes[0].set_ylabel("unit offset projected above/below horizontal")
    fig.suptitle("Relation offsets and the seed-1730 shuffled control")
    fig.tight_layout()
    EVIDENCE.mkdir(exist_ok=True)
    fig.savefig(EVIDENCE / "part1-embedding-relations.png", dpi=180)
    plt.close(fig)


def main() -> None:
    settings = load_settings()
    p1 = settings["part1"]
    with (ROOT / settings["paths"]["embedding_tokenizer"]).open(encoding="utf-8") as handle:
        tokenizer = json.load(handle)
    vocab = {token: int(idx) for token, idx in tokenizer["model"]["vocab"].items()}
    table = load_file(str(ROOT / settings["paths"]["embeddings"]))["embeddings"]
    assert table.shape == (29528, 256)

    relations: list[list[str]] = p1["relations"]
    names: list[str] = p1["relation_names"]
    for words in relations:
        missing = [word for word in words if word not in vocab]
        if missing:
            raise KeyError(f"words are not single potion tokens: {missing}")

    found = {name: relation_scores(table, vocab, words) for name, words in zip(names, relations)}
    primary_rng = np.random.default_rng(int(p1["seed"]))
    shuffle = primary_rng.permutation(len(table))
    control = {
        name: relation_scores(table[shuffle], vocab, words)
        for name, words in zip(names, relations)
    }

    controls: list[dict[str, float]] = []
    for seed in range(int(p1["seed"]), int(p1["seed"]) + int(p1["control_shuffles"])):
        permutation = np.random.default_rng(seed).permutation(len(table))
        controls.append(
            {
                name: float(relation_scores(table[permutation], vocab, words).mean())
                for name, words in zip(names, relations)
            }
        )

    metrics: dict[str, object] = {
        "shape": list(table.shape),
        "primary_control_seed": int(p1["seed"]),
        "relations": {},
        "control_shuffle_max_abs": {},
        "checks": {},
    }
    for name, words in zip(names, relations):
        pairs = [f"{words[i]}->{words[i + 1]}" for i in range(0, len(words), 2)]
        relation_metrics = {
            "pairs": pairs,
            "pair_cosines": found[name].tolist(),
            "mean": float(found[name].mean()),
            "control_pair_cosines": control[name].tolist(),
            "control_mean": float(control[name].mean()),
        }
        metrics["relations"][name] = relation_metrics
        max_abs = max(abs(row[name]) for row in controls)
        metrics["control_shuffle_max_abs"][name] = max_abs
        metrics["checks"][name] = {
            "table_mean_at_least_0.3": bool(found[name].mean() >= 0.3),
            "primary_control_within_0.15": bool(abs(control[name].mean()) <= 0.15),
            "twenty_shuffle_max_within_0.15": bool(max_abs <= 0.15),
        }

    make_figure(relations, names, found, control)
    metrics["all_checks_pass"] = all(
        all(check.values()) for check in metrics["checks"].values()
    )
    write_json("part1-metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not metrics["all_checks_pass"]:
        raise SystemExit("Part 1 acceptance check failed")


if __name__ == "__main__":
    main()

