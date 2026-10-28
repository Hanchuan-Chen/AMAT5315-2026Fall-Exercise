from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from common import ARTIFACTS, EVIDENCE, INPUTS, load_settings, write_json

STORES = ("outer", "relu", "relu2", "softmax")


def find_cjk_font() -> Path:
    candidates = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("no CJK font found for the 32x32 character pattern")


def make_picture(character: str) -> np.ndarray:
    size = 256
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(find_cjk_font()), 220)
    box = draw.textbbox((0, 0), character, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text(((size - width) / 2 - box[0], (size - height) / 2 - box[1]), character, 255, font=font)
    small = canvas.resize((32, 32), Image.Resampling.LANCZOS)
    ink = np.asarray(small) >= 72
    picture = np.where(ink, 1.0, -1.0)
    preview = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8), mode="L")
    preview.save(INPUTS / "part4-picture.png")
    return picture


def unit_rows(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    keys = rng.normal(size=(count, dimension))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    return keys


def random_pictures(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.choice(np.asarray([-1.0, 1.0]), size=(count, 1024))


def recall(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    store: str,
    beta: float,
) -> np.ndarray:
    similarities = queries @ keys.T
    if store == "outer":
        weights = similarities
    elif store == "relu":
        weights = np.maximum(similarities, 0.0)
    elif store == "relu2":
        weights = np.maximum(similarities, 0.0) ** 2
    elif store == "softmax":
        weights = np.exp(beta * (similarities - 1.0))
    else:
        raise ValueError(store)
    own = weights[np.arange(len(queries)), np.arange(len(queries))]
    return (weights @ values) / own[:, None], own


def measure(
    settings: dict, picture: np.ndarray
) -> tuple[dict, dict, dict]:
    p4 = settings["part4"]
    beta = float(p4["softmax_beta"])
    dimension = int(p4["dimension"])
    memory_grid = [int(value) for value in p4["memories"]]
    draws = int(p4["draws"])
    all_ratios = {str(m): {store: [] for store in STORES} for m in memory_grid}
    figure_recalls: dict[str, dict[str, np.ndarray]] = {}
    figure_ratios: dict[str, dict[str, float]] = {}

    for memory in memory_grid:
        for draw in range(draws):
            rng = np.random.default_rng(10000 * memory + draw + int(p4["seed"]))
            keys = unit_rows(rng, memory, dimension)
            values = random_pictures(rng, memory)
            values[0] = picture.ravel()
            for store in STORES:
                recalled, own = recall(keys[:1], keys, values, store, beta)
                flat_picture = picture.ravel()
                ratio = float(np.linalg.norm(recalled[0] - flat_picture) / np.linalg.norm(flat_picture))
                all_ratios[str(memory)][store].append(ratio)
                if draw == 0:
                    figure_recalls.setdefault(str(memory), {})[store] = recalled[0].reshape(32, 32)
                    figure_ratios.setdefault(str(memory), {})[store] = ratio

    medians = {
        memory: {
            store: float(np.median(values)) for store, values in stores.items()
        }
        for memory, stores in all_ratios.items()
    }
    return medians, figure_recalls, figure_ratios


def make_figure(
    picture: np.ndarray,
    recalls: dict[str, dict[str, np.ndarray]],
    figure_ratios: dict[str, dict[str, float]],
) -> None:
    memories = list(recalls)
    fig, axes = plt.subplots(
        len(memories), len(STORES) + 1, figsize=(2.55 * (len(STORES) + 1), 2.55 * len(memories))
    )
    y, x = np.mgrid[0:32, 0:32]
    flat_y, flat_x = y.ravel(), x.ravel()
    for row, memory in enumerate(memories):
        panels = [("stored", picture, float("nan"))] + [
            (store, recalls[memory][store], figure_ratios[memory][store])
            for store in STORES
        ]
        for column, (store, image, ratio) in enumerate(panels):
            ax = axes[row, column]
            flat = image.ravel()
            ink = flat >= 0
            sizes = np.where(ink, 7.0 + 8.0 * np.minimum(np.abs(flat), 3), 1.1)
            colors = np.where(ink, "#c0392b" if np.isfinite(ratio) and ratio < 1 else "#2878b5", "#d8d8d8")
            ax.scatter(flat_x, flat_y, s=sizes, c=colors, marker="o", linewidths=0)
            ax.set_xlim(-1, 32)
            ax.set_ylim(31, -2)
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(store)
            if column == 0:
                ax.text(-0.1, 0.5, f"M={memory}", transform=ax.transAxes, rotation=90, va="center", ha="right")
            if store != "stored":
                ax.text(
                    0.5,
                    -0.07,
                    f"R draw 0={ratio:.3g}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
    fig.suptitle("Recall of the 32×32 pixel character: dots show recalled pixel sign and magnitude")
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.96))
    fig.savefig(EVIDENCE / "part4-picture-recall.png", dpi=180)
    plt.close(fig)


def main() -> None:
    settings = load_settings()
    picture = make_picture(str(settings["part4"]["character"]))
    medians, recalls, figure_ratios = measure(settings, picture)
    make_figure(picture, recalls, figure_ratios)
    np.savez_compressed(ARTIFACTS / "part4-picture.npz", picture=picture, **{
        f"recall_m{memory}_{store}": image
        for memory, images in recalls.items()
        for store, image in images.items()
    })
    outer_1024 = medians["1024"]["outer"]
    relu_1024 = medians["1024"]["relu"]
    checks = {
        "outer_at_1024_within_15pct": {
            "pass": abs(outer_1024 / 4.0 - 1.0) <= 0.15,
            "actual": outer_1024,
            "reference_formula": 4.0,
        },
        "relu_at_1024_within_15pct": {
            "pass": abs(relu_1024 / (1024.0 / 128.0) ** 0.5 - 1.0) <= 0.15,
            "actual": relu_1024,
            "reference_formula": (1024.0 / 128.0) ** 0.5,
        },
        "relu2_at_1024_readable": {
            "pass": medians["1024"]["relu2"] < 1.0,
            "actual": medians["1024"]["relu2"],
        },
        "relu2_at_4096_above_one": {
            "pass": medians["4096"]["relu2"] > 1.0,
            "actual": medians["4096"]["relu2"],
        },
        "softmax_below_0.1_everywhere": {
            "pass": all(medians[memory]["softmax"] < 0.1 for memory in medians),
            "actual": {memory: medians[memory]["softmax"] for memory in medians},
        },
    }
    metrics = {
        "character": settings["part4"]["character"],
        "picture_sha256": hashlib.sha256(picture.astype(np.int8).tobytes()).hexdigest(),
        "ink_fraction": float((picture > 0).mean()),
        "dimension": settings["part4"]["dimension"],
        "beta": settings["part4"]["softmax_beta"],
        "draws": settings["part4"]["draws"],
        "ratio_draw0": figure_ratios,
        "ratio_medians": medians,
        "checks": checks,
        "all_checks_pass": all(item["pass"] for item in checks.values()),
    }
    write_json("part4-metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not metrics["all_checks_pass"]:
        raise SystemExit("Part 4 acceptance check failed")


if __name__ == "__main__":
    main()

