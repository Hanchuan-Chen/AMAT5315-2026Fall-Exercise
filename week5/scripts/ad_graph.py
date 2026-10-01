#!/usr/bin/env python3
"""Part 1, check (2): draw the jaxpr graphs of U and grad(U).

One node per operation, labelled with the operation's name; edges follow the
jaxpr's data dependencies, so the gradient graph shows `add_any` joining the two
contributions to `a`.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WEEK = Path(__file__).resolve().parents[1]
OUT = WEEK / "artifacts" / "ad"


def is_literal(value):
    return hasattr(value, "val")


def literal_text(value):
    number = float(value.val)
    return repr(number)


def energy(r):
    a = r**-6
    b = a**2
    c = b - a
    return 4 * c


def primitive_label(equation):
    name = equation.primitive.name
    exponent = equation.params.get("y")
    if name.startswith("integer_pow") and exponent is not None:
        return f"integer_pow[y={exponent}]"
    return name


def draw(jaxpr, title, path):
    nodes = {}

    def node(key, label, kind):
        if key not in nodes:
            nodes[key] = {"label": label, "kind": kind, "inputs": []}
        return key

    def key_of(value):
        return id(value)

    def label_of(value):
        if is_literal(value):
            return literal_text(value), "const"
        return None, "var"

    for index, variable in enumerate(jaxpr.invars):
        key = node(key_of(variable), "r" if index == 0 else "input", "input")
        if index != 0:
            nodes[key]["label"] = str(variable)
    for equation in jaxpr.eqns:
        target = key_of(equation.outvars[0])
        label = primitive_label(equation)
        kind = "add" if label == "add_any" else "op"
        node(target, label, kind)
        for value in equation.invars:
            if is_literal(value):
                text, const_kind = label_of(value)
                source = node(key_of(value), text, const_kind)
            else:
                text, const_kind = label_of(value)
                source = node(key_of(value), str(value), const_kind)
            nodes[target]["inputs"].append(source)
    for index, variable in enumerate(jaxpr.outvars):
        if is_literal(variable):
            source = node(key_of(variable), literal_text(variable), "const")
        else:
            source = node(key_of(variable), str(variable), "var")
        out = node(("out", index), "output", "output")
        nodes[out]["inputs"].append(source)

    layer = {}

    def depth(key):
        if key in layer:
            return layer[key]
        inputs = nodes[key]["inputs"]
        layer[key] = 0 if not inputs else 1 + max(depth(other) for other in inputs)
        return layer[key]

    order = list(nodes)
    for key in order:
        depth(key)
    columns = {}
    for key in order:
        columns.setdefault(layer[key], []).append(key)
    widths = {key: max(0.9, 0.13 * len(nodes[key]["label"]) + 0.32) for key in nodes}
    gap = 0.55
    left = {}
    cursor = 0.0
    for level in sorted(columns):
        left[level] = cursor
        cursor += max(widths[key] for key in columns[level]) + gap
    positions = {
        key: (left[layer[key]] + widths[key] / 2, -row * 0.95)
        for level, keys in columns.items()
        for row, key in enumerate(keys)
    }
    fig, ax = plt.subplots(
        figsize=(0.62 * cursor + 1.2, 0.95 * max(len(v) for v in columns.values()) + 1.1)
    )
    colors = {
        "var": "#eeeeee",
        "input": "#dbe7f2",
        "const": "#f2ecd8",
        "op": "#e6e6ef",
        "add": "#f6d6cc",
        "output": "#dcefe0",
    }
    for key, info in nodes.items():
        x, y = positions[key]
        width = widths[key]
        box = FancyBboxPatch(
            (x - width / 2, y - 0.3),
            width,
            0.6,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=colors[info["kind"]],
            edgecolor="#4a4a4a" if info["kind"] != "add" else "#a33a20",
            linewidth=1.4 if info["kind"] == "add" else 0.8,
        )
        ax.add_patch(box)
        ax.text(x, y, info["label"], ha="center", va="center", fontsize=7.4)
        for source in info["inputs"]:
            sx, sy = positions[source]
            ax.add_patch(
                FancyArrowPatch(
                    (sx + widths[source] / 2, sy),
                    (x - width / 2, y),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color="#5a5a5a",
                    linewidth=0.8,
                    shrinkA=1,
                    shrinkB=1,
                )
            )
    rightmost = left[max(columns)] + max(widths[key] for key in columns[max(columns)])
    ax.set(
        xlim=(-0.5, rightmost + 0.5),
        ylim=(-1.2 * max(len(v) for v in columns.values()), 1.0),
    )
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return nodes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = jnp.float64(1.3)
    primal = jax.make_jaxpr(energy)(r)
    gradient = jax.make_jaxpr(jax.grad(energy))(r)
    nodes = draw(primal, "$U$ at $r=1.3$", OUT / "graph.png")
    grad_nodes = draw(gradient, "$\\nabla U$ at $r=1.3$", OUT / "grad-graph.png")
    labels = [info["label"] for info in grad_nodes.values()]
    assert any(label == "add_any" for label in labels), labels
    print("primal operations:", [info["label"] for info in nodes.values() if info["kind"] == "op"])
    print("gradient operations:", [label for label in labels if label not in ("r", "output")])
    print("gradient graph contains add_any joining two contributions to a: PASS")


if __name__ == "__main__":
    main()
