"""Part 1 evidence: one labeled gallery plus class frequencies per dataset.

Usage: ``.venv/bin/python scripts/dataset_figures.py mnist`` writes
``evidence/mnist.png`` from the training and test splits, drawing each image at
its native resolution; ``... cifar10`` writes ``evidence/cifar10.png``.  Both
print the counts, image shapes, pixel range and class fractions the sheet's
VERIFY items ask to confirm.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets


def main(name: str) -> None:
    class_names, x_train, y_train, x_test, y_test = datasets.dataset_parts(name)
    datasets.report(name, x_train, y_train, x_test, y_test, class_names)

    if name == "mnist":
        per_class, cmap = 10, "gray"
        title = "MNIST: first 10 training images per digit (28x28 grayscale, 0-255)"
    else:
        per_class, cmap = 8, None
        title = "CIFAR-10: first 8 training images per class (32x32 RGB, 0-255)"

    fig, freq_rect = datasets.gallery_figure(
        x_train, class_names, per_class, title, labels=y_train, cmap=cmap
    )
    datasets.frequency_axes(
        fig,
        freq_rect,
        datasets.class_fractions(y_train),
        datasets.class_fractions(y_test),
        class_names,
    )
    out = f"evidence/{name}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("mnist", "cifar10"):
        raise SystemExit("usage: dataset_figures.py {mnist|cifar10}")
    main(sys.argv[1])
