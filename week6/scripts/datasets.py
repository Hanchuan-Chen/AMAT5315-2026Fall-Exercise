"""Reusable readers and figures for the Week 6 datasets.

Part 1 reads MNIST (grayscale, 28x28, values 0-255) and CIFAR-10 (RGB, 32x32,
values 0-255) from the downloads in ``inputs/``, and draws one figure per
dataset: a labeled gallery plus the class-frequency plot of the two splits.
Images are drawn with ``interpolation="none"`` and at their own resolution, so
the figure shows native pixels and invents nothing.
"""

from __future__ import annotations

import gzip
import struct
import tarfile

import matplotlib.pyplot as plt
import numpy as np

MNIST_DIR = "inputs/mnist"
CIFAR_TAR = "inputs/cifar10/cifar-10-binary.tar.gz"

DIGIT_NAMES = [str(d) for d in range(10)]
CIFAR_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_mnist_images(path: str) -> np.ndarray:
    """Read an IDX3 image file into ``(N, 28, 28)`` uint8."""
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        assert magic == 2051, f"{path}: not an IDX3 file"
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_mnist_labels(path: str) -> np.ndarray:
    """Read an IDX1 label file into ``(N,)`` uint8."""
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        assert magic == 2049, f"{path}: not an IDX1 file"
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def read_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)`` for MNIST."""
    x_train = read_mnist_images(f"{MNIST_DIR}/train-images-idx3-ubyte.gz")
    y_train = read_mnist_labels(f"{MNIST_DIR}/train-labels-idx1-ubyte.gz")
    x_test = read_mnist_images(f"{MNIST_DIR}/t10k-images-idx3-ubyte.gz")
    y_test = read_mnist_labels(f"{MNIST_DIR}/t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


def _read_cifar_member(archive: tarfile.TarFile, name: str):
    member = archive.extractfile(name)
    assert member is not None, f"{name}: missing from the CIFAR-10 archive"
    raw = np.frombuffer(member.read(), dtype=np.uint8)
    records = raw.reshape(-1, 1 + 3 * 32 * 32)
    labels = records[:, 0].astype(np.uint8)
    planes = records[:, 1:].reshape(-1, 3, 32, 32)
    return np.ascontiguousarray(planes.transpose(0, 2, 3, 1)), labels


def read_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)`` for CIFAR-10."""
    with tarfile.open(CIFAR_TAR, "r:gz") as archive:
        members = [
            f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)
        ]
        trains = [_read_cifar_member(archive, name) for name in members]
        x_test, y_test = _read_cifar_member(
            archive, "cifar-10-batches-bin/test_batch.bin"
        )
    x_train = np.concatenate([images for images, _ in trains])
    y_train = np.concatenate([labels for _, labels in trains])
    return x_train, y_train, x_test, y_test


def class_fractions(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Fraction of records carrying each class: the empirical p-hat of Eq. 2."""
    return np.bincount(labels, minlength=n_classes) / labels.size


def gallery_figure(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    per_class: int,
    title: str,
    cmap: str | None = None,
) -> tuple[plt.Figure, list[float]]:
    """A labeled gallery: one row per class, ``per_class`` examples per row.

    Cells are laid out in inches and drawn with ``interpolation="none"``, so
    each image occupies a square box at its own resolution and no smoothing or
    invented detail appears.  Returns the figure and the rectangle, in figure
    fractions, left free underneath the gallery for the frequency panel.
    """
    n_classes = len(class_names)
    label_in, tile_in, right_in, title_in = 0.85, 0.42, 0.06, 0.52
    gap_in, freq_in, bottom_in = 0.35, 1.60, 0.80
    fig_w = label_in + per_class * tile_in + right_in
    fig_h = title_in + n_classes * tile_in + gap_in + freq_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def rect(x_in: float, y_in: float, w_in: float, h_in: float) -> list[float]:
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    gallery_top = fig_h - title_in
    for row in range(n_classes):
        picks = np.flatnonzero(labels == row)[:per_class]
        for col, index in enumerate(picks):
            ax = fig.add_axes(
                rect(
                    label_in + col * tile_in,
                    gallery_top - (row + 1) * tile_in,
                    tile_in,
                    tile_in,
                )
            )
            ax.imshow(images[index], cmap=cmap, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{labels[index]}", fontsize=6, pad=1.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.text(
            (label_in - 0.06) / fig_w,
            (gallery_top - (row + 0.5) * tile_in) / fig_h,
            class_names[row],
            ha="right",
            va="center",
            fontsize=9,
        )
    fig.suptitle(title, fontsize=9, y=(fig_h - 0.20) / fig_h)
    freq_rect = rect(label_in, bottom_in, per_class * tile_in, freq_in)
    return fig, freq_rect


def frequency_axes(
    fig: plt.Figure,
    rect: list[float],
    train_fractions: np.ndarray,
    test_fractions: np.ndarray,
    class_names: list[str],
) -> plt.Axes:
    """Grouped class-frequency bars for the training and test splits."""
    ax = fig.add_axes(rect)
    x = np.arange(len(class_names))
    width = 0.4
    bars_train = ax.bar(x - width / 2, train_fractions, width, label="train")
    bars_test = ax.bar(x + width / 2, test_fractions, width, label="test")
    for bars, fractions in ((bars_train, train_fractions), (bars_test, test_fractions)):
        for bar, value in zip(bars, fractions):
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 1.5),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("class fraction", fontsize=8)
    ax.set_ylim(0, 1.26 * max(train_fractions.max(), test_fractions.max()))
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=7, loc="upper right")
    return ax


def dataset_parts(name: str):
    """Return ``(class_names, x_train, y_train, x_test, y_test)`` by name."""
    if name == "mnist":
        x_train, y_train, x_test, y_test = read_mnist()
        return DIGIT_NAMES, x_train, y_train, x_test, y_test
    if name == "cifar10":
        x_train, y_train, x_test, y_test = read_cifar10()
        return CIFAR_NAMES, x_train, y_train, x_test, y_test
    raise SystemExit(f"unknown dataset {name!r}")


def report(name: str, x_train, y_train, x_test, y_test, class_names) -> None:
    """Print the numbers the sheet's Part 1 VERIFY items ask for."""
    train_fractions = class_fractions(y_train)
    test_fractions = class_fractions(y_test)
    n0, n1 = x_train.shape[0], x_test.shape[0]
    print(f"{name}: train {n0} test {n1} images")
    print(f"{name}: train shape {x_train.shape[1:]} dtype {x_train.dtype}")
    print(
        f"{name}: pixel range {min(x_train.min(), x_test.min())}.."
        f"{max(x_train.max(), x_test.max())}"
    )
    counts = [
        (class_names[c], n0 * train_fractions[c], n1 * test_fractions[c])
        for c in range(len(class_names))
    ]
    print(f"{name}: train counts {[f'{c}:{int(v)}' for c, v, _ in counts]}")
    print(f"{name}: test counts {[f'{c}:{int(v)}' for c, _, v in counts]}")
    print(
        f"{name}: train fractions "
        f"{[f'{c}:{f:.4f}' for c, f in zip(class_names, train_fractions)]}"
    )
    print(
        f"{name}: test fractions "
        f"{[f'{c}:{f:.4f}' for c, f in zip(class_names, test_fractions)]}"
    )
    print(
        f"{name}: most common train {class_names[train_fractions.argmax()]} "
        f"({train_fractions.max():.4f}) and test "
        f"{class_names[test_fractions.argmax()]} ({test_fractions.max():.4f})"
    )
    print(
        f"{name}: least common train {class_names[train_fractions.argmin()]} "
        f"({train_fractions.min():.4f}) and test "
        f"{class_names[test_fractions.argmin()]} ({test_fractions.min():.4f})"
    )
    print(f"{name}: every class fraction exactly 0.1? "
          f"{bool(np.allclose(train_fractions, 0.1) and np.allclose(test_fractions, 0.1))}")
