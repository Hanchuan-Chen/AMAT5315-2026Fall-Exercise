"""Contract checks for Part 1's datasets; needs `make data` to have run."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

import datasets


def test_mnist_counts_shape_and_range():
    x_train, y_train, x_test, y_test = datasets.read_mnist()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert x_train.dtype == np.uint8
    assert x_train.min() == 0 and x_train.max() == 255
    assert y_train.shape == (60000,) and y_test.shape == (10000,)


def test_mnist_class_frequencies():
    _, y_train, _, y_test = datasets.read_mnist()
    train = datasets.class_fractions(y_train)
    test = datasets.class_fractions(y_test)
    assert np.isclose(train.sum(), 1.0) and np.isclose(test.sum(), 1.0)
    assert train.argmax() == 1 and test.argmax() == 1
    assert train.argmin() == 5 and test.argmin() == 5


def test_cifar10_is_exactly_balanced():
    x_train, y_train, x_test, y_test = datasets.read_cifar10()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert x_train.dtype == np.uint8
    assert x_train.max() == 255
    assert np.allclose(datasets.class_fractions(y_train), 0.1)
    assert np.allclose(datasets.class_fractions(y_test), 0.1)
    assert np.all(np.bincount(y_train, minlength=10) == 5000)
    assert np.all(np.bincount(y_test, minlength=10) == 1000)


def test_gallery_and_frequency_figure_builds(tmp_path):
    x_train, y_train, _, _ = datasets.read_mnist()
    fig, rect = datasets.gallery_figure(
        x_train, y_train, datasets.DIGIT_NAMES, 3, "test", cmap="gray"
    )
    datasets.frequency_axes(
        fig,
        rect,
        datasets.class_fractions(y_train),
        datasets.class_fractions(y_train),
        datasets.DIGIT_NAMES,
    )
    out = tmp_path / "gallery.png"
    fig.savefig(out, dpi=40)
    assert out.exists() and out.stat().st_size > 0
