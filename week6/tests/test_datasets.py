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
        x_train, datasets.DIGIT_NAMES, 3, "test", labels=y_train, cmap="gray"
    )
    tiles = fig.axes
    assert len(tiles) == len(datasets.DIGIT_NAMES) * 3
    # One row per class: the first tile of each row is that class's first image.
    for digit in range(len(datasets.DIGIT_NAMES)):
        first = np.flatnonzero(y_train == digit)[0]
        assert np.array_equal(tiles[digit * 3].images[0].get_array(), x_train[first])
    for tile in tiles:
        assert tile.images[0].get_array().shape == (28, 28)
        assert tile.images[0].get_interpolation() == "none"

    fractions = datasets.class_fractions(y_train)
    ax = datasets.frequency_axes(fig, rect, fractions, fractions, datasets.DIGIT_NAMES)
    heights = np.array([patch.get_height() for patch in ax.patches])
    assert heights.shape == (2 * len(datasets.DIGIT_NAMES),)
    assert np.allclose(heights[:10], fractions)
    assert np.allclose(heights[10:], fractions)
    out = tmp_path / "gallery.png"
    fig.savefig(out, dpi=40)
    assert out.stat().st_size > 0


def test_gallery_without_labels_takes_the_stored_order():
    x_train, _, _, _ = datasets.read_mnist()
    fig, _ = datasets.gallery_figure(
        x_train, datasets.DIGIT_NAMES, 2, "no labels", cmap="gray"
    )
    drawn = [tile.images[0].get_array() for tile in fig.axes]
    assert len(drawn) == 20
    assert np.array_equal(np.asarray(drawn[0]), x_train[0])
    assert np.array_equal(np.asarray(drawn[-1]), x_train[19])
