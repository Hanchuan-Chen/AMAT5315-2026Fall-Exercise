# Week 6: energy-based models

This directory holds the Week 6 exercise: the MNIST and CIFAR-10 datasets and
their measured class frequencies, a JAX restricted Boltzmann machine trained by
contrastive divergence on MNIST, a JAX classical Hopfield network whose recall
collapses near the storage capacity, and the modern Hopfield update that is one
step of softmax attention. The contract is `rbm.toml`, copied verbatim from the
learning sheet.

This README regenerates every committed file from a clean clone. Run every
command from `week6/`.

## Repository layout

```text
week6/
  Makefile                      `make reproduce`, `make test`, `make data`
  README.md                     this document
  rbm.toml                      Part 2's settings block, verbatim from the sheet
  scripts/                      the download, dataset, RBM, Hopfield and attention scripts
  tests/                        the pytest checks behind `make test`
  evidence/                     the committed figures
  artifacts/                    raw runs and settings (generated, ignored)
  inputs/                       the dataset downloads (generated, ignored)
```

## Prerequisites

| tool | version this README was verified with | why |
| --- | --- | --- |
| `uv` | 0.12.9 | creates the Python virtual environment |
| `python3` | 3.12.14 | the interpreter `uv` uses |
| `curl` | /usr/bin/curl (macOS) | Part 1's dataset downloads |

Network access is needed once, for the two dataset downloads.

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib jax pytest
.venv/bin/python -c "import numpy, matplotlib, jax, pytest; \
  print(numpy.__version__, matplotlib.__version__, jax.__version__, pytest.__version__)"
```

Verified with numpy 2.5.4, matplotlib 3.11.2, jax 0.11.2 (CPU) and pytest 9.1.1.
Every Python command below is run from `week6/` as `.venv/bin/python ...`.

## Dataset provenance

Part 1's download commands, run by `scripts/download_data.sh`, appended the
following effective URLs to this file:

https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz

MNIST is Yann LeCun's handwritten-digit set as distributed by the CVDF mirror;
CIFAR-10 is the Alex Krizhevsky / Geoffrey Hinton binary release, obtained here
from the mirror the sheet's command names. Both downloads need no account, and
both keep their original split assignments: MNIST's `train-*` and `t10k-*`
files, and CIFAR-10's five `data_batch_*.bin` plus `test_batch.bin`.

## Part 1: the datasets and their empirical distributions

`scripts/datasets.py` is the reusable module: it reads the IDX files and the
CIFAR-10 binary records, and it draws a labeled gallery (one row per class) and
the class-frequency plot of the two splits. Gallery cells are sized in inches
and drawn with `interpolation="none"`, so each image occupies a square box at
its own resolution with no smoothing and no invented detail.
`scripts/dataset_figures.py` is one command per dataset:

```bash
.venv/bin/python scripts/dataset_figures.py mnist     # -> evidence/mnist.png
.venv/bin/python scripts/dataset_figures.py cifar10   # -> evidence/cifar10.png
```

Measured, printed by the two commands:

```text
mnist:   train 60000 test 10000 images, shape (28, 28), dtype uint8, pixels 0..255
mnist:   train fractions 0:0.0987 1:0.1124 2:0.0993 3:0.1022 4:0.0974
         5:0.0903 6:0.0986 7:0.1044 8:0.0975 9:0.0992
mnist:   most common train 1 (0.1124) and test 1 (0.1135)
mnist:   least common train 5 (0.0903) and test 5 (0.0892)
cifar10: train 50000 test 10000 images, shape (32, 32, 3), dtype uint8, pixels 0..255
cifar10: 5000 train / 1000 test per class, every fraction 0.1000
```

MNIST's test split repeats the training split's ordering: digit 1 is most
common and digit 5 least common in both. CIFAR-10 is exactly balanced, so its
frequency plot is ten pairs of equal bars at 0.1.

## Part 2: the RBM on MNIST

`rbm.toml` is the sheet's settings block, verbatim. `scripts/rbm_lib.py` holds
the model: the energy of Eq. 3, the visible free energy of Eq. 4, the block
Gibbs sampler of Eqs. 6-7, and the CD surrogate of Eq. 10 whose gradient is the
CD update of Eqs. 8-9 (checked in `tests/test_rbm.py` against the explicit
`E[v p(h=1|v)]` form). Everything is JAX on CPU with `float32`, the hot loops
are jitted, and the visible layer is binarized at half intensity.

Training both models writes the raw run into `artifacts/rbm/` (parameters,
settings record, and the histories of the surrogate and of the held-out
one-step reconstruction error):

```bash
.venv/bin/python scripts/rbm_train.py      # -> artifacts/rbm/{cd1,cd10}-params.npz
                                           #    artifacts/rbm/run.json
```

Measured, printed by that command and re-read by the figure script:

```text
cd1:  20 epochs over 60000 images, batch 100, 8.9 s
      reconstruction error 0.17084 at initialization -> 0.04616 (27.0% of it)
cd10: 20 epochs, 56.1 s
      reconstruction error 0.17084 at initialization -> 0.04652 (27.2% of it)
```

CD-1 and CD-10 share the initialization (seed 6601), the data order (seed
6602's per-epoch permutation and per-batch Gibbs keys) and the evaluation
randomness (seed 6606), so they differ only in the chain length. The sheet's
reference is about 17% -> 4.7% of pixels with 18 s and 105 s of training.

The four evidence figures:

```bash
.venv/bin/python scripts/rbm_spins.py        # -> evidence/rbm-spins.png
.venv/bin/python scripts/rbm_training.py     # -> evidence/rbm-training.png
.venv/bin/python scripts/rbm_generation.py   # -> evidence/rbm-generation.png
.venv/bin/python scripts/rbm_completion.py   # -> evidence/rbm-completion.png
```

* `rbm-spins.png`: the first 16 training images binarized, one enlarged spin
  configuration (ink +1, background -1), and the 784-to-128 coupling diagram.
* `rbm-training.png`: the CD surrogate and the held-out one-step reconstruction
  error against epoch; the surrogate rises (it need not decrease) while
  reconstruction improves.
* `rbm-generation.png`: two reconstruction chains and two fresh-generation
  chains per model at 1, 10, 100 and 1000 steps from matched starts, plus the
  two learning curves. Reconstruction keeps its starting digit; fresh
  generation develops blobs and rings and rarely a clean digit.
* `rbm-completion.png`: held-out digits with only the upper half observed, 8
  chains each, true and wrong clamps, both models, with the ink overlap against
  steps underneath.

## Part 3: the classical Hopfield network

`scripts/hopfield_lib.py` holds the Hebbian couplings of Eq. 12 with a zero
diagonal, the overlap, the unstable-bit fraction ``xi_i h_i < 0`` over all
stored patterns, and zero-temperature asynchronous recall ``s_i <- sign(h_i)``
in a fresh random order each sweep, leaving ``s_i`` unchanged when ``h_i = 0``
and stopping when a whole sweep flips nothing. The fields come from
``X (X^T s) / N`` rather than a materialized ``J``, so a sweep costs ``O(N P)``
per chain; `tests/test_hopfield.py` checks the vectorized recall against a
naive spin-by-spin loop and the unstable fraction against an explicit ``J``.

The sweep is ``alpha = P/N`` from 0.05 to 0.25 in steps of 0.01 for N = 100,
200, 400, 800, with 20 independent pattern sets per point:

```bash
.venv/bin/python scripts/hopfield_scan.py       # -> artifacts/hopfield/scan.npz, run.json
.venv/bin/python scripts/hopfield_unstable.py   # -> evidence/hopfield-unstable.png
.venv/bin/python scripts/hopfield_recall.py     # -> evidence/hopfield-recall.png
```

The pattern sets have fixed seeds, `numpy.random.default_rng([2026, N,
alpha_index, set_index])`, and the recall order uses
`jax.random.key(2026 * 10000 + N * 10 + alpha_index)`; both are recorded in
`artifacts/hopfield/run.json` beside the wall-clock time, which was 40.1 s for
the whole sweep (N = 100: 3.7 s, 200: 4.1 s, 400: 7.1 s, 800: 25.1 s), against
the sheet's reference of about 7 minutes for the same study.

Measured:

```text
unstable-bit fraction, alpha = 0.20   reference   measured
  N = 800                             1.25e-2     1.245e-2   (large-N 1.267e-2)
unstable-bit fraction, alpha = 0.10   reference   measured
  N = 100                             3.50e-4     2.500e-4   (large-N 7.827e-4,
                                                finite-N 4.556e-4)
mean-overlap crossing below 0.9      reference   measured
  N = 100                             0.186       0.1938
  N = 200                             0.170       0.1655
  N = 400                             0.156       0.1548
  N = 800                             0.150       0.1487
```

Every crossing is within the sheet's 0.02 and they decrease with N; the
unstable-bit fraction approaches the ``Phi(-1/sqrt(alpha))`` curve as N grows.
