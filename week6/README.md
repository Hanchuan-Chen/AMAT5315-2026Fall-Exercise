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
