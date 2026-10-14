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
  Makefile                      `make data`, `make venv`, `make reproduce`, `make test`
  README.md                     this document
  rbm.toml                      Part 2's settings block, verbatim from the sheet
  scripts/
    download_data.sh            Part 1's two dataset downloads
    datasets.py                 IDX and CIFAR-10 readers, gallery and frequency plots
    dataset_figures.py          Part 1's evidence, one command per dataset
    figlib.py                   the tiled-image mosaic helper
    rbm_lib.py                  Part 2's RBM: energy, free energy, Gibbs, CD training
    rbm_train.py                trains CD-1 and CD-10 into artifacts/rbm/
    rbm_spins.py, rbm_training.py, rbm_generation.py, rbm_completion.py
    hopfield_lib.py             Part 3's Hebbian couplings and asynchronous recall
    hopfield_scan.py            the load sweep into artifacts/hopfield/
    hopfield_unstable.py, hopfield_recall.py
    modern_lib.py               Part 4's modern update, attention form and scoring
    modern_scan.py              the P and beta scan into artifacts/modern/
    modern_weights.py, modern_beta.py, modern_retrieval.py
  tests/                        pytest checks behind `make test`
  evidence/                     the eleven committed figures
  artifacts/                    raw runs and settings (generated, ignored)
  inputs/                       the dataset downloads (generated, ignored)
```

Everything tracked under `week6/` is source, `rbm.toml`, this README, `tests/`
or committed evidence. `inputs/`, `artifacts/` and `.venv/` are ignored by
`week6/.gitignore` and are rebuilt by the commands below.

## Prerequisites

| tool | version this README was verified with | why |
| --- | --- | --- |
| `uv` | 0.12.9 | creates the Python virtual environment |
| `python3` | 3.12.14 | the interpreter `uv` uses |
| `curl` | /usr/bin/curl (macOS) | Part 1's dataset downloads |

Network access is needed once, for the two dataset downloads. Everything runs
on CPU.

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib jax pytest
.venv/bin/python -c "import numpy, matplotlib, jax, pytest; \
  print(numpy.__version__, matplotlib.__version__, jax.__version__, pytest.__version__)"
```

Verified with numpy 2.5.4, matplotlib 3.11.2, jax 0.11.2 (CPU; `jax.devices()`
reports `cpu`) and pytest 9.1.1. Every Python command below is run from `week6/`
as `.venv/bin/python ...`.

## Dataset provenance

Part 1's download commands, run by `scripts/download_data.sh`, appended the
following effective URLs to this file:

https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz

MNIST is Yann LeCun's handwritten-digit set as distributed by the CVDF mirror;
CIFAR-10 is the Alex Krizhevsky / Geoffrey Hinton binary release, obtained from
the mirror the sheet's command names. Both need no account, and both keep their
original split assignments: MNIST's `train-*` and `t10k-*` files, and CIFAR-10's
five `data_batch_*.bin` plus `test_batch.bin`. The script appends a URL only
when `README.md` does not already contain it, so re-running it (or `make
reproduce`) leaves this committed file unchanged.

## Part 1: the datasets and their empirical distributions

`scripts/datasets.py` is the reusable module: it reads the IDX files and the
CIFAR-10 binary records, and it draws a labeled gallery (one row per class, the
first images of each class in stored order) and the class-frequency plot of the
two splits. Gallery cells are sized in inches and drawn with
`interpolation="none"`, so each image occupies a square box at its own
resolution with no smoothing and no invented detail. `scripts/dataset_figures.py`
is one command per dataset:

```bash
bash scripts/download_data.sh                          # -> inputs/ (ignored)
.venv/bin/python scripts/dataset_figures.py mnist      # -> evidence/mnist.png    (0.8 s)
.venv/bin/python scripts/dataset_figures.py cifar10    # -> evidence/cifar10.png  (1.8 s)
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
frequency plot is ten pairs of equal bars at 0.1. The gallery shows what one
sample represents (handwriting varies within a digit class; a 32x32 photograph
keeps object shape and colour but loses most texture), while the frequency plot
shows the empirical distribution, which is what a model is later fitted to.

## Part 2: the RBM on MNIST

`rbm.toml` is the sheet's settings block, verbatim (only the wrapped
`wrong_upper` string is joined back onto one line, since a TOML string cannot
contain a raw newline). `scripts/rbm_lib.py` holds the model: the energy of
Eq. 3, the visible free energy of Eq. 4, the block Gibbs sampler of Eqs. 6-7,
and the CD surrogate of Eq. 10 whose gradient is the CD update of Eqs. 8-9.
`tests/test_rbm.py` checks the free energy against an explicit sum over the
hidden layer, the surrogate's gradient against the explicit
`E[v p(h=1|v)]` form, and the sampled hidden bits against `p(h=1|v)`. Everything
is JAX on CPU with `float32`, the hot loops are jitted, and the visible layer is
binarized at half intensity (`v = 1` for a pixel value at least 128).

Training both models writes the raw run into `artifacts/rbm/` (parameters, a
settings record, and the histories of the surrogate and of the held-out
one-step reconstruction error):

```bash
.venv/bin/python scripts/rbm_train.py       # -> artifacts/rbm/{cd1,cd10}-params.npz
                                            #    artifacts/rbm/run.json          (65.3 s)
```

Measured, printed by that command and re-read by the figure script:

```text
cd1:  20 epochs over 60000 images (the full training split), batch 100, 8.9 s
      reconstruction error 0.17084 at initialization -> 0.04616 (27.0% of it)
cd10: the same data order and evaluation randomness, chain length 10, 56.1 s
      reconstruction error 0.17084 at initialization -> 0.04652 (27.2% of it)
```

CD-1 and CD-10 share the initialization (seed 6601), the data order (seed
6602's per-epoch permutation and per-batch Gibbs keys) and the evaluation
randomness (seed 6606), so they differ only in their chain length. The sheet's
reference is about 17% -> 4.7% of pixels with 18 s and 105 s of training.

The four evidence figures:

```bash
.venv/bin/python scripts/rbm_spins.py        # -> evidence/rbm-spins.png        (0.5 s)
.venv/bin/python scripts/rbm_training.py     # -> evidence/rbm-training.png     (0.5 s)
.venv/bin/python scripts/rbm_generation.py   # -> evidence/rbm-generation.png   (2.2 s)
.venv/bin/python scripts/rbm_completion.py   # -> evidence/rbm-completion.png   (7.2 s)
```

* `rbm-spins.png`: the first 16 training images binarized, one enlarged spin
  configuration (ink +1, background -1), and the 784-to-128 coupling diagram;
  every edge joins a visible bit to a hidden unit and none joins two bits of the
  same layer.
* `rbm-training.png`: the CD surrogate and the held-out one-step reconstruction
  error against epoch. The surrogate rises from -17.3 to -5.8 (CD-1) and from
  -14.5 to -5.0 (CD-10) while reconstruction improves, which is why the
  surrogate alone is not a likelihood.
* `rbm-generation.png`: two reconstruction chains and two fresh-generation
  chains per model at 1, 10, 100 and 1000 steps from matched starts, plus the
  two learning curves. Both models start from the same held-out images and the
  same random spins (largest entry difference 0.0). Reconstruction keeps its
  starting digit at every length; fresh generation starts as noise and, by 1000
  steps, has 11 connected ink components against 65 after one step, i.e. blobs
  and rings of stroke-like edges rather than recognizable digits.
* `rbm-completion.png`: held-out digits with only the upper half observed, eight
  chains each, true and wrong clamps, both models, with the ink overlap against
  steps underneath.

The completion acceptance checks, printed by that command:

```text
observed upper half unchanged in every frame (largest difference 0 pixels),
  for both models and both clamps
true-half ink overlap after 1000 steps     reference   measured
  CD-1                                     0.40        0.416 +/- 0.023
  CD-10                                    0.51        0.510 +/- 0.017
wrong-half ink overlap after 1000 steps    reference   measured
  CD-1                                     0.24        0.225 +/- 0.016
  CD-10                                    0.27        0.253 +/- 0.015
gap (true - wrong)                          at least 0.1   0.191 (CD-1), 0.257 (CD-10)
one-step speckle: isolated-ink fraction 0.143 at 1 step against 0.027 at 1000
  steps; the true-half and wrong-half overlaps differ by only 0.019 at 1 step
```

## Part 3: the classical Hopfield network

`scripts/hopfield_lib.py` holds the Hebbian couplings of Eq. 12 with a zero
diagonal, the overlap, the unstable-bit fraction `xi_i h_i < 0` over all stored
patterns, and zero-temperature asynchronous recall `s_i <- sign(h_i)` in a
fresh random order each sweep, leaving `s_i` unchanged when `h_i = 0` and
stopping when a whole sweep flips nothing. The fields come from `X (X^T s) / N`
rather than a materialized `J`, so a sweep costs `O(N P)` per chain;
`tests/test_hopfield.py` checks the unstable fraction against an explicit `J`,
the two recall implementations against each other, and a zero-field state
against leaving every spin unchanged.

The sweep is `alpha = P/N` from 0.05 to 0.25 in steps of 0.01 for N = 100, 200,
400, 800, with 20 independent pattern sets per point:

```bash
.venv/bin/python scripts/hopfield_scan.py       # -> artifacts/hopfield/scan.npz, run.json (40.1 s)
.venv/bin/python scripts/hopfield_unstable.py   # -> evidence/hopfield-unstable.png (0.3 s)
.venv/bin/python scripts/hopfield_recall.py     # -> evidence/hopfield-recall.png   (0.3 s)
```

The pattern sets have fixed seeds, `numpy.random.default_rng([2026, N,
alpha_index, set_index])`, and the recall order uses
`jax.random.key(2026 * 10000 + N * 10 + alpha_index)`; both are recorded in
`artifacts/hopfield/run.json` beside the wall-clock time, which was 40.1 s for
the whole sweep (N = 100: 3.7 s, 200: 4.1 s, 400: 7.1 s, 800: 25.1 s), against
the sheet's reference of about 7 minutes for the same study.

Measured:

```text
unstable-bit fraction                   reference   measured    large-N    finite-N
  N = 800, alpha = 0.20                 1.25e-2     1.245e-2    1.267e-2   1.249e-2
  N = 100, alpha = 0.10                 3.50e-4     2.500e-4    7.827e-4   4.556e-4
mean-overlap crossing below 0.9         reference   measured
  N = 100                               0.186       0.1938
  N = 200                               0.170       0.1655
  N = 400                               0.156       0.1548
  N = 800                               0.150       0.1487
```

Every crossing is within the sheet's 0.02 and they decrease with N; the
unstable-bit fraction approaches the large-N crosstalk curve as N grows, and at
N = 100 it sits below it, where the finite-N form accounts for most of the gap.
The sheet's optional N = 1600 comparison row is not drawn.

## Part 4: the modern Hopfield network and attention

`scripts/modern_lib.py` stores the first `P` MNIST training images in stored
order as binarized spin columns of `X`, implements the update of Eq. 17,
`x <- X softmax(beta X^T x)`, computes the same update with JAX's
`jax.nn.dot_product_attention` (`x` as the query, the stored patterns as keys
and values, `scale=beta`), and scores retrieval as an overlap of at least 0.95
with the target *and* the target being the nearest stored digit to the signed
result. The classical comparison is Part 3's recall on the same patterns.

```bash
.venv/bin/python scripts/modern_scan.py        # -> artifacts/modern/scan.npz, run.json (3.1 s)
.venv/bin/python scripts/modern_weights.py     # -> evidence/modern-weights.png    (0.2 s)
.venv/bin/python scripts/modern_beta.py        # -> evidence/modern-beta.png       (1.5 s)
.venv/bin/python scripts/modern_retrieval.py   # -> evidence/modern-retrieval.png  (0.2 s)
```

The 100 queries cycle through the stored digits, each with its lower half
replaced by random spins drawn from `jax.random.key(6608)`; the classical recall
order uses `jax.random.key(6609)`. Both seeds and the whole run's wall-clock
time are recorded in `artifacts/modern/run.json`.

Measured:

```text
largest |modern update - softmax attention| over all queries, P, beta
  measured 1.19e-06    required below 1e-5    sheet's reference 6.0e-7
  (the library default scale 1/sqrt(784) would give about 2, so scale=beta)
queried digit's softmax weight, 100 stored digits      reference   measured
  beta = 0.003                                          0.01        0.0143
  beta = 0.03                                           0.25        0.2309
  beta = 0.3                                            1.00        1.0000
modern retrieval success at beta = 0.3                 0.98-1.00   1.00 1.00 0.99 1.00 0.99 0.97
  for P = 3, 10, 30, 100, 300, 1000
modern retrieval success at beta = 0.1 at P = 1000      0.91        0.88
classical recall success at every P                     0           0.00
```

Every modern success at `beta = 0.3` is at least 0.90, the classical recall
never succeeds, and the weights move from a uniform blend (0.01 = 1/100) to
one-hot retrieval as `beta` grows. `modern-beta.png` shows the same behaviour in
images: at `beta <= 0.01` any two queries' updates differ by only 0.003 on
average, so every query returns the same blurred average digit; at 0.03 the
queried digit emerges from that average; at `beta >= 0.1` one update returns the
stored digit, and the classical recall of the same queries returns the same
sparse fragment of strokes for all of them.

## Evidence inventory

Every committed evidence file, beside the exact command that produces it. The
raw runs those commands read (and, for Part 2, write first) live in the
git-ignored `artifacts/`.

| file | produced by |
| --- | --- |
| `evidence/mnist.png` | `.venv/bin/python scripts/dataset_figures.py mnist` |
| `evidence/cifar10.png` | `.venv/bin/python scripts/dataset_figures.py cifar10` |
| `evidence/rbm-spins.png` | `.venv/bin/python scripts/rbm_spins.py` |
| `evidence/rbm-training.png` | `.venv/bin/python scripts/rbm_training.py` |
| `evidence/rbm-generation.png` | `.venv/bin/python scripts/rbm_generation.py` |
| `evidence/rbm-completion.png` | `.venv/bin/python scripts/rbm_completion.py` |
| `evidence/hopfield-unstable.png` | `.venv/bin/python scripts/hopfield_unstable.py` |
| `evidence/hopfield-recall.png` | `.venv/bin/python scripts/hopfield_recall.py` |
| `evidence/modern-weights.png` | `.venv/bin/python scripts/modern_weights.py` |
| `evidence/modern-beta.png` | `.venv/bin/python scripts/modern_beta.py` |
| `evidence/modern-retrieval.png` | `.venv/bin/python scripts/modern_retrieval.py` |

The run scripts that the figures read, and everything they leave behind:

| command | writes |
| --- | --- |
| `bash scripts/download_data.sh` | `inputs/` (ignored) |
| `.venv/bin/python scripts/rbm_train.py` | `artifacts/rbm/{cd1,cd10}-params.npz`, `artifacts/rbm/run.json` |
| `.venv/bin/python scripts/hopfield_scan.py` | `artifacts/hopfield/scan.npz`, `artifacts/hopfield/run.json` |
| `.venv/bin/python scripts/modern_scan.py` | `artifacts/modern/scan.npz`, `artifacts/modern/run.json` |

The numeric checks are printed by the scripts themselves: the dataset counts,
shapes, pixel range and class fractions by `dataset_figures.py`; the shared
initialization and the error ratio by `rbm_training.py`; the unchanged clamped
half, the speckle measure and the overlap gaps by `rbm_completion.py`; the
measured unstable fractions beside both predictions by `hopfield_unstable.py`;
the four crossings beside the sheet's values by `hopfield_recall.py`; and the
attention difference, the weights and the success fractions by
`modern_weights.py` and `modern_retrieval.py`.

## Script order

| # | command | writes |
| ---: | --- | --- |
| 1 | `bash scripts/download_data.sh` | `inputs/` |
| 2 | `.venv/bin/python scripts/dataset_figures.py mnist` | `evidence/mnist.png` |
| 3 | `.venv/bin/python scripts/dataset_figures.py cifar10` | `evidence/cifar10.png` |
| 4 | `.venv/bin/python scripts/rbm_train.py` | `artifacts/rbm/` |
| 5 | `.venv/bin/python scripts/rbm_spins.py` | `evidence/rbm-spins.png` |
| 6 | `.venv/bin/python scripts/rbm_training.py` | `evidence/rbm-training.png` |
| 7 | `.venv/bin/python scripts/rbm_generation.py` | `evidence/rbm-generation.png` |
| 8 | `.venv/bin/python scripts/rbm_completion.py` | `evidence/rbm-completion.png` |
| 9 | `.venv/bin/python scripts/hopfield_scan.py` | `artifacts/hopfield/` |
| 10 | `.venv/bin/python scripts/hopfield_unstable.py` | `evidence/hopfield-unstable.png` |
| 11 | `.venv/bin/python scripts/hopfield_recall.py` | `evidence/hopfield-recall.png` |
| 12 | `.venv/bin/python scripts/modern_scan.py` | `artifacts/modern/` |
| 13 | `.venv/bin/python scripts/modern_weights.py` | `evidence/modern-weights.png` |
| 14 | `.venv/bin/python scripts/modern_beta.py` | `evidence/modern-beta.png` |
| 15 | `.venv/bin/python scripts/modern_retrieval.py` | `evidence/modern-retrieval.png` |

## `make reproduce` and `make test`

`make reproduce` downloads the datasets, runs the three scans (which include
training both RBMs), and redraws every committed figure. `make test` runs the
pytest suite, which needs `make data` to have run because the dataset tests read
`inputs/`. `make venv` rebuilds the environment.

```bash
make venv
make reproduce      # downloads, all runs and all eleven figures
make test           # 26 tests
```

Wall-clock for the whole of `make reproduce` on this machine: 2 min 18 s,
including about 20 s of downloads, the two largest compute items being the 56 s
CD-10 training and the 40 s Hopfield sweep. The figures are byte-identical
between runs, so no committed evidence changes when the commands are repeated.

## Final verification

**Unknown distribution, empirical distribution, fitted model, sampler.** The
unknown `p_data` is the distribution the images were drawn from; nothing
observes it. A finite dataset defines the empirical distribution, which puts
weight 1/N on each record, and its class frequencies (Part 1) are measurable
statistics of it — unequal for MNIST, exactly 0.1 for CIFAR-10. A fitted model
`p_theta` is a family of probabilities whose parameters are chosen from the data:
Part 2's RBM is such a family, and Part 3's Hopfield couplings are set from the
patterns in one step rather than fitted. A sampling algorithm draws from a
fixed `p_theta` without evaluating it: block Gibbs for the RBM, asynchronous
zero-temperature updates for the Hopfield network. Reconstruction, generation
and completion all run one sampler; they differ only in what is held fixed.
Part 4's queries are more extreme: the model does not describe the data at all,
it stores them, and retrieval is one deterministic map rather than a sample.

**Where CD approximates a likelihood gradient.** The likelihood gradient of
Eq. 8 has two expectations: a data term, cheap, and a model term that needs
samples from `p_theta`. CD replaces the second with the endpoint `v_k` of a
short chain started at each training image, so the update direction becomes
Eq. 9's difference of visible-hidden correlations. Differentiating
`F(v_0) - F(stop_gradient(v_k))` reproduces exactly that direction, because
`stop_gradient` keeps the sampled endpoint fixed, and `grad F(v)` is the
data/model term for one configuration; the same expression is checked against
the explicit correlation form in `tests/test_rbm.py`. The surrogate is not the
NLL: it drops `log Z_theta`, which is intractable here, so a rising surrogate
curve alongside a falling held-out reconstruction error, as in
`rbm-training.png`, is expected rather than contradictory. A low reconstruction
error alone can also mean the chain barely moved, which is why
`rbm-generation.png` runs the same frozen models from random spins.

**How clamping turns generation into inference.** Free generation draws the
whole visible layer from `p(v|h)`. Completion holds the observed bits `v_O` at
their values after every draw, so the chain samples the conditional
`p(v_M | v_O)`: the clamped bits are never resampled and the unclamped ones
approach the conditional law. That is inference with fixed parameters, and it
is why the observed upper half is identical in all 1000 frames of every chain
while the missing half changes. After one step the lower half is speckle
(isolated-ink fraction 0.143 against 0.027 after 1000 steps) because the chain
has not yet used the observation; the true-half and wrong-half overlaps differ
by only 0.019 there and separate later, reaching 0.416 +- 0.023 and
0.510 +- 0.017 for the true halves of CD-1 and CD-10 against 0.225 and 0.253 for
the wrong halves. The gap of 0.191 and 0.257 is the evidence that the
completion uses its input; the wrong-half overlap stays near 0.25 rather than
zero because different digits share pixels near the image centre.
CD-10 completes better than CD-1 although their one-step reconstruction errors
agree to 0.0004, because reconstruction only needs one step while completion
needs a well-mixed conditional.

**Why the unstable-bit fraction follows the crosstalk curve.** From a stored
pattern the local field splits into the pattern's own contribution and
crosstalk from the other `P - 1` patterns, a sum of `(P-1)(N-1)` random signs
divided by `N`. Its variance approaches `alpha`, so the field is Gaussian with
mean `(N-1)/N` and standard deviation `sqrt(alpha)` in the large-N limit, and
the fraction of bits with `xi_i h_i < 0` is `Phi(-1/sqrt(alpha))`. The measured
points follow that curve and approach it as N grows: at N = 800, alpha = 0.20
the measurement is 1.245e-2 against the prediction 1.267e-2, while N = 100 sits
below it (2.500e-4 against 7.827e-4 at alpha = 0.10) where the finite-N form
predicts 4.556e-4 and explains most of the gap. Recall fails because unstable
bits flip, change the fields of the others, and can cascade: the mean overlap
stays above 0.988 to alpha = 0.12 and then drops, and the drop sharpens with N,
crossing 0.9 at 0.1938, 0.1655, 0.1548 and 0.1487 for N = 100, 200, 400 and
800, each within 0.02 of the sheet's values and decreasing toward
`alpha_c = 0.138`.

**The modern update is attention, and beta chooses what it returns.** With the
stored patterns as the columns of `X`, `x <- X softmax(beta X^T x)` is
`softmax(scale * Q K^T) V` with `Q = x^T`, `K = V = X^T` and `scale = beta`, so
the two computations agree to 1.19e-6 over every query, P and beta (the required
1e-5; the library's default scale `1/sqrt(784)` gives about 2, which is why the
scale is passed explicitly). The same `beta` decides the retrieved image: at
beta = 0.003 the queried digit carries 0.0143, close to the uniform 0.01, and
every query returns the same blurred average of the stored digits; at beta =
0.03 the queried digit's weight is 0.2309 and it appears inside that average;
at beta = 0.3 the weight is 1.0000 and one step returns the stored digit. This
is also why modern retrieval survives loads where the classical recall fails:
the classical network's error grows with the correlated-pattern crosstalk, while
the modern energy's log-sum-exp grows only logarithmically in P, so the success
fraction at beta = 0.3 stays between 0.97 and 1.00 up to P = 1000 even though
the classical recall succeeds for no query at any P, not even P = 3.

## What stays out of git

`week6/.gitignore` ignores `/inputs/`, `/artifacts/`, `/.venv/`,
`__pycache__/`, `.pytest_cache/` and `.DS_Store`. None of those paths is
tracked:

```bash
git ls-files week6 | grep -E '/inputs/|/artifacts/|\.venv' || echo "no generated path tracked"
```

The exercise was committed locally and is **not pushed**, per the session
instruction; the sheet's "Push the source, environment and evidence" is
therefore reported as not done.

## Regeneration checklist

From a clean clone: `make venv`; `make data` (which appends nothing to
`README.md`, since the URLs are already there); `make reproduce`, which runs
`scripts/rbm_train.py`, `scripts/hopfield_scan.py` and `scripts/modern_scan.py`
into `artifacts/` and then redraws all eleven figures in the order of the script
table above; `make test`. At that point every tracked file under `week6/`
exists, `git status` shows only the ignored `inputs/`, `artifacts/` and
`.venv/`, and every number quoted in this README can be read line by line from
the printed output of the scripts.

This was verified by cloning the repository to a fresh directory and running
`make venv && make reproduce && make test` there: the tree stayed clean, all
eleven figures came out byte-identical to the committed ones, and all 26 tests
passed.

## Resolutions of ambiguity

* `rbm.toml`'s `wrong_upper` value is one line here; the sheet's rendering wraps
  it across two lines, which a TOML basic string cannot contain.
* The sheet fixes seeds 6601, 6602, 6606 and 6607 but not one for fresh
  generation (Part 2) or for Part 4's queries and recall order. Fresh generation
  uses `fold_in(key(6606), 1)`; Part 4 uses `key(6608)` for the corrupted
  queries and `key(6609)` for the classical recall order. All are recorded in
  `artifacts/`.
* Part 4's success rule is read literally: the signed result must overlap the
  target by at least 0.95 *and* the target must be the nearest stored digit to
  that result.
* The download script appends a URL only when `README.md` does not already
  contain it, so re-running the sheet's commands does not dirty a committed
  file.
