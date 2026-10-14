# `make reproduce` rebuilds every generated path this week commits from a
# clean clone: it fetches the datasets into the git-ignored inputs/, trains
# both RBMs into artifacts/, runs the Hopfield and modern-Hopfield scans, and
# redraws every figure in evidence/.  `make data` only fetches them, and does
# nothing while inputs/ is present, so `make test` never re-downloads.

PY = .venv/bin/python
MNIST_IMAGES = inputs/mnist/train-images-idx3-ubyte.gz

.PHONY: data venv reproduce test clean

data: $(MNIST_IMAGES)

$(MNIST_IMAGES):
	bash scripts/download_data.sh

venv:
	uv venv .venv
	uv pip install --python $(PY) numpy matplotlib jax pytest

reproduce: data
	$(PY) scripts/dataset_figures.py mnist
	$(PY) scripts/dataset_figures.py cifar10
	$(PY) scripts/rbm_train.py
	$(PY) scripts/rbm_spins.py
	$(PY) scripts/rbm_training.py
	$(PY) scripts/rbm_generation.py
	$(PY) scripts/rbm_completion.py
	$(PY) scripts/hopfield_scan.py
	$(PY) scripts/hopfield_unstable.py
	$(PY) scripts/hopfield_recall.py
	$(PY) scripts/modern_scan.py
	$(PY) scripts/modern_weights.py
	$(PY) scripts/modern_beta.py
	$(PY) scripts/modern_retrieval.py

test: data
	$(PY) -m pytest tests

clean:
	rm -rf artifacts
