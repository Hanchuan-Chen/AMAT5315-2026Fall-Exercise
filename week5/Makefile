# `make reproduce` rebuilds every generated artifact: it installs `seismic`
# (Enzyme kernel included), runs the four modes, then redraws every figure.
# The published gate reads `artifacts/`, so run it after this:
#   python3 week5/checker/check <repo>
# Every step puts cargo's bin directory on PATH first, so the recipe works even
# when the caller's PATH does not carry it.

CARGO_BIN = $${CARGO_HOME:-$$HOME/.cargo}/bin
PY = .venv/bin/python

.PHONY: install venv reproduce test clean

install:
	PATH="$(CARGO_BIN):$$PATH"; export PATH; cargo install --path seismic --quiet

venv:
	uv venv .venv
	uv pip install --python $(PY) numpy matplotlib jax pytest

reproduce: install
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	seismic --experiment inputs/reflector.json --mode born --out artifacts/born
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy \
	  --every 3 --out artifacts/adjoint
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	for b in 1 3 5 10; do \
	  seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy \
	    --storage treeverse --checkpoints $$b --out artifacts/checkpoint-$$b; \
	done
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	seismic --experiment inputs/marmousi.json --mode adjoint \
	  --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 \
	  --out artifacts/marmousi-image
	$(PY) scripts/ad_modes.py
	$(PY) scripts/ad_graph.py
	$(PY) scripts/ad_scaling.py
	$(PY) scripts/inputs_figure.py
	$(PY) scripts/forward_figures.py
	$(PY) scripts/adjoint_figures.py
	$(PY) scripts/checkpoint_figures.py
	$(PY) scripts/marmousi_figure.py
	$(PY) scripts/audit.py

test:
	cd seismic && cargo test --release
	$(PY) -m pytest tests

clean:
	rm -rf artifacts seismic/target
