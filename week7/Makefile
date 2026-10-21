PY = .venv/bin/python

.PHONY: venv data reproduce test clean

venv:
	uv sync

data: venv
	scripts/download_inputs.sh

reproduce: data
	$(PY) scripts/run_diffusion.py
	$(PY) scripts/diffusion_evidence.py part1
	$(PY) scripts/diffusion_evidence.py part2
	$(PY) scripts/run_flow.py
	$(PY) scripts/flow_evidence.py
	$(PY) scripts/run_cifar.py
	$(PY) scripts/run_repaint.py

test: venv
	$(PY) -m pytest tests

clean:
	rm -rf artifacts
