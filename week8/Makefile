PY = .venv/bin/python

.PHONY: venv data reproduce test clean

venv:
	uv sync

data: venv
	scripts/download_inputs.sh

reproduce: data
	$(PY) scripts/part1_embeddings.py
	$(PY) scripts/part2_ngrams.py
	$(PY) scripts/part3_memory.py
	$(PY) scripts/part4_attention.py
	$(PY) scripts/part5_transformer.py --force
	$(PY) scripts/part6_scaling.py

test: venv
	$(PY) -m pytest

clean:
	rm -rf artifacts

