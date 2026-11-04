PY = .venv/bin/python
INFER = cargo +1.96.0 run --release --quiet --

.PHONY: data venv install reproduce evidence test clean

data:
	sh scripts/download_inputs.sh

venv:
	uv venv --allow-existing .venv
	uv pip install --python $(PY) numpy matplotlib pytest

install:
	cargo +1.96.0 build --release --locked

reproduce: data venv install
	$(INFER) --model asia.uai --evidence prior --out artifacts/prior
	$(INFER) --model asia.uai --evidence AD --out artifacts/AD
	$(INFER) --model asia.uai --evidence ADX --out artifacts/ADX
	$(INFER) --model asia.uai --evidence AD --order random --seed 42 --out artifacts/random
	$(INFER) --model asia.uai --evidence AD --out artifacts/greedy
	$(INFER) --model asia.uai --evidence AD --analyze artifacts/random/order.json --out artifacts/analysis
	$(INFER) --model asia.uai --evidence AD --plan artifacts/greedy/order.json --out artifacts/replay
	$(INFER) --model asia.uai --evidence prior --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-prior
	$(INFER) --model asia.uai --evidence AD --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-AD
	$(INFER) --model asia.uai --evidence ADX --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-ADX
	$(INFER) --model promedus/Promedus_14.uai --evidence prior \
	  --evidence-file promedus/Promedus_14.uai.evid --queries 59,102,139 --out artifacts/promedus-greedy
	$(INFER) --model promedus/Promedus_14.uai --evidence prior \
	  --evidence-file promedus/Promedus_14.uai.evid --queries 59,102,139 \
	  --order treesa --seed 2026 --out artifacts/promedus-treesa
	$(INFER) --model promedus/Promedus_14.uai --evidence prior \
	  --evidence-file promedus/Promedus_14.uai.evid --queries 59,102,139 \
	  --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/promedus-bp
	$(PY) scripts/evidence.py

evidence: venv
	$(PY) scripts/evidence.py

test: install venv
	cargo +1.96.0 test --release
	$(PY) -m pytest

clean:
	rm -rf artifacts
