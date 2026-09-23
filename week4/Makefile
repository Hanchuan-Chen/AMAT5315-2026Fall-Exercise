# `make reproduce` writes the raw artifacts the published gate reads
# (week4-resources.zip: `week4/checker/check <repo>`), and nothing else: the
# tracked evidence/ figures are redrawn separately by week4/scripts/.
# SEED picks the random field, the same variable the gate reads.
# Every step puts cargo's bin directory on PATH first, so `make reproduce`
# works even when the caller's PATH does not carry it.

SEED ?= 2026
CARGO_BIN = $${CARGO_HOME:-$$HOME/.cargo}/bin

.PHONY: reproduce

reproduce:
	PATH="$(CARGO_BIN):$$PATH"; export PATH; cargo install --path . --quiet
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	mkdir -p artifacts/taylor-green artifacts/random artifacts/unstable/taylor-green
	PATH="$(CARGO_BIN):$$PATH"; export PATH; field taylor-green --n 64 \
	  | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
	      --out artifacts/taylor-green > artifacts/taylor-green.tsv
	PATH="$(CARGO_BIN):$$PATH"; export PATH; field random --n 128 --seed $(SEED) \
	  --k-min 2 --k-max 6 \
	  | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
	      --out artifacts/random > artifacts/random.tsv
	rm -rf artifacts/order
	mkdir -p artifacts/order
	PATH="$(CARGO_BIN):$$PATH"; export PATH; \
	for dt in 0.4 0.25 0.2; do field taylor-green --n 8 \
	  | fluid --method rk4 --nu 0.5 --dt $$dt --t-end 2 --every 2 \
	      --out artifacts/order/rk4-dt$$dt > artifacts/order/rk4-dt$$dt.tsv; done
	PATH="$(CARGO_BIN):$$PATH"; export PATH; field taylor-green --n 64 \
	  | fluid --method rk4 --nu 0.1 --dt 0.04 --t-end 4 --every 0.1 \
	      --out artifacts/unstable/taylor-green > artifacts/unstable/taylor-green.tsv \
	  || [ $$? -eq 1 ]
