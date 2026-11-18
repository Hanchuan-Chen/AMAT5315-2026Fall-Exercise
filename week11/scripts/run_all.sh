#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/pred

if [[ "$(pred --version 2>/dev/null || true)" != "pred 0.6.0" ]]; then
  cargo install problemreductions-cli@0.6.0 --locked
fi

pred path Factoring SpinGlass > runs/pred/path.txt
pred create Factoring --target 15 --m 2 --n 3 -o runs/pred/factor_15.json
pred reduce runs/pred/factor_15.json --to SpinGlass -o runs/pred/factor_15.bundle.json
pred solve runs/pred/factor_15.bundle.json > runs/pred/factor_15.stdout.txt 2>&1
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1 > runs/pred/factor_15.evaluate.txt

for N in 6 7; do
  pred create Factoring --target "$N" --m 2 --n 2 -o "runs/pred/factor_$N.json"
  pred reduce "runs/pred/factor_$N.json" --to SpinGlass -o "runs/pred/factor_$N.bundle.json"
  pred solve "runs/pred/factor_$N.bundle.json" > "runs/pred/factor_$N.stdout.txt" 2>&1
done

cargo install --path . --force
tnsample --mode gadgets --seed 2026 --out runs/gadgets

tnsample --mode count --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
tnsample --mode count --input runs/pred/factor_15.bundle.json --beta 2.0 --seed 2026 --out runs/count_f15
tnsample --mode sample --input runs/pred/factor_15.bundle.json --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15

for k in 2 3 4 5 6 7 8; do
  T=$(( ((1 << k) - 1) * ((1 << k) - 1) ))
  pred create Factoring --target "$T" --m "$k" --n "$k" -o "runs/pred/scale_$k.json"
  pred reduce "runs/pred/scale_$k.json" --to SpinGlass -o "runs/pred/scale_$k.bundle.json"
  for ord in greedy treesa; do
    tnsample --mode plan --order "$ord" --input "runs/pred/scale_$k.bundle.json" --seed 2026 --out "runs/plan_k${k}_$ord"
  done
done

CFG=$(python3 -c "import json; j=json.load(open('runs/sample_f15/run.json')); print(','.join(map(str, next(s['bits'] for s in j['samples'] if s['energy'] == j['e_min']))))")
pred extract runs/pred/factor_15.bundle.json --config "$CFG" > runs/pred/factor_15.extract.txt

for N in 4 5 6 7 8 9 10; do
  tnsample --mode count --n "$N" --seed 2026 --out "runs/count_nq$N"
done
for N in 6 8; do
  tnsample --mode sample --n "$N" --samples 200 --seed 2026 --out "runs/sample_nq$N"
done

.venv/bin/python scripts/plot_evidence.py

