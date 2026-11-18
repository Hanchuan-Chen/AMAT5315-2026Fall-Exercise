# Week 11: hard problems, reductions, and tensor-network sampling

This directory implements only the Week 11 tasks: the `pred` reduction round trip, the seven gate gadgets and the AND-NOT-SET1 circuit, spin-glass plan/count/sample operations, and sparse N-queens count/sample operations. The three evidence figures are generated from saved runs by `scripts/plot_evidence.py`.

## Environment

The recorded environment is macOS arm64 with Rust/Cargo 1.98.1, Python 3.12.14, `pred` 0.6.0, `omeco` 0.2.6, NumPy 2.5.5, and Matplotlib 3.11.2. CMake is required by `pred` at installation time. Python dependencies are in `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
pred --version
```

## Reproduction

Run every required command from this directory:

```bash
bash scripts/run_all.sh
```

The script installs the Rust binary, runs the exact sheet commands, saves raw `pred` output and all run JSON, extracts the factor from the lowest-energy N=15 sample, and finally runs the evidence checks and plots. The individual required commands are:

```bash
cargo install problemreductions-cli@0.6.0 --locked
mkdir -p runs/pred
pred path Factoring SpinGlass
pred create Factoring --target 15 --m 2 --n 3 -o runs/pred/factor_15.json
pred reduce runs/pred/factor_15.json --to SpinGlass \
  -o runs/pred/factor_15.bundle.json
pred solve runs/pred/factor_15.bundle.json
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1
for N in 6 7; do
  pred create Factoring --target $N --m 2 --n 2 -o runs/pred/factor_$N.json
  pred reduce runs/pred/factor_$N.json --to SpinGlass \
    -o runs/pred/factor_$N.bundle.json
  pred solve runs/pred/factor_$N.bundle.json
done

cargo install --path . --force
tnsample --mode gadgets --seed 2026 --out runs/gadgets

tnsample --mode count --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
tnsample --mode count --input runs/pred/factor_15.bundle.json \
  --beta 2.0 --seed 2026 --out runs/count_f15
tnsample --mode sample --input runs/pred/factor_15.bundle.json \
  --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15
for k in 2 3 4 5 6 7 8; do
  T=$(( ((1 << k) - 1) * ((1 << k) - 1) ))
  pred create Factoring --target $T --m $k --n $k -o runs/pred/scale_$k.json
  pred reduce runs/pred/scale_$k.json --to SpinGlass \
    -o runs/pred/scale_$k.bundle.json
  for ord in greedy treesa; do
    tnsample --mode plan --order $ord --input runs/pred/scale_$k.bundle.json \
      --seed 2026 --out runs/plan_k${k}_$ord
  done
done
for N in 4 5 6 7 8 9 10; do
  tnsample --mode count --n $N --seed 2026 --out runs/count_nq$N
done
for N in 6 8; do
  tnsample --mode sample --n $N --samples 200 --seed 2026 \
    --out runs/sample_nq$N
done
```

Every run uses seed 2026. This seed controls lattice disorder and all conditional draws. `omeco` 0.2.6 uses its own internal random search for TreeSA, so plan metrics can vary slightly when the planner is rerun; the saved plan evidence and figures below describe the recorded run. Spin-glass counts and samples use `beta=2`; the 10 by 10 check uses `p-af=0`; N=15 uses 2000 samples; both N-queens sampling runs use 200 samples.

## Results

The reduction round trip is `Factoring -> CircuitSAT -> SpinGlass`. For 15, `pred solve` returns `[1,1,1,0,1]` with `Min(0)`, decoding to 3 x 5. For 6 it returns `[0,1,1,1]` with `Min(0)`, decoding to 2 x 3. For 7 the best target state maps back to `[1,0,1,1]` with `Min(4)`, exposing the absence of a two-bit factor pair.

The gadget enumeration has floors `-3,-3,-1,-1,-1,-1,-4` for AND, OR, NOT, COPY, SET0, SET1, XOR. The joined AND-NOT-SET1 circuit has exactly three states at its floor `-5`, for inputs 00, 01, and 10.

The 10 by 10 ferromagnet gives `E_min=-180`, `g0=2`. The 104-spin factor-15 bundle gives `E_min=-299`, `g0=1`, `log Z=598.317945217003`, and `sc=5`. The 2000 draws at beta 2 give 1436 ground states versus the exact prediction 1455.285. `pred extract` maps the sampled ground state to `[1,1,1,0,1]` and reports `Min(0)`.

For k=2 through 8, treesa space complexities are `4,7,9,11,13,15,17` and time complexities are `10.62,12.53,14.46,16.57,18.98,21.52,24.04`. Greedy complexities are recorded separately in each `runs/plan_k*_greedy/run.json`; greedy tie-breaking may vary at random between runs, so its profile can differ from the reference run.

The contracted N-queens counts for N=4 through 10 are `2,10,4,40,92,352,724`, matching OEIS A000170. All 200 sampled boards for N=6 and N=8 satisfy the queen rules. The N=6 sample contains all 4 boards with frequencies `52,51,50,47`; the N=8 sample contains 79 distinct boards.

## Evidence

`runs/` contains the raw JSON for every gadget, count, sample, and plan run, plus all `pred` instances, bundles, and extracted solutions. `artifacts/` contains only the three required figures:

- `artifacts/gates.png`: gate-row energies and the 16 joined-circuit states.
- `artifacts/factoring.png`: greedy/treesa contraction memory and the N=15 beta-2 energy histogram.
- `artifacts/nqueens.png`: exact counts against OEIS and the 200 N=6 sample frequencies.

Both generated directories are excluded by `.gitignore`.
