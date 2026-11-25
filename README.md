# AMAT5315-2026Fall-Exercise

Weekly exercises for AMAT5315 in Fall 2026 at HKUST(GZ). The Week 1 exercise uses an AI coding agent and Git to develop, test, and publish a Monte Carlo estimator for pi.

## Requirements

- Python 3.10 or newer
- pytest

Install pytest with:

```bash
python3 -m pip install pytest
```

## Run the test

From the repository root, run:

```bash
python3 -m pytest week1/
```

## Test evidence

![Terminal showing the Week 1 pytest command completing with one passing test](assets/pytest-passed.png)

The screenshot shows `python3 -m pytest week1/` completing successfully with `1 passed`.

## Estimate and error

From the repository root, run:

```bash
python3 -c 'from week1.pi import estimate_pi; import math; v=estimate_pi(1000000, seed=2026); print(f"pi estimate = {v:.12f}"); print(f"absolute error = {abs(v-math.pi):.12f}")'
```

![Terminal showing the seeded pi estimate and its absolute error](week1/result.png)
