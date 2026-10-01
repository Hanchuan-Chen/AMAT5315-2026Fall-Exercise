#!/usr/bin/env python3
"""Part 4, check (1): audit every Treeverse action file.

Prints, per run, the three counts the sheet asks for: grad steps out of order,
missing or extra; invalid restores; and budget overruns. All must be zero.
"""

import json
import sys
from pathlib import Path

WEEK = Path(__file__).resolve().parents[1]
ART = WEEK / "artifacts"


def audit(directory, steps, delta):
    """Replay availability, time indices and the exact reverse coverage."""
    peak = 1
    calls = 0
    invalid_restores = 0
    budget_overruns = 0
    mismatch = 0
    shots = 0
    expected = list(range(steps - 1, -1, -1))
    for name in sorted(directory.glob("actions-*.json")):
        shots += 1
        available = {0}
        working = None
        reverse = []
        for event in json.loads(name.read_text()):
            action, step = event["action"], event["step"]
            if action == "restore":
                if step not in available:
                    invalid_restores += 1
                working = step
            elif action == "call":
                assert working == step and step < steps
                working = step + 1
                calls += 1
            elif action == "store":
                available.add(step)
                if len(available) > delta + 1:
                    budget_overruns += 1
                peak = max(peak, len(available))
            elif action == "grad":
                reverse.append(step)
            elif action == "fetch":
                if step in available and step != 0:
                    available.remove(step)
                else:
                    invalid_restores += 1
            else:
                raise ValueError(f"unknown action {action}")
            assert event["saved_states"] == len(available), f"{name}: saved_states counter"
        assert available == {0}, f"{name}: leaked checkpoints"
        mismatch += sum(1 for a, b in zip(reverse, expected) if a != b) + abs(len(reverse) - len(expected))
    assert shots > 0, f"{directory}: no action files"
    return dict(
        calls=calls,
        peak=peak,
        grad_mismatch=mismatch,
        invalid_restores=invalid_restores,
        budget_overruns=budget_overruns,
    )


def main():
    totals = dict(grad_mismatch=0, invalid_restores=0, budget_overruns=0)
    for delta in (1, 3, 5, 10):
        folder = ART / f"checkpoint-{delta}"
        steps = json.loads((folder / "result.json").read_text())["steps"]
        stats = audit(folder, steps, delta)
        for key in totals:
            totals[key] += stats[key]
        print(
            f"checkpoint-{delta}: grad out of order/missing/extra={stats['grad_mismatch']}, "
            f"invalid restores={stats['invalid_restores']}, budget overruns={stats['budget_overruns']}, "
            f"replay calls={stats['calls']}, peak={stats['peak']}"
        )
    folder = ART / "marmousi-image"
    steps = json.loads((folder / "result.json").read_text())["steps"]
    stats = audit(folder, steps, 5)
    for key in totals:
        totals[key] += stats[key]
    print(
        f"marmousi-image: grad out of order/missing/extra={stats['grad_mismatch']}, "
        f"invalid restores={stats['invalid_restores']}, budget overruns={stats['budget_overruns']}, "
        f"replay calls={stats['calls']}, peak={stats['peak']}"
    )
    print(
        f"TOTAL audit: grad out of order/missing/extra={totals['grad_mismatch']}, "
        f"invalid restores={totals['invalid_restores']}, budget overruns={totals['budget_overruns']}"
    )
    assert sum(totals.values()) == 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
