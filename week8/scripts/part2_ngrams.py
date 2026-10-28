from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from common import (
    EVIDENCE,
    load_settings,
    load_token_cache,
    lookup_counts,
    ngram_codes,
    ngram_probabilities,
    ngram_statistics,
    write_json,
)


VOCAB = 512


def dense_bigram(
    full_keys: np.ndarray, full_counts: np.ndarray, k: float
) -> np.ndarray:
    table = np.zeros((VOCAB, VOCAB), dtype=np.float64)
    flat_codes = np.arange(VOCAB * VOCAB, dtype=np.uint64)
    values = lookup_counts(full_keys, full_counts, flat_codes)
    table[:] = values.reshape(VOCAB, VOCAB)
    table += k
    table /= table.sum(axis=1, keepdims=True)
    return table


def explicit_normalization_errors(p: np.ndarray, eos: int = 2) -> dict[str, float]:
    """Enumerate all sequences of length 1, 2 and 3 by matrix products."""
    initial = p[eos]
    ones = np.ones(VOCAB, dtype=np.float64)
    length1 = float(initial @ ones)
    length2 = float(initial @ p @ ones)
    length3 = float(initial @ p @ p @ ones)
    return {
        "length_1": abs(length1 - 1.0),
        "length_2": abs(length2 - 1.0),
        "length_3": abs(length3 - 1.0),
    }


def sample_bigram(
    p: np.ndarray, contexts: np.ndarray, n_tokens: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    context_lookup = {int(token): i for i, token in enumerate(contexts)}
    observed = np.zeros((len(contexts), VOCAB), dtype=np.int64)
    tokens = np.empty(n_tokens + 1, dtype=np.int32)
    tokens[0] = int(rng.choice(contexts))
    cdf = np.cumsum(p, axis=1)
    for t in range(n_tokens):
        previous = int(tokens[t])
        u = rng.random()
        tokens[t + 1] = int(np.searchsorted(cdf[previous], u, side="right"))
        row = context_lookup.get(previous)
        if row is not None:
            observed[row, tokens[t + 1]] += 1
    return observed, tokens


def per_context_chi_square(
    observed: np.ndarray, probabilities: np.ndarray
) -> tuple[float, int, float]:
    """Combine separate context tests by adding chi-square values and degrees."""
    statistic = 0.0
    degrees = 0
    for row, probability in zip(observed, probabilities):
        expected = probability * row.sum()
        keep = expected >= 5.0
        if np.any(~keep):
            observed_bins = np.append(row[keep], row[~keep].sum())
            expected_bins = np.append(expected[keep], expected[~keep].sum())
        else:
            observed_bins = row
            expected_bins = expected
        statistic += float(np.sum((observed_bins - expected_bins) ** 2 / expected_bins))
        degrees += len(expected_bins) - 1
    return statistic, degrees, float(chi2.sf(statistic, degrees))


def wrong_context_chi_square(
    observed: np.ndarray, full_p: np.ndarray, contexts: np.ndarray, seed: int
) -> tuple[float, int, float]:
    rng = np.random.default_rng(seed + 1)
    wrong = np.zeros_like(observed)
    cdf = np.cumsum(full_p, axis=1)
    context_lookup = {int(token): i for i, token in enumerate(contexts)}
    sequence = np.empty(int(observed.sum()) + 2, dtype=np.int32)
    sequence[0] = int(rng.choice(contexts))
    sequence[1] = int(rng.choice(contexts))
    for t in range(1, int(observed.sum())):
        lag_two_context = int(sequence[t - 1])
        sequence[t + 1] = int(
            np.searchsorted(cdf[lag_two_context], rng.random(), side="right")
        )
        row = context_lookup.get(int(sequence[t]))
        if row is not None:
            wrong[row, sequence[t + 1]] += 1
    return per_context_chi_square(wrong, full_p[contexts])


def main() -> None:
    settings = load_settings()
    p2 = settings["part2"]
    cache = load_token_cache()
    tokens = cache["train_tokens"]
    valid = cache["valid_tokens"]
    vocab = cache["vocab_size"]
    assert vocab == VOCAB
    train_chars = cache["train_chars"]
    valid_chars = cache["valid_chars"]
    n_predictions = len(valid) - 1

    uniform_nats = math.log(VOCAB)
    uniform_bpc = uniform_nats / math.log(2) * n_predictions / valid_chars
    models: dict[str, dict[str, float]] = {
        "uniform": {
            "nats": uniform_nats,
            "perplexity": float(VOCAB),
            "bits_per_char": uniform_bpc,
            "k": None,
            "unseen_fraction": 0.0,
        }
    }

    bigram_dense: np.ndarray | None = None
    bigram_contexts: np.ndarray | None = None
    for n in range(1, int(p2["max_n"]) + 1):
        print(f"[Part 2] counting n={n} ...", flush=True)
        prediction_count = len(tokens) - 1
        probe_start = int(0.9 * prediction_count)
        probe = tokens[max(0, probe_start - n + 1) :]
        inner_keys, inner_counts, inner_context_keys, inner_context_counts = (
            ngram_statistics(tokens[:probe_start], n)
        )
        best_k = None
        best_loss = float("inf")
        for k in p2["candidate_k"]:
            probabilities = ngram_probabilities(
                probe,
                n,
                inner_keys,
                inner_counts,
                inner_context_keys,
                inner_context_counts,
                float(k),
                VOCAB,
            )
            loss = float(-np.log(probabilities).mean())
            if loss < best_loss:
                best_loss = loss
                best_k = float(k)
        assert best_k is not None

        del inner_keys, inner_counts, inner_context_keys, inner_context_counts
        full_keys, full_counts, context_keys, context_counts = ngram_statistics(tokens, n)

        probabilities = ngram_probabilities(
            valid,
            n,
            full_keys,
            full_counts,
            context_keys,
            context_counts,
            best_k,
            VOCAB,
        )
        total_nats = float(-np.log(probabilities).sum())
        nats = total_nats / len(probabilities)
        model = {
            "nats": nats,
            "perplexity": float(math.exp(nats)),
            "bits_per_char": total_nats / math.log(2) / valid_chars,
            "k": best_k,
            "probe_nats": best_loss,
            "training_unique_ngrams": int(len(full_keys)),
        }
        if n == 2:
            bigram_dense = dense_bigram(full_keys, full_counts, best_k)
            rows = tokens[:-1]
            values, counts = np.unique(rows, return_counts=True)
            order = np.argsort(counts)[::-1]
            bigram_contexts = values[order[: int(p2["chi2_contexts"])]]
        heldout_full = (
            np.asarray(valid[1:], dtype=np.uint64)
            if n == 1
            else ngram_codes(valid, n)
        )
        heldout_seen = lookup_counts(full_keys, full_counts, heldout_full) > 0
        model["unseen_fraction"] = float(1.0 - heldout_seen.mean())
        models[f"{n}gram"] = model
        print(
            f"  n={n}: k={best_k:g}, nats={nats:.6f}, "
            f"ppl={model['perplexity']:.4f}, bpc={model['bits_per_char']:.6f}",
            flush=True,
        )

    assert bigram_dense is not None and bigram_contexts is not None
    norm_errors = explicit_normalization_errors(bigram_dense)
    observed, _ = sample_bigram(
        bigram_dense,
        bigram_contexts,
        int(p2["bigram_samples"]),
        int(p2["seed"]),
    )
    chi_stat, chi_degrees, chi_p = per_context_chi_square(
        observed, bigram_dense[bigram_contexts]
    )
    wrong_stat, wrong_degrees, wrong_p = wrong_context_chi_square(
        observed, bigram_dense, bigram_contexts, int(p2["seed"])
    )

    ns = np.arange(1, 7)
    bpcs = np.asarray([models[f"{n}gram"]["bits_per_char"] for n in ns])
    unseen = np.asarray([models[f"{n}gram"]["unseen_fraction"] for n in ns], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(
        np.r_[0, ns],
        np.r_[models["uniform"]["bits_per_char"], bpcs],
        "o-",
        color="#205493",
    )
    axes[0].set_xlabel("n-gram order n (0 = uniform)")
    axes[0].set_ylabel("held-out bits per character")
    axes[0].set_title("Held-out loss")
    axes[0].grid(alpha=0.25)
    axes[0].annotate(
        "4-gram minimum",
        xy=(4, bpcs[3]),
        xytext=(4.25, bpcs[3] + 0.18),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    axes[1].plot(ns, 100 * unseen, "o-", color="#b24c3d")
    axes[1].set_xlabel("n-gram order n")
    axes[1].set_ylabel("held-out n-grams unseen (%)")
    axes[1].set_title("Curse of dimensionality")
    axes[1].grid(alpha=0.25)
    fig.suptitle("TinyStories BPE n-gram ladder")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part2-ngram-ladder.png", dpi=180)
    plt.close(fig)

    ref_nats = {"uniform": 6.238, "1gram": 4.839, "2gram": 3.038, "3gram": 2.094}
    ref_ppl = {"uniform": 512.0, "1gram": 126.4, "2gram": 20.9, "3gram": 8.12}
    ref_unseen = {"3gram": 0.006, "4gram": 0.042, "5gram": 0.122, "6gram": 0.235}
    checks = {
        "split_counts": {
            "pass": cache["train_stories"] == 24867 and cache["valid_stories"] == 2763,
            "actual": [cache["train_stories"], cache["valid_stories"]],
        },
        "chars_per_token": {
            "pass": abs(valid_chars / n_predictions - 2.052) < 0.02,
            "actual": valid_chars / n_predictions,
        },
        "decode_mismatches": {
            "pass": cache["decode_mismatches"] == 3,
            "actual": cache["decode_mismatches"],
        },
        "normalization": {
            "pass": max(norm_errors.values()) <= 1e-12,
            "actual": norm_errors,
            "target": 1e-12,
        },
        "nats_within_0.03": {
            key: {
                "pass": abs(models[key]["nats"] - value) <= 0.03,
                "actual": models[key]["nats"],
                "reference": value,
            }
            for key, value in ref_nats.items()
        },
        "perplexity_within_3pct": {
            key: {
                "pass": abs(models[key]["perplexity"] / value - 1.0) <= 0.03,
                "actual": models[key]["perplexity"],
                "reference": value,
            }
            for key, value in ref_ppl.items()
        },
        "best_order": {
            "pass": int(np.argmin(bpcs) + 1) == 4,
            "actual": int(np.argmin(bpcs) + 1),
        },
        "five_and_six_bpc": {
            key: {
                "pass": abs(models[key]["bits_per_char"] - target) <= 0.03,
                "actual": models[key]["bits_per_char"],
                "reference": target,
            }
            for key, target in (("5gram", 1.46), ("6gram", 1.79))
        },
        "four_gram_bpc": {
            "pass": abs(models["4gram"]["bits_per_char"] - 1.310) <= 0.03,
            "actual": models["4gram"]["bits_per_char"],
            "reference": 1.310,
        },
        "unseen_within_2pct": {
            key: {
                "pass": abs(value - target) <= 0.02,
                "actual": value,
                "reference": target,
            }
            for key, target in ref_unseen.items()
            for value in [models[key]["unseen_fraction"]]
        },
        "bigram_sample_chi2": {
            "pass": chi_p > 0.001,
            "p_value": chi_p,
            "statistic": chi_stat,
            "degrees_of_freedom": chi_degrees,
            "target": "> 0.001",
        },
        "wrong_context_negative_control": {
            "pass": wrong_p < 0.001,
            "p_value": wrong_p,
            "statistic": wrong_stat,
            "degrees_of_freedom": wrong_degrees,
        },
        "uniform_exact": {
            "pass": abs(uniform_nats - math.log(512)) < 1e-15,
            "actual": uniform_nats,
        },
    }

    def flatten(node: object) -> list[bool]:
        if isinstance(node, dict):
            if "pass" in node and isinstance(node["pass"], bool):
                return [node["pass"]]
            return [item for value in node.values() for item in flatten(value)]
        return []

    all_pass = all(flatten(checks))
    result = {
        "corpus": {
            "stories": cache["train_stories"] + cache["valid_stories"],
            "train_stories": cache["train_stories"],
            "valid_stories": cache["valid_stories"],
            "train_tokens": len(tokens),
            "valid_target_tokens": n_predictions,
            "train_characters": train_chars,
            "valid_characters": valid_chars,
            "characters_per_valid_token": valid_chars / n_predictions,
            "decode_mismatches": cache["decode_mismatches"],
        },
        "models": models,
        "sampling": {
            "top_context_tokens": bigram_contexts.tolist(),
            "samples": int(p2["bigram_samples"]),
            "chi2_statistic": chi_stat,
            "chi2_degrees_of_freedom": chi_degrees,
            "chi2_p_value": chi_p,
            "wrong_context_chi2_statistic": wrong_stat,
            "wrong_context_chi2_degrees_of_freedom": wrong_degrees,
            "wrong_context_p_value": wrong_p,
        },
        "checks": checks,
        "all_checks_pass": all_pass,
    }
    write_json("part2-metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not all_pass:
        raise SystemExit("Part 2 acceptance check failed")


if __name__ == "__main__":
    main()

