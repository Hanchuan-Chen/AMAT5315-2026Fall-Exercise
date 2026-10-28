from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import sentencepiece as spm
from scipy.stats import chisquare

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    DATA,
    EVIDENCE,
    load_settings,
    load_token_cache,
    lookup_counts,
    ngram_statistics,
    path_from,
    write_json,
)
from transformer_lib import (
    VOCAB,
    cross_entropy,
    init_transformer,
    load_checkpoint,
    sampler_probabilities,
    save_checkpoint,
    transformer_logits,
)

jax.config.update("jax_enable_x64", False)


def make_batch(tokens: np.ndarray, rng: np.random.Generator, batch: int, context: int) -> tuple:
    starts = rng.integers(0, len(tokens) - context - 1, size=batch)
    inputs = np.stack([tokens[start : start + context] for start in starts])
    targets = np.stack([tokens[start + 1 : start + context + 1] for start in starts])
    return jnp.asarray(inputs, dtype=jnp.int32), jnp.asarray(targets, dtype=jnp.int32)


def train(settings: dict, cache: dict, force: bool) -> tuple[dict, dict]:
    p5 = settings["part5"]
    checkpoint_path = path_from(settings["paths"]["transformer_checkpoint"])
    if checkpoint_path.exists() and not force:
        print("[Part 5] loading existing checkpoint", flush=True)
        return load_checkpoint(checkpoint_path)

    layers = int(p5["n_layers"])
    width = int(p5["width"])
    heads = int(p5["heads"])
    context = int(p5["context"])
    params = init_transformer(
        jax.random.PRNGKey(int(p5["seed"])),
        layers=layers,
        width=width,
        heads=heads,
        context=context,
    )
    moments_m = jax.tree.map(jnp.zeros_like, params)
    moments_v = jax.tree.map(jnp.zeros_like, params)
    steps = int(p5["steps"])
    batch_size = int(p5["batch_size"])
    peak_lr = float(p5["learning_rate"])
    warmup = int(p5["warmup_steps"])
    final_fraction = float(p5["final_lr_fraction"])
    weight_decay = float(p5["weight_decay"])
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def loss_fn(current: dict, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        return cross_entropy(current, inputs, targets)

    @jax.jit
    def step(
        current: dict,
        m: dict,
        v: dict,
        inputs: jax.Array,
        targets: jax.Array,
        learning_rate: jax.Array,
        iteration: jax.Array,
    ) -> tuple:
        loss, gradient = jax.value_and_grad(loss_fn, argnums=0)(current, inputs, targets)
        leaves = jax.tree.leaves(gradient)
        norm = jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))
        scale = jnp.minimum(1.0, 1.0 / (norm + 1e-12))
        gradient = jax.tree.map(lambda leaf: leaf * scale, gradient)
        m = jax.tree.map(lambda old, g: beta1 * old + (1 - beta1) * g, m, gradient)
        v = jax.tree.map(lambda old, g: beta2 * old + (1 - beta2) * g * g, v, gradient)
        correction1 = 1 - beta1 ** (iteration + 1)
        correction2 = 1 - beta2 ** (iteration + 1)
        updated = jax.tree.map(
            lambda parameter, m_i, v_i: parameter * (1 - learning_rate * weight_decay)
            - learning_rate * (m_i / correction1) / (jnp.sqrt(v_i / correction2) + eps),
            current,
            m,
            v,
        )
        return updated, m, v, loss

    rng = np.random.default_rng(int(p5["seed"]))
    trajectory: list[dict[str, float]] = []
    started = time.time()
    for iteration in range(steps):
        if iteration < warmup:
            learning_rate = peak_lr * (iteration + 1) / warmup
        else:
            progress = (iteration - warmup) / max(1, steps - warmup)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            learning_rate = peak_lr * (final_fraction + (1 - final_fraction) * cosine)
        inputs, targets = make_batch(cache["train_tokens"], rng, batch_size, context)
        params, moments_m, moments_v, loss = step(
            params,
            moments_m,
            moments_v,
            inputs,
            targets,
            jnp.asarray(learning_rate, dtype=jnp.float32),
            jnp.asarray(iteration, dtype=jnp.int32),
        )
        if iteration % 100 == 0 or iteration + 1 == steps:
            value = float(loss)
            trajectory.append(
                {"step": iteration + 1, "loss": value, "learning_rate": learning_rate}
            )
            print(
                f"[Part 5] transformer step {iteration + 1}/{steps}: "
                f"loss {value:.5f}, lr {learning_rate:.6f}",
                flush=True,
            )
    elapsed = time.time() - started
    metadata = {"elapsed_seconds": elapsed}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_path, params, metadata)
    write_json("part5-training-trajectory.json", trajectory)
    return params, metadata


def evaluate(
    params: dict, tokens: np.ndarray, valid_chars: int, batch_size: int = 256
) -> dict:
    context = 64
    maximum_start = len(tokens) - context - 1
    starts = list(range(0, maximum_start + 1, context))
    covered_until = starts[-1] + context
    if covered_until < len(tokens) - 1:
        starts.append(maximum_start)
    apply = jax.jit(lambda current, inputs: transformer_logits(current, inputs))
    total_nats = 0.0
    total_positions = 0
    position_sum = np.zeros(context, dtype=np.float64)
    position_square_sum = np.zeros(context, dtype=np.float64)
    position_count = np.zeros(context, dtype=np.int64)
    next_target = 1
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset : offset + batch_size]
        inputs = np.stack([tokens[start : start + context] for start in selected])
        targets = np.stack([tokens[start + 1 : start + context + 1] for start in selected])
        logits = np.asarray(apply(params, jnp.asarray(inputs, dtype=jnp.int32)))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        losses = -np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        for row, start in enumerate(selected):
            first_column = max(0, next_target - start - 1)
            last_column = min(context, len(tokens) - start - 1)
            selected_losses = losses[row, first_column:last_column]
            total_nats += float(selected_losses.sum())
            total_positions += len(selected_losses)
            position_indices = np.arange(first_column, last_column)
            position_sum[position_indices] += selected_losses
            position_square_sum[position_indices] += selected_losses**2
            position_count[position_indices] += 1
            next_target = max(next_target, start + last_column + 1)
    mean_loss = position_sum / position_count
    variance = np.maximum(position_square_sum / position_count - mean_loss**2, 0.0)
    standard_error = np.sqrt(variance / position_count)
    nats = total_nats / total_positions
    return {
        "nats": nats,
        "bits_per_char": total_nats / math.log(2) / valid_chars,
        "target_tokens": total_positions,
        "position_nats": mean_loss.tolist(),
        "position_standard_error": standard_error.tolist(),
        "coverage": {
            "validation_targets": int(len(tokens) - 1),
            "scored_targets": total_positions,
            "complete": total_positions == len(tokens) - 1,
        },
    }


def ngram_model_tables(tokens: np.ndarray, max_n: int = 4) -> dict[int, dict]:
    part2_path = path_from("artifacts/part2-metrics.json")
    with part2_path.open(encoding="utf-8") as handle:
        part2 = json.load(handle)
    tables: dict[int, dict] = {}
    for n in range(1, max_n + 1):
        full_keys, full_counts, context_keys, context_counts = ngram_statistics(tokens, n)
        tables[n] = {
            "full_keys": full_keys,
            "full_counts": full_counts,
            "context_keys": context_keys,
            "context_counts": context_counts,
            "k": float(part2["models"][f"{n}gram"]["k"]),
        }
    return tables


def sample_ngram(
    table: dict,
    n: int,
    length: int,
    seed: int,
    eos: int = 2,
    initial_history: np.ndarray | None = None,
) -> list[int]:
    rng = np.random.default_rng(seed)
    history = (
        np.asarray(initial_history, dtype=np.uint64)
        if initial_history is not None
        else np.asarray([eos] * max(1, n - 1), dtype=np.uint64)
    )
    generated: list[int] = []
    all_tokens = np.arange(VOCAB, dtype=np.uint64)
    while len(generated) < length:
        if n == 1:
            flat_codes = all_tokens
            context_code = np.asarray([0], dtype=np.uint64)
        else:
            packed = np.uint64(0)
            for value in history[-n + 1 :]:
                packed = packed * np.uint64(VOCAB) + value
            context_code = np.asarray([packed], dtype=np.uint64)
            flat_codes = context_code[0] * np.uint64(VOCAB) + all_tokens
        full_counts = lookup_counts(table["full_keys"], table["full_counts"], flat_codes)
        context_count = lookup_counts(
            table["context_keys"], table["context_counts"], context_code
        )
        probabilities = (full_counts.astype(float) + table["k"]) / (
            float(context_count[0]) + table["k"] * VOCAB
        )
        probabilities /= probabilities.sum()
        token = int(rng.choice(VOCAB, p=probabilities))
        history = np.concatenate([history, np.asarray([token], dtype=np.uint64)])
        if token != eos:
            generated.append(token)
    return generated


def decode_ids(sp, ids: list[int]) -> str:
    return sp.decode([int(token) for token in ids]).replace("\n", " / ").strip()


def generate_transformer(
    params: dict,
    sp,
    seed: int,
    mode: str,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    length: int = 60,
    eos: int = 2,
) -> str:
    apply = jax.jit(lambda current, inputs: transformer_logits(current, inputs))
    key = jax.random.PRNGKey(seed)
    context = np.full((1, 64), eos, dtype=np.int32)
    generated: list[int] = []
    while len(generated) < length:
        logits = np.asarray(apply(params, jnp.asarray(context)))[0, -1]
        key, subkey = jax.random.split(key)
        probabilities = sampler_probabilities(
            jnp.asarray(logits), mode, temperature, top_k, top_p, min_p
        )
        token = int(jax.random.categorical(subkey, jnp.log(probabilities)))
        context = np.roll(context, -1, axis=1)
        context[0, -1] = token
        if token != eos:
            generated.append(token)
    return decode_ids(sp, generated)


def grouped_chi_square(observed: np.ndarray, expected_probabilities: np.ndarray) -> tuple[float, float]:
    expected = expected_probabilities * observed.sum()
    expected *= observed.sum() / expected.sum()
    keep = expected >= 5.0
    observed_bins = observed[keep]
    expected_bins = expected[keep]
    dropped_expected = expected[~keep].sum()
    if dropped_expected > 0:
        observed_bins = np.concatenate([observed_bins, [observed[~keep].sum()]])
        expected_bins = np.concatenate([expected_bins, [dropped_expected]])
    result = chisquare(observed_bins, expected_bins)
    return float(result.statistic), float(result.pvalue)


def sampler_checks(params: dict, valid: np.ndarray) -> dict:
    apply = jax.jit(lambda current, inputs: transformer_logits(current, inputs))
    context = jnp.asarray(valid[:64][None, :], dtype=jnp.int32)
    all_logits = np.asarray(apply(params, context))[0]
    shifted = all_logits - all_logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-30)).sum(axis=-1)
    position = int(np.argmax(entropy))
    logits = jnp.asarray(all_logits[position])
    configurations = [
        ("temperature", 0.8, None, None, None),
        ("top_k", 0.8, 20, None, None),
        ("top_p", 0.8, None, 0.9, None),
        ("min_p", 0.7, None, None, 0.1),
    ]
    sample_count = 100000
    results = {}
    base_key = jax.random.PRNGKey(2027 + position)
    for index, (mode, temperature, top_k, top_p, min_p) in enumerate(configurations):
        exact = sampler_probabilities(logits, mode, temperature, top_k, top_p, min_p)
        draws = np.asarray(
            jax.random.categorical(
                jax.random.fold_in(base_key, index),
                jnp.log(exact),
                shape=(sample_count,),
            )
        )
        counts = np.bincount(draws, minlength=VOCAB)
        statistic, p_value = grouped_chi_square(counts, np.asarray(exact))
        results[mode] = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "chi2_statistic": statistic,
            "chi2_p_value": p_value,
            "draws": sample_count,
            "pass": p_value > 0.001,
        }
        if mode == "top_p":
            sorted_probabilities = np.sort(np.asarray(exact))[::-1]
            positive = sorted_probabilities[sorted_probabilities > 0]
            wrong_sorted = np.cumsum(positive) <= 0.9
            if not wrong_sorted.any():
                wrong_sorted[0] = True
            wrong_probabilities = np.zeros_like(np.asarray(exact))
            original_positive_indices = np.argsort(np.asarray(exact))[::-1][
                : len(positive)
            ]
            wrong_probabilities[original_positive_indices[wrong_sorted]] = positive[
                wrong_sorted
            ]
            wrong_probabilities /= wrong_probabilities.sum()
            wrong_draws = np.asarray(
                jax.random.categorical(
                    jax.random.fold_in(base_key, 99),
                    jnp.log(jnp.asarray(wrong_probabilities)),
                    shape=(sample_count,),
                )
            )
            wrong_counts = np.bincount(wrong_draws, minlength=VOCAB)
            wrong_stat, wrong_p = grouped_chi_square(wrong_counts, np.asarray(exact))
            results["top_p_drop_crossing_negative_control"] = {
                "chi2_statistic": wrong_stat,
                "chi2_p_value": wrong_p,
                "draws": sample_count,
                "pass": wrong_p < 0.001,
            }
    greedy = jax.vmap(
        lambda key: jnp.argmax(
            sampler_probabilities(logits, "temperature", 0.0001)
        )
    )(jax.random.split(base_key, 1000))
    most_likely = int(jnp.argmax(logits))
    return {
        "context_position": position,
        "context_entropy": float(entropy[position]),
        "context_logits": all_logits[position].tolist(),
        "samplers": results,
        "temperature_0.0001_greedy": {
            "pass": bool(np.all(np.asarray(greedy) == most_likely)),
            "draws": 1000,
            "most_likely": most_likely,
        },
    }


def self_consistency(params: dict, sequences: int, length: int) -> dict:
    apply = jax.jit(lambda current, inputs: transformer_logits(current, inputs))

    def run(temperature: float, seed: int) -> dict[str, float]:
        key = jax.random.PRNGKey(seed)
        context = jnp.full((sequences, 64), 2, dtype=jnp.int32)
        surprise = np.zeros(sequences, dtype=np.float64)
        entropy = np.zeros(sequences, dtype=np.float64)
        for _ in range(length):
            logits = apply(params, context)[:, -1]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            probs = jnp.exp(log_probs)
            entropy_values = -jnp.sum(probs * log_probs, axis=-1)
            key, sample_key = jax.random.split(key)
            sampling_logits = logits / temperature
            token = jax.random.categorical(sample_key, sampling_logits)
            token_surprise = -jnp.take_along_axis(
                log_probs, token[:, None], axis=-1
            )[:, 0]
            surprise += np.asarray(token_surprise)
            entropy += np.asarray(entropy_values)
            context = jnp.roll(context, -1, axis=1).at[:, -1].set(token)
        difference = surprise / length - entropy / length
        z_score = float(
            difference.mean() / (difference.std(ddof=1) / math.sqrt(sequences))
        )
        return {
            "mean_surprise": float((surprise / length).mean()),
            "mean_predicted_entropy": float((entropy / length).mean()),
            "z_score": z_score,
        }

    return {
        "temperature_1": run(1.0, 5315),
        "temperature_0.7": run(0.7, 5316),
        "sequences": sequences,
        "length": length,
    }


def make_figures(
    metrics: dict,
    part2: dict,
) -> None:
    model_names = ["uniform", "unigram", "bigram", "trigram", "4-gram", "transformer"]
    values = [
        part2["models"]["uniform"]["bits_per_char"],
        part2["models"]["1gram"]["bits_per_char"],
        part2["models"]["2gram"]["bits_per_char"],
        part2["models"]["3gram"]["bits_per_char"],
        part2["models"]["4gram"]["bits_per_char"],
        metrics["evaluation"]["bits_per_char"],
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = ["#b8b8b8"] * 5 + ["#c0392b"]
    ax.bar(model_names, values, color=colors)
    for x, value in enumerate(values):
        ax.text(x, value + 0.05, f"{value:.3f}", ha="center", fontsize=9)
    ax.axhline(float(part2["models"]["4gram"]["bits_per_char"]), color="black", ls="--", lw=1)
    ax.set_ylabel("held-out bits per character")
    ax.set_title("Count models and the transformer on the same validation stories")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part5-model-bpc-comparison.png", dpi=180)
    plt.close(fig)

    position = np.arange(1, 65)
    loss = np.asarray(metrics["evaluation"]["position_nats"])
    error = np.asarray(metrics["evaluation"]["position_standard_error"])
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(position, loss, color="#c0392b", label="transformer")
    ax.fill_between(position, loss - 2 * error, loss + 2 * error, color="#c0392b", alpha=0.18)
    ax.axhline(part2["models"]["2gram"]["nats"], color="#2878b5", ls="--", label="bigram")
    ax.axhline(part2["models"]["3gram"]["nats"], color="#2a9d8f", ls="--", label="trigram")
    ax.set_xlabel("tokens read in the window")
    ax.set_ylabel("mean held-out cross-entropy (nats/token)")
    ax.set_title("Loss at every position of a 64-token context")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part5-position-loss.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    cache = load_token_cache()
    params, training_metadata = train(settings, cache, args.force)
    evaluation = evaluate(params, cache["valid_tokens"], cache["valid_chars"])
    print(
        f"[Part 5] held-out: {evaluation['nats']:.6f} nats/token, "
        f"{evaluation['bits_per_char']:.6f} bits/char",
        flush=True,
    )
    with path_from("artifacts/part2-metrics.json").open(encoding="utf-8") as handle:
        part2 = json.load(handle)

    tables = ngram_model_tables(cache["train_tokens"], max_n=4)
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "tok512.model"))
    seed = int(settings["part5"]["sample_seed"])
    story_boundaries = np.flatnonzero(cache["train_tokens"] == 2)
    boundary = int(
        story_boundaries[
            np.random.default_rng(seed).integers(1, len(story_boundaries) - 1)
        ]
    )
    next_story_position = boundary + 1

    def prompt_history(order: int) -> np.ndarray:
        count = max(1, order - 1)
        return cache["train_tokens"][
            next_story_position - count : next_story_position
        ]

    uniform_rng = np.random.default_rng(seed)
    uniform_ids: list[int] = []
    while len(uniform_ids) < 60:
        token = int(uniform_rng.integers(0, VOCAB))
        if token != 2:
            uniform_ids.append(token)
    samples = {
        "uniform": decode_ids(sp, uniform_ids),
        "unigram": decode_ids(
            sp, sample_ngram(tables[1], 1, 60, seed, initial_history=prompt_history(1))
        ),
        "bigram": decode_ids(
            sp, sample_ngram(tables[2], 2, 60, seed, initial_history=prompt_history(2))
        ),
        "trigram": decode_ids(
            sp, sample_ngram(tables[3], 3, 60, seed, initial_history=prompt_history(3))
        ),
        "4gram": decode_ids(
            sp, sample_ngram(tables[4], 4, 60, seed, initial_history=prompt_history(4))
        ),
    }
    sampler_settings = [
        ("temperature", 0.8, None, None, None),
        ("top_k", 0.8, 20, None, None),
        ("top_p", 0.8, None, 0.9, None),
        ("min_p", 0.7, None, None, 0.1),
    ]
    transformer_samples = {}
    for mode, temperature, top_k, top_p, min_p in sampler_settings:
        transformer_samples[mode] = generate_transformer(
            params,
            sp,
            seed,
            mode,
            temperature,
            top_k,
            top_p,
            min_p,
        )
    samples["transformer"] = generate_transformer(
        params, sp, seed, "temperature", 1.0
    )
    samples["transformer_temperature_0.8"] = transformer_samples["temperature"]
    samples["transformer_top_k_20"] = transformer_samples["top_k"]
    samples["transformer_top_p_0.9"] = transformer_samples["top_p"]
    samples["transformer_min_p_0.1"] = transformer_samples["min_p"]

    sampler_results = sampler_checks(params, cache["valid_tokens"])
    consistency = self_consistency(
        params,
        int(settings["part5"]["self_consistency_sequences"]),
        int(settings["part5"]["self_consistency_length"]),
    )
    late_loss = float(np.mean(np.asarray(evaluation["position_nats"])[31:]))
    last_eight_loss = float(np.mean(np.asarray(evaluation["position_nats"])[-8:]))
    checks = {
        "beats_4gram_bpc": {
            "pass": evaluation["bits_per_char"] < float(part2["models"]["4gram"]["bits_per_char"]),
            "actual": evaluation["bits_per_char"],
            "target": part2["models"]["4gram"]["bits_per_char"],
        },
        "late_positions_below_trigram": {
            "pass": late_loss < float(part2["models"]["3gram"]["nats"]),
            "actual": late_loss,
            "target": part2["models"]["3gram"]["nats"],
        },
        "last_eight_positions": {
            "pass": last_eight_loss < float(part2["models"]["3gram"]["nats"]),
            "actual": last_eight_loss,
            "reference": 1.65,
            "target": float(part2["models"]["3gram"]["nats"]),
        },
        "complete_validation_coverage": {
            "pass": bool(evaluation["coverage"]["complete"]),
            **evaluation["coverage"],
        },
        "sampler_chi_square": {
            name: {
                "pass": bool(result["pass"]),
                "p_value": result["chi2_p_value"],
            }
            for name, result in sampler_results["samplers"].items()
        },
        "greedy_limit": sampler_results["temperature_0.0001_greedy"],
        "top_p_drop_crossing_negative_control": sampler_results["samplers"][
            "top_p_drop_crossing_negative_control"
        ],
        "self_consistency_temperature_1": {
            "pass": -4 <= consistency["temperature_1"]["z_score"] <= 4,
            "actual_z": consistency["temperature_1"]["z_score"],
        },
        "self_consistency_temperature_0.7": {
            "pass": abs(consistency["temperature_0.7"]["z_score"]) > 40,
            "actual_z": consistency["temperature_0.7"]["z_score"],
        },
    }

    def leaves(node: object) -> list[bool]:
        if isinstance(node, dict):
            if isinstance(node.get("pass"), bool):
                return [node["pass"]]
            return [item for value in node.values() for item in leaves(value)]
        return []

    metrics = {
        "architecture": {
            "pre_norm": True,
            "layers": settings["part5"]["n_layers"],
            "width": settings["part5"]["width"],
            "heads": settings["part5"]["heads"],
            "context": settings["part5"]["context"],
            "all_linear_layers_have_bias": True,
            "mlp_width": 4 * int(settings["part5"]["width"]),
        },
        "training": {
            "steps": settings["part5"]["steps"],
            "batch_size": settings["part5"]["batch_size"],
            "optimizer": "AdamW",
            "peak_learning_rate": settings["part5"]["learning_rate"],
            "warmup_steps": settings["part5"]["warmup_steps"],
            "cosine_final_fraction": settings["part5"]["final_lr_fraction"],
            "gradient_global_norm_clip": 1.0,
            **training_metadata,
        },
        "evaluation": evaluation,
        "samples": samples,
        "sampler_checks": sampler_results,
        "self_consistency": consistency,
        "checks": checks,
        "all_checks_pass": all(leaves(checks)),
    }
    make_figures(metrics, part2)
    write_json("part5-metrics.json", metrics)
    print(json.dumps({key: metrics[key] for key in ("evaluation", "sampler_checks", "self_consistency", "checks", "all_checks_pass")}, indent=2))
    if not metrics["all_checks_pass"]:
        raise SystemExit("Part 5 acceptance check failed")


if __name__ == "__main__":
    main()

