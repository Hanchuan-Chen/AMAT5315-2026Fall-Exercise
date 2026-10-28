from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, capacity_crossing, load_settings, load_token_cache, write_json

jax.config.update("jax_enable_x64", False)
VOCAB = 512
STORE_NAMES = ("outer", "least_squares", "relu1", "relu2", "relu3")


def unit_rows(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    values = rng.normal(size=(count, dimension))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def recall_ratios(
    keys: np.ndarray,
    values: np.ndarray,
    store: str,
    query_indices: np.ndarray,
) -> np.ndarray:
    queries = keys[query_indices]
    if store == "outer":
        weights = queries @ keys.T
        recalled = weights @ values
        signal = values[query_indices]
    elif store == "least_squares":
        # Expand the fitted prediction in the basis of stored values.  The
        # diagonal coefficient is the recalled memory's signal; every
        # off-diagonal coefficient is interference.
        coefficients = queries @ np.linalg.pinv(keys.T @ keys, rcond=1e-12) @ keys.T
        recalled = coefficients @ values
        diagonal = coefficients[np.arange(len(query_indices)), query_indices]
        signal = diagonal[:, None] * values[query_indices]
    else:
        power = int(store[-1])
        similarities = queries @ keys.T
        weights = np.maximum(similarities, 0.0) ** power
        recalled = weights @ values
        signal = values[query_indices]
    return np.linalg.norm(recalled - signal, axis=1) / np.linalg.norm(signal, axis=1)


def measure_store_curve(settings: dict) -> tuple[dict, dict, dict]:
    p3 = settings["part3"]
    dimensions = [int(x) for x in p3["dimensions"]]
    memory_grid = [int(x) for x in p3["memories"]]
    draws = int(p3["draws"])
    query_sample = int(p3["query_sample"])
    curves: dict[str, dict[str, dict[str, float]]] = {}
    raw: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    capacities: dict[str, dict[str, float]] = {}

    for dimension in dimensions:
        print(f"[Part 3] memory curves d={dimension}", flush=True)
        draws_by_store = {
            name: {"memory": [], "ratio": [], "draw_ratios": [[] for _ in memory_grid]}
            for name in STORE_NAMES
        }
        for draw in range(draws):
            rng = np.random.default_rng(1000 * dimension + draw)
            for index, count in enumerate(memory_grid):
                keys = unit_rows(rng, count, dimension)
                values = unit_rows(rng, count, dimension)
                query_indices = rng.choice(
                    count, size=min(count, query_sample), replace=False
                )
                for store in STORE_NAMES:
                    ratio = recall_ratios(keys, values, store, query_indices)
                    draws_by_store[store]["draw_ratios"][index].append(
                        float(np.median(ratio))
                    )

        curves[str(dimension)] = {}
        raw[str(dimension)] = {}
        for store in STORE_NAMES:
            medians = np.asarray(
                [
                    np.median(values)
                    for values in draws_by_store[store]["draw_ratios"]
                ],
                dtype=np.float64,
            )
            curves[str(dimension)][store] = {
                "memory": memory_grid,
                "ratio": medians.tolist(),
            }
            raw[str(dimension)][store] = medians
            capacities.setdefault(store, {})[str(dimension)] = capacity_crossing(
                np.asarray(memory_grid), medians
            )

    for dimension in dimensions:
        np.savez_compressed(
            ARTIFACTS / f"part3-memory-d{dimension}.npz",
            memory=np.asarray(memory_grid),
            **{name: raw[str(dimension)][name] for name in STORE_NAMES},
        )
    return curves, capacities, raw


def plot_curves(raw: dict, output_name: str) -> dict[str, float]:
    settings = load_settings()["part3"]
    memory = np.asarray(settings["memories"], dtype=float)
    d = int(settings["dimensions"][-1])
    colors = {
        "outer": "#1f4e79",
        "least_squares": "#2a9d8f",
        "relu1": "#e9c46a",
        "relu2": "#f4a261",
        "relu3": "#e76f51",
    }
    labels = {
        "outer": "outer product",
        "least_squares": "least squares",
        "relu1": r"ReLU$^1$",
        "relu2": r"ReLU$^2$",
        "relu3": r"ReLU$^3$",
    }
    predictions = {
        "outer": np.sqrt(memory / d),
        "least_squares": np.sqrt(np.maximum(memory / d - 1.0, 1e-15)),
        "relu1": np.sqrt(memory / (2.0 * d)),
        "relu2": np.sqrt(3.0 * memory / (2.0 * d**2)),
        "relu3": np.sqrt(15.0 * memory / (2.0 * d**3)),
    }
    slopes: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for store in STORE_NAMES:
        ratio = raw[str(d)][store]
        ax.loglog(memory, np.maximum(ratio, 1e-14), "o-", color=colors[store], label=labels[store])
        ax.loglog(memory, predictions[store], "--", color=colors[store], alpha=0.35)
        mask = ratio > 0.1
        if mask.sum() >= 2:
            slopes[store] = float(
                np.polyfit(np.log(memory[mask]), np.log(ratio[mask]), 1)[0]
            )
        else:
            slopes[store] = float("nan")
    ax.axhline(1.0, color="black", lw=0.9)
    ax.set_xlabel("number of memories M")
    ax.set_ylabel("interference-to-signal ratio R")
    ax.set_title(r"Memory recall at d=64 (median of 3 draws)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(EVIDENCE / output_name, dpi=180)
    plt.close(fig)
    return slopes


def fit_capacity_slopes(capacities: dict, dimensions: list[int]) -> dict[str, float]:
    slopes: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    colors = {
        "outer": "#1f4e79",
        "least_squares": "#2a9d8f",
        "relu1": "#e9c46a",
        "relu2": "#f4a261",
        "relu3": "#e76f51",
    }
    for store in STORE_NAMES:
        y = np.asarray([capacities[store][str(d)] for d in dimensions], dtype=float)
        valid = np.isfinite(y)
        slopes[store] = float(np.polyfit(np.log(np.asarray(dimensions)[valid]), np.log(y[valid]), 1)[0])
        ax.loglog(dimensions, y, "o-", color=colors[store], label=store.replace("_", " "))
        if valid.sum() >= 2:
            fit = np.exp(np.polyval(np.polyfit(np.log(np.asarray(dimensions)[valid]), np.log(y[valid]), 1), np.log(dimensions)))
            ax.loglog(dimensions, fit, "--", color=colors[store], alpha=0.35)
    ax.set_xlabel("vector dimension d")
    ax.set_ylabel("capacity M where R = 1")
    ax.set_title("Capacity exponents")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "part3-capacity-vs-dimension.png", dpi=180)
    plt.close(fig)
    return slopes


def init_mlp(key: jax.Array) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 4)
    return {
        "E": 0.05 * jax.random.normal(keys[0], (VOCAB, 64)),
        "W1": jax.random.normal(keys[1], (64, 512)) / np.sqrt(64),
        "b1": jnp.zeros((512,)),
        "W2": jax.random.normal(keys[2], (512, VOCAB)) / np.sqrt(512),
        "b2": jnp.zeros((VOCAB,)),
    }


def mlp_logits(params: dict[str, jax.Array]) -> jax.Array:
    hidden = jnp.tanh(params["E"] @ params["W1"] + params["b1"])
    return hidden @ params["W2"] + params["b2"]


def sampled_counts(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, batch: int) -> np.ndarray:
    indices = rng.integers(0, len(x), size=batch)
    flat = x[indices].astype(np.int64) * VOCAB + y[indices]
    return np.bincount(flat, minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB).astype(np.float32)


def train_mlp(settings: dict, cache: dict) -> dict[str, object]:
    p3 = settings["part3"]
    train = cache["train_tokens"].astype(np.int64)
    valid = cache["valid_tokens"].astype(np.int64)
    x_train, y_train = train[:-1], train[1:]
    x_valid, y_valid = valid[:-1], valid[1:]
    rng = np.random.default_rng(int(p3["seed"]))
    params = init_mlp(jax.random.PRNGKey(int(p3["seed"])))
    adam_m = jax.tree.map(jnp.zeros_like, params)
    adam_v = jax.tree.map(jnp.zeros_like, params)
    steps = int(p3["steps"])
    batch = int(p3["batch_size"])
    peak_lr = float(p3["learning_rate"])
    warmup = int(0.05 * steps)
    weight_decay = float(p3["weight_decay"])
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def loss_fn(current: dict[str, jax.Array], counts: jax.Array) -> jax.Array:
        logits = mlp_logits(current)
        log_probs = jax.nn.log_softmax(logits, axis=1)
        return -jnp.sum(counts * log_probs) / batch

    @jax.jit
    def step(
        current: dict[str, jax.Array],
        m: dict[str, jax.Array],
        v: dict[str, jax.Array],
        counts: jax.Array,
        lr: jax.Array,
        iteration: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(current, counts)
        m = jax.tree.map(lambda old, g: beta1 * old + (1 - beta1) * g, m, grads)
        v = jax.tree.map(lambda old, g: beta2 * old + (1 - beta2) * g * g, v, grads)
        power = iteration + 1
        correction1 = 1 - beta1**power
        correction2 = 1 - beta2**power
        updated = jax.tree.map(
            lambda p, m_i, v_i: p * (1 - lr * weight_decay)
            - lr * (m_i / correction1) / (jnp.sqrt(v_i / correction2) + eps),
            current,
            m,
            v,
        )
        return updated, m, v, loss

    started = time.time()
    last_loss = float("nan")
    for iteration in range(steps):
        if iteration < warmup:
            lr = peak_lr * (iteration + 1) / warmup
        else:
            progress = (iteration - warmup) / max(1, steps - warmup)
            lr = 0.05 * peak_lr + 0.95 * peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
        counts = sampled_counts(x_train, y_train, rng, batch)
        params, adam_m, adam_v, loss = step(
            params,
            adam_m,
            adam_v,
            jnp.asarray(counts),
            jnp.asarray(lr, dtype=jnp.float32),
            jnp.asarray(iteration, dtype=jnp.int32),
        )
        last_loss = float(loss)
        if iteration % 250 == 0 or iteration + 1 == steps:
            print(
                f"[Part 3] MLP step {iteration + 1}/{steps}: "
                f"train sample loss {last_loss:.6f}, lr {lr:.6f}",
                flush=True,
            )
    elapsed = time.time() - started

    def full_loss(x: np.ndarray, y: np.ndarray) -> float:
        counts = np.bincount(
            x.astype(np.int64) * VOCAB + y.astype(np.int64),
            minlength=VOCAB * VOCAB,
        ).reshape(VOCAB, VOCAB)
        logits = np.asarray(mlp_logits(params))
        log_probs = logits - logits.max(axis=1, keepdims=True)
        log_probs -= np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
        return float(-(counts * log_probs).sum() / counts.sum())

    train_loss = full_loss(x_train, y_train)
    valid_loss = full_loss(x_valid, y_valid)
    np.savez_compressed(ARTIFACTS / "part3-mlp.npz", **jax.tree.map(np.asarray, params))
    return {
        "steps": steps,
        "examples_per_step": batch,
        "final_sample_loss": last_loss,
        "train_nats": train_loss,
        "heldout_nats": valid_loss,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    settings = load_settings()
    curves, capacities, raw = measure_store_curve(settings)
    slopes_r = plot_curves(raw, "part3-r-vs-memories.png")
    dimensions = [int(x) for x in settings["part3"]["dimensions"]]
    slopes_capacity = fit_capacity_slopes(capacities, dimensions)
    mlp = train_mlp(settings, load_token_cache())
    with (ARTIFACTS / "part2-metrics.json").open(encoding="utf-8") as handle:
        part2 = json.load(handle)
    bigram_loss = float(part2["models"]["2gram"]["nats"])
    d64 = curves["64"]
    checks = {
        "outer_crossing_50_to_85": {
            "pass": 50.0 <= capacities["outer"]["64"] <= 85.0,
            "actual": capacities["outer"]["64"],
        },
        "least_squares_exact_through_64": {
            "pass": float(np.max(raw["64"]["least_squares"][:4])) < 1e-10,
            "actual": float(np.max(raw["64"]["least_squares"][:4])),
            "target": 1e-10,
        },
        "least_squares_crossing_100_to_170": {
            "pass": 100.0 <= capacities["least_squares"]["64"] <= 170.0,
            "actual": capacities["least_squares"]["64"],
        },
        "power_slopes": {
            "outer_near_1": {"pass": abs(slopes_capacity["outer"] - 1.0) <= 0.1, "actual": slopes_capacity["outer"]},
            "least_squares_near_1": {"pass": abs(slopes_capacity["least_squares"] - 1.0) <= 0.1, "actual": slopes_capacity["least_squares"]},
            "relu1_near_1": {"pass": abs(slopes_capacity["relu1"] - 1.0) <= 0.1, "actual": slopes_capacity["relu1"]},
            "relu2_near_2": {"pass": abs(slopes_capacity["relu2"] - 2.0) <= 0.15, "actual": slopes_capacity["relu2"]},
            "relu3_above_2.5": {"pass": slopes_capacity["relu3"] > 2.5, "actual": slopes_capacity["relu3"]},
        },
        "reported_crossings_near_reference": {
            "relu2": {"pass": 1500 <= capacities["relu2"]["64"] <= 5000, "actual": capacities["relu2"]["64"], "reference": 2871},
            "relu3": {"pass": 10000 <= capacities["relu3"]["64"] <= 80000, "actual": capacities["relu3"]["64"], "reference": 38124},
        },
        "curve_slopes_after_r_0.1": {
            name: {"pass": abs(value - 0.5) <= 0.15, "actual": value}
            for name, value in slopes_r.items()
        },
        "mlp_matches_bigram": {
            "pass": abs(mlp["heldout_nats"] - bigram_loss) <= 0.01,
            "actual": mlp["heldout_nats"],
            "bigram_nats": bigram_loss,
            "difference": mlp["heldout_nats"] - bigram_loss,
            "target": 0.01,
        },
    }

    def leaves(node: object) -> list[bool]:
        if isinstance(node, dict):
            if isinstance(node.get("pass"), bool):
                return [node["pass"]]
            return [v for item in node.values() for v in leaves(item)]
        return []

    all_pass = all(leaves(checks))
    result = {
        "query_policy": f"median over up to {settings['part3']['query_sample']} stored keys per draw",
        "curves": curves,
        "capacities": capacities,
        "r_curve_slopes": slopes_r,
        "capacity_slopes": slopes_capacity,
        "mlp": mlp,
        "checks": checks,
        "all_checks_pass": all_pass,
    }
    write_json("part3-metrics.json", result)
    print(json.dumps(result, indent=2))
    if not all_pass:
        raise SystemExit("Part 3 acceptance check failed")


if __name__ == "__main__":
    main()

