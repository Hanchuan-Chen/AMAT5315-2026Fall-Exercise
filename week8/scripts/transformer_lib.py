from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

VOCAB = 512
HEADS = 4


def linear_init(key: jax.Array, inputs: int, outputs: int, scale: float = 1.0) -> tuple:
    weight = scale * jax.random.normal(key, (inputs, outputs)) / math.sqrt(inputs)
    return weight, jnp.zeros((outputs,))


def init_transformer(
    key: jax.Array,
    layers: int = 2,
    width: int = 96,
    heads: int = 4,
    context: int = 64,
    vocab: int = VOCAB,
) -> dict:
    if heads != HEADS:
        raise ValueError(f"the Week 8 transformer requires {HEADS} heads")
    keys = list(jax.random.split(key, 10 + layers * 8))
    params: dict = {
        "E": 0.02 * jax.random.normal(keys[0], (vocab, width)),
        "P": 0.01 * jax.random.normal(keys[1], (context, width)),
        "out_g": jnp.ones((width,)),
        "out_b": jnp.zeros((width,)),
        "blocks": [],
    }
    params["Wout"], params["bout"] = linear_init(keys[2], width, vocab)
    cursor = 3
    for _ in range(layers):
        block = {
            "ln1_g": jnp.ones((width,)),
            "ln1_b": jnp.zeros((width,)),
            "ln2_g": jnp.ones((width,)),
            "ln2_b": jnp.zeros((width,)),
        }
        for name in ("Wq", "Wk", "Wv", "Wo"):
            block[name], block[name.lower().replace("w", "b")] = linear_init(
                keys[cursor], width, width
            )
            cursor += 1
        block["W1"], block["b1"] = linear_init(keys[cursor], width, 4 * width)
        block["W2"], block["b2"] = linear_init(keys[cursor + 1], 4 * width, width)
        cursor += 2
        params["blocks"].append(block)
    return params


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale + bias


def causal_attention(x: jax.Array, block: dict, heads: int) -> jax.Array:
    batch, length, width = x.shape
    head_width = width // heads
    q = x @ block["Wq"] + block["bq"]
    k = x @ block["Wk"] + block["bk"]
    v = x @ block["Wv"] + block["bv"]
    q = q.reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
    k = k.reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
    v = v.reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(head_width)
    future = jnp.triu(jnp.ones((length, length), dtype=bool), k=1)
    scores = jnp.where(future[None, None, :, :], -1e9, scores)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
    return attended @ block["Wo"] + block["bo"]


def transformer_logits(params: dict, tokens: jax.Array) -> jax.Array:
    _, length = tokens.shape
    x = params["E"][tokens] + params["P"][:length][None, :, :]
    heads = HEADS
    for block in params["blocks"]:
        normalized = layer_norm(x, block["ln1_g"], block["ln1_b"])
        x = x + causal_attention(normalized, block, heads)
        normalized = layer_norm(x, block["ln2_g"], block["ln2_b"])
        hidden = jax.nn.gelu(normalized @ block["W1"] + block["b1"])
        x = x + hidden @ block["W2"] + block["b2"]
    x = layer_norm(x, params["out_g"], params["out_b"])
    return x @ params["Wout"] + params["bout"]


def cross_entropy(params: dict, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    logits = transformer_logits(params, inputs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))


def sampler_probabilities(
    logits: jax.Array,
    mode: str = "temperature",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> jax.Array:
    """Exact target probabilities after temperature and optional truncation."""
    logits = jnp.asarray(logits)
    if temperature <= 1e-4:
        winners = jax.nn.one_hot(jnp.argmax(logits), logits.shape[-1])
        return winners / winners.sum()
    scaled = logits / temperature
    probabilities = jax.nn.softmax(scaled, axis=-1)
    if mode == "temperature":
        return probabilities
    if mode == "top_k":
        assert top_k is not None and top_k > 0
        keep = jnp.argsort(jnp.argsort(-scaled)) < top_k
    elif mode == "top_p":
        assert top_p is not None and 0 < top_p <= 1
        order = jnp.argsort(-scaled)
        sorted_probabilities = jax.nn.softmax(jnp.take(scaled, order), axis=-1)
        cumulative_before = jnp.cumsum(sorted_probabilities) - sorted_probabilities
        sorted_keep = cumulative_before < top_p
        keep = jnp.zeros_like(sorted_keep).at[order].set(sorted_keep)
    elif mode == "min_p":
        assert min_p is not None and 0 <= min_p <= 1
        keep = probabilities >= min_p * jnp.max(probabilities)
    else:
        raise ValueError(f"unknown sampler mode {mode}")
    kept = jnp.where(keep, probabilities, 0.0)
    return kept / kept.sum()


def save_checkpoint(path: Path, params: dict, metadata: dict | None = None) -> None:
    leaves, tree = jax.tree_util.tree_flatten(params)
    arrays = {f"p{i:03d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    arrays["_tree"] = np.asarray([repr(tree)], dtype=object)
    if metadata:
        for key, value in metadata.items():
            arrays[f"_meta_{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=True) as data:
        keys = sorted(key for key in data.files if key.startswith("p"))
        leaves = [jnp.asarray(data[key]) for key in keys]
        template = init_transformer(jax.random.PRNGKey(0))
        _, tree = jax.tree_util.tree_flatten(template)
        params = jax.tree_util.tree_unflatten(tree, leaves)
        metadata = {
            key.removeprefix("_meta_"): data[key].item()
            for key in data.files
            if key.startswith("_meta_")
        }
    return params, metadata

