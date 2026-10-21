"""JAX implementation of the Diffusers v0.40.0 UNet2DModel checkpoint.

The checkpoint is ``google/ddpm-cifar10-32``.  Its tensors stay in the
published PyTorch layout on disk and are converted once to JAX's HWIO linear
and convolution layouts here.  No PyTorch runtime is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors import safe_open


def load_unet_params(model_dir: Path) -> tuple[dict[str, jax.Array], dict]:
    with (model_dir / "config.json").open() as handle:
        config = json.load(handle)
    params: dict[str, jax.Array] = {}
    path = model_dir / "diffusion_pytorch_model.safetensors"
    with safe_open(path, framework="numpy") as handle:
        assert handle.metadata() is not None
        for name in handle.keys():
            value = handle.get_tensor(name)
            if name.endswith(".weight") and value.ndim == 4:
                value = value.transpose(2, 3, 1, 0)
            elif name.endswith(".weight") and value.ndim == 2:
                value = value.T
            params[name] = jnp.asarray(value, jnp.float32)
    return params, config


def _conv(
    params: dict[str, jax.Array],
    prefix: str,
    x: jax.Array,
    *,
    stride: int = 1,
    padding: int = 1,
) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        x,
        params[prefix + ".weight"],
        (stride, stride),
        [(padding, padding), (padding, padding)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + params[prefix + ".bias"]


def _linear(params: dict[str, jax.Array], prefix: str, x: jax.Array) -> jax.Array:
    return x @ params[prefix + ".weight"] + params[prefix + ".bias"]


def _group_norm(
    params: dict[str, jax.Array],
    prefix: str,
    x: jax.Array,
    groups: int = 32,
    eps: float = 1e-6,
) -> jax.Array:
    batch, height, width, channels = x.shape
    grouped = x.reshape(batch, height, width, groups, channels // groups)
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = jnp.mean((grouped - mean) ** 2, axis=(1, 2, 4), keepdims=True)
    normalized = ((grouped - mean) / jnp.sqrt(variance + eps)).reshape(x.shape)
    return normalized * params[prefix + ".weight"] + params[prefix + ".bias"]


def _layer_norm(
    params: dict[str, jax.Array],
    prefix: str,
    x: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (
        (x - mean)
        / jnp.sqrt(variance + eps)
        * params[prefix + ".weight"]
        + params[prefix + ".bias"]
    )


def _resnet(
    params: dict[str, jax.Array],
    prefix: str,
    x: jax.Array,
    temb: jax.Array,
) -> jax.Array:
    residual = x
    h = jax.nn.silu(_group_norm(params, prefix + ".norm1", x))
    h = _conv(params, prefix + ".conv1", h)
    time = _linear(params, prefix + ".time_emb_proj", jax.nn.silu(temb))[:, None, None, :]
    h = h + time
    h = jax.nn.silu(_group_norm(params, prefix + ".norm2", h))
    h = _conv(params, prefix + ".conv2", h)
    if prefix + ".conv_shortcut.weight" in params:
        residual = _conv(params, prefix + ".conv_shortcut", residual, padding=0)
    return residual + h


def _attention(params: dict[str, jax.Array], prefix: str, x: jax.Array) -> jax.Array:
    residual = x
    h = _group_norm(params, prefix + ".group_norm", x)
    batch, height, width, channels = h.shape
    h = h.reshape(batch, height * width, channels)
    query = _linear(params, prefix + ".query", h)
    key = _linear(params, prefix + ".key", h)
    value = _linear(params, prefix + ".value", h)
    scores = jnp.einsum("bic,bjc->bij", query, key) / jnp.sqrt(float(channels))
    h = jnp.einsum("bij,bjc->bic", jax.nn.softmax(scores, axis=-1), value)
    h = _linear(params, prefix + ".proj_attn", h)
    return h.reshape(residual.shape) + residual


def _upsample(params: dict[str, jax.Array], prefix: str, x: jax.Array) -> jax.Array:
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return _conv(params, prefix + ".conv", x)


def _downsample(params: dict[str, jax.Array], prefix: str, x: jax.Array) -> jax.Array:
    # The checkpoint sets downsample_padding=0.  Diffusers then pads the
    # bottom and right by one before the stride-two 3x3 convolution.
    x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    return _conv(params, prefix + ".conv", x, stride=2, padding=0)


def _time_embedding(params: dict[str, jax.Array], t: jax.Array) -> jax.Array:
    half = 64
    exponent = -jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1)
    angles = t[:, None].astype(jnp.float32) * jnp.exp(exponent)[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    emb = _linear(params, "time_embedding.linear_1", emb)
    return _linear(params, "time_embedding.linear_2", jax.nn.silu(emb))


@jax.jit
def unet_forward(
    params: dict[str, jax.Array], sample_nhwc: jax.Array, t: jax.Array
) -> jax.Array:
    """Predict noise. ``t`` is already the checkpoint index (mathematical t - 1)."""
    temb = _time_embedding(params, t)
    h = _conv(params, "conv_in", sample_nhwc)
    skips = [h]

    for block in range(4):
        for layer in range(2):
            prefix = f"down_blocks.{block}.resnets.{layer}"
            h = _resnet(params, prefix, h, temb)
            if block == 1:
                h = _attention(params, f"down_blocks.1.attentions.{layer}", h)
            skips.append(h)
        if block < 3:
            h = _downsample(params, f"down_blocks.{block}.downsamplers.0", h)
            skips.append(h)

    h = _resnet(params, "mid_block.resnets.0", h, temb)
    h = _attention(params, "mid_block.attentions.0", h)
    h = _resnet(params, "mid_block.resnets.1", h, temb)

    for block in range(4):
        for layer in range(3):
            skip = skips.pop()
            h = jnp.concatenate([h, skip], axis=-1)
            h = _resnet(params, f"up_blocks.{block}.resnets.{layer}", h, temb)
            if block == 2:
                h = _attention(params, f"up_blocks.2.attentions.{layer}", h)
        if block < 3:
            h = _upsample(params, f"up_blocks.{block}.upsamplers.0", h)

    h = jax.nn.silu(_group_norm(params, "conv_norm_out", h))
    return _conv(params, "conv_out", h)
