import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

FEATURES = 12
HIDDEN = 48
LEARNING_RATE = 4e-3
BATCH_SIZE = 64
EPOCHS = 120
WEIGHT_DECAY = 2e-4
SEED = 2026


def load_nodes(path):
    data = json.loads(Path(path).read_text())
    if data["features"] != [
        "x_val", "frac_dist", "field_h", "abs_field_h", "deg_norm", "bond_bias",
        "eff_field", "abs_eff_field", "mean_bond_dual", "mean_bond_gap",
        "mean_nbr_frac", "gap_ratio",
    ]:
        raise ValueError("dataset feature order does not match the design")
    train = data["nodes"]
    train_ids = {node["instance"] for node in train if node["instance"] < 36}
    val = [node for node in train if node["instance"] >= 36]
    train = [node for node in train if node["instance"] < 36]
    val_ids = {node["instance"] for node in val}
    if train_ids & val_ids or not train or not val:
        raise ValueError("training and validation instances overlap or are empty")
    return train, val, train_ids, val_ids


def padded(nodes):
    width = max(len(node["candidates"]) for node in nodes)
    x = np.zeros((len(nodes), width, FEATURES), dtype=np.float64)
    mask = np.zeros((len(nodes), width), dtype=np.float64)
    target = np.zeros((len(nodes), width), dtype=np.float64)
    for row, node in enumerate(nodes):
        count = len(node["candidates"])
        if len(set(node["candidates"])) != count:
            raise ValueError("candidate spin ids must be unique")
        x[row, :count] = np.asarray(node["features"], dtype=np.float64)
        mask[row, :count] = 1.0
        position = {spin: index for index, spin in enumerate(node["candidates"])}
        missing = [spin for spin in node["best_candidates"] if spin not in position]
        if not node["best_candidates"] or missing:
            raise ValueError(f"best candidates not in candidate list: {missing}")
        for spin in node["best_candidates"]:
            target[row, position[spin]] += 1.0 / len(node["best_candidates"])
        if not np.isclose(target[row].sum(), 1.0):
            raise ValueError("target probabilities must sum to one")
    return jnp.asarray(x), jnp.asarray(mask), jnp.asarray(target)


def init_params(key):
    key_w1, key_w2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(key_w1, (FEATURES, HIDDEN)) * np.sqrt(2.0 / FEATURES),
        "b1": jnp.zeros((HIDDEN,)),
        "w2": jax.random.normal(key_w2, (HIDDEN,)) * np.sqrt(2.0 / HIDDEN),
        "b2": jnp.asarray(0.0),
    }


def logits(params, features, mean, scale):
    standardized = (features - mean) / scale
    hidden = jnp.tanh(jnp.einsum("...k,kj->...j", standardized, params["w1"]) + params["b1"])
    return jnp.einsum("...j,j->...", hidden, params["w2"]) + params["b2"]


def masked_cross_entropy(params, x, mask, target, mean, scale, penalize):
    raw = logits(params, x, mean, scale)
    masked = jnp.where(mask > 0, raw, -jnp.inf)
    log_prob = masked - jax.nn.logsumexp(masked, axis=-1, keepdims=True)
    loss = -jnp.sum(jnp.where(mask > 0, target * log_prob, 0.0)) / x.shape[0]
    penalty = WEIGHT_DECAY * (jnp.sum(params["w1"] ** 2) + jnp.sum(params["w2"] ** 2))
    return loss + jnp.where(penalize, penalty, 0.0)


def label_accuracy(params, nodes, mean, scale):
    x, mask, _ = padded(nodes)
    raw = logits(params, x, mean, scale)
    best_local = np.asarray(jnp.argmax(jnp.where(mask > 0, raw, -jnp.inf), axis=1))
    hits = sum(
        node["candidates"][int(local)] in set(node["best_candidates"])
        for node, local in zip(nodes, best_local)
    )
    return hits / len(nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_nodes, val_nodes, train_ids, val_ids = load_nodes(args.dataset)
    x_train, mask_train, target_train = padded(train_nodes)
    x_val, mask_val, target_val = padded(val_nodes)
    mean = x_train[mask_train > 0].mean(axis=0)
    scale = jnp.maximum(x_train[mask_train > 0].std(axis=0), 1e-6)

    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    moments = {key: jnp.zeros_like(value) for key, value in params.items()}
    velocities = {key: jnp.zeros_like(value) for key, value in params.items()}
    beta1, beta2 = 0.9, 0.999
    step = 0
    history = {"train": [], "validation": []}
    best_val = np.inf
    best_epoch = 0
    best_params = jax.tree.map(lambda value: value, params)

    value_and_grad = jax.value_and_grad(masked_cross_entropy)
    for epoch in range(1, EPOCHS + 1):
        key, batch_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(batch_key, len(train_nodes)))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            _, grads = value_and_grad(
                params,
                x_train[batch],
                mask_train[batch],
                target_train[batch],
                mean,
                scale,
                True,
            )
            step += 1
            for name in params:
                moments[name] = beta1 * moments[name] + (1 - beta1) * grads[name]
                velocities[name] = beta2 * velocities[name] + (1 - beta2) * grads[name] ** 2
                m_hat = moments[name] / (1 - beta1**step)
                v_hat = velocities[name] / (1 - beta2**step)
                params[name] = params[name] - LEARNING_RATE * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        train_loss = float(
            masked_cross_entropy(
                params, x_train, mask_train, target_train, mean, scale, False
            )
        )
        val_loss = float(
            masked_cross_entropy(params, x_val, mask_val, target_val, mean, scale, False)
        )
        history["train"].append(train_loss)
        history["validation"].append(val_loss)
        if (epoch == 1 or epoch % 5 == 0) and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_params = jax.tree.map(lambda value: value, params)

    probes = []
    indices = np.unique(np.rint(np.linspace(0, len(val_nodes) - 1, 6)).astype(int))
    if len(indices) != 6:
        raise ValueError("dataset does not contain six distinct probe nodes")
    for index in indices:
        node = val_nodes[int(index)]
        features = jnp.asarray(node["features"])
        raw = logits(best_params, features, mean, scale)
        probabilities = jax.nn.softmax(raw)
        probes.append(
            {
                "instance": node["instance"],
                "node": node["node"],
                "candidates": node["candidates"],
                "features": node["features"],
                "logits": np.asarray(raw).tolist(),
                "probabilities": np.asarray(probabilities).tolist(),
            }
        )

    output = {
        "format_version": 1,
        "precision": "64-bit JAX on CPU",
        "features": [
            "x_val", "frac_dist", "field_h", "abs_field_h", "deg_norm", "bond_bias",
            "eff_field", "abs_eff_field", "mean_bond_dual", "mean_bond_gap",
            "mean_nbr_frac", "gap_ratio",
        ],
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(scale).tolist(),
        "weights": {
            "w1": np.asarray(best_params["w1"]).tolist(),
            "b1": np.asarray(best_params["b1"]).tolist(),
            "w2": np.asarray(best_params["w2"]).tolist(),
            "b2": float(best_params["b2"]),
        },
        "history": history,
        "checkpoint_epoch": best_epoch,
        "best_validation_cross_entropy": best_val,
        "split": {
            "train_instances": sorted(train_ids),
            "validation_instances": sorted(val_ids),
            "overlap": sorted(train_ids & val_ids),
            "train_nodes": len(train_nodes),
            "validation_nodes": len(val_nodes),
        },
        "probes": probes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    train_accuracy = label_accuracy(best_params, train_nodes, mean, scale)
    validation_accuracy = label_accuracy(best_params, val_nodes, mean, scale)
    print(
        "train_nodes={} train_instances={} validation_nodes={} validation_instances={} "
        "overlap={} checkpoint={} validation_ce={:.6f} "
        "train_label_accuracy={:.6f} validation_label_accuracy={:.6f}".format(
            len(train_nodes),
            len(train_ids),
            len(val_nodes),
            len(val_ids),
            len(train_ids & val_ids),
            best_epoch,
            best_val,
            train_accuracy,
            validation_accuracy,
        )
    )


if __name__ == "__main__":
    main()
