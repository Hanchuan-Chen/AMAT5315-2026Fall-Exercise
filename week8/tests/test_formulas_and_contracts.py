from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common import capacity_crossing, load_settings, ngram_codes, ngram_statistics
from part1_embeddings import relation_scores, unit_arrow
from part2_ngrams import explicit_normalization_errors, per_context_chi_square
from part3_memory import recall_ratios
from part4_attention import recall as picture_recall
from part5_transformer import sample_ngram
from part6_scaling import law, objective_and_gradient_log, optimal
from transformer_lib import (
    init_transformer,
    sampler_probabilities,
    transformer_logits,
)


def test_settings_match_the_learning_sheet() -> None:
    settings = load_settings()
    assert settings["part1"]["seed"] == 1730
    assert settings["part2"]["max_n"] == 6
    assert settings["part2"]["bigram_samples"] == 250000
    assert settings["part3"]["dimensions"] == [16, 32, 64]
    assert settings["part3"]["steps"] == 4000
    assert settings["part4"]["character"] == "算"
    assert settings["part4"]["softmax_beta"] == 8.0
    assert settings["part5"]["width"] == 96
    assert settings["part5"]["heads"] == 4
    assert settings["part5"]["context"] == 64
    assert settings["part5"]["steps"] == 6000
    assert settings["part6"]["bootstrap"] == 400
    assert settings["part6"]["huber_delta"] == 0.001


def test_relation_scores_use_leave_one_out_direction() -> None:
    rng = np.random.default_rng(7)
    dimension = 32
    words: list[str] = []
    vocab: dict[str, int] = {}
    rows = []
    common = rng.normal(size=dimension)
    common /= np.linalg.norm(common)
    for pair in range(9):
        start = rng.normal(size=dimension)
        start /= np.linalg.norm(start)
        end = start + common + 0.03 * rng.normal(size=dimension)
        end /= np.linalg.norm(end)
        words.extend([f"a{pair}", f"b{pair}"])
        vocab[f"a{pair}"] = len(rows)
        rows.append(start)
        vocab[f"b{pair}"] = len(rows)
        rows.append(end)
    scores = relation_scores(np.asarray(rows), vocab, words)
    assert scores.shape == (9,)
    assert scores.mean() > 0.8
    assert np.all(scores > 0.65)


def test_relation_arrows_have_the_requested_horizontal_endpoint() -> None:
    for cosine in (-0.8, -0.2, 0.0, 0.4, 0.9):
        horizontal, vertical = unit_arrow(cosine)
        assert horizontal == pytest.approx(cosine)
        assert horizontal**2 + vertical**2 == pytest.approx(1.0)


def test_packed_ngrams_and_statistics_align_one_to_one() -> None:
    tokens = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    codes = ngram_codes(tokens, 3)
    expected = np.asarray(
        [
            (0 * 512 + 1) * 512 + 2,
            (1 * 512 + 2) * 512 + 3,
            (2 * 512 + 3) * 512 + 4,
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(codes, expected)
    keys, counts, context_keys, context_counts = ngram_statistics(tokens, 4)
    assert counts.sum() == len(tokens) - 3
    assert context_counts.sum() == len(tokens) - 3
    assert len(keys) == len(counts)
    assert len(context_keys) == len(context_counts)


def test_bigram_sequence_normalization_is_one_through_length_three() -> None:
    rng = np.random.default_rng(11)
    p = rng.random((512, 512))
    p /= p.sum(axis=1, keepdims=True)
    errors = explicit_normalization_errors(p)
    assert max(errors.values()) < 1e-12


def test_context_chi_squares_combine_statistics_and_degrees_of_freedom() -> None:
    observed = np.asarray([[60, 30, 10], [80, 10, 10]])
    probabilities = observed / observed.sum(axis=1, keepdims=True)
    statistic, degrees_of_freedom, p_value = per_context_chi_square(
        observed, probabilities
    )
    assert statistic == pytest.approx(0.0, abs=1e-12)
    assert degrees_of_freedom == 4
    assert p_value == pytest.approx(1.0)


def test_outer_and_least_squares_are_exact_below_capacity() -> None:
    keys = np.eye(4)
    values = np.eye(4)
    queries = np.arange(4)
    assert np.max(recall_ratios(keys, values, "outer", queries)) < 1e-12
    assert np.max(recall_ratios(keys, values, "least_squares", queries)) < 1e-12
    assert np.max(recall_ratios(keys, values, "relu2", queries)) < 1e-12


def test_capacity_interpolation_is_in_log_memory() -> None:
    memory = np.asarray([16.0, 64.0])
    ratio = np.asarray([0.5, 2.0])
    crossing = capacity_crossing(memory, ratio)
    assert crossing == pytest.approx(32.0, rel=1e-10)


def test_softmax_picture_store_silences_interference() -> None:
    rng = np.random.default_rng(1730)
    dimension = 64
    count = 4096
    keys = rng.normal(size=(count, dimension))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.choice(np.asarray([-1.0, 1.0]), size=(count, 1024))
    picture = rng.choice(np.asarray([-1.0, 1.0]), size=1024)
    values[0] = picture
    recalled, own = picture_recall(keys[:1], keys, values, "softmax", 8.0)
    ratio = np.linalg.norm(recalled[0] - picture) / np.linalg.norm(picture)
    assert ratio < 0.1
    assert own[0] == pytest.approx(1.0)


def test_transformer_has_the_required_linear_biases() -> None:
    params = init_transformer(jax.random.PRNGKey(0))
    assert len(params["blocks"]) == 2
    for block in params["blocks"]:
        for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            assert name in block
            assert name.replace("W", "b") in block
    tokens = jnp.zeros((2, 64), dtype=jnp.int32)
    assert transformer_logits(params, tokens).shape == (2, 64, 512)


def test_causal_mask_blocks_future_tokens() -> None:
    params = init_transformer(jax.random.PRNGKey(3))
    first = jnp.zeros((1, 64), dtype=jnp.int32)
    second = first.at[0, 40].set(123).at[0, 50].set(321)
    logits_first = transformer_logits(params, first)
    logits_second = transformer_logits(params, second)
    np.testing.assert_allclose(
        np.asarray(logits_first[0, :40]),
        np.asarray(logits_second[0, :40]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_top_k_keeps_exactly_k_tokens() -> None:
    logits = jnp.arange(20, dtype=jnp.float32)
    probabilities = sampler_probabilities(logits, "top_k", 0.8, top_k=5)
    assert int(jnp.sum(probabilities > 0)) == 5
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)


def test_top_p_includes_the_token_that_crosses_the_threshold() -> None:
    probabilities = jnp.asarray([0.6, 0.3, 0.1])
    logits = jnp.log(probabilities)
    sampled = sampler_probabilities(logits, "top_p", 1.0, top_p=0.8)
    assert int(jnp.sum(sampled > 0)) == 2
    assert float(sampled[0]) == pytest.approx(0.6 / 0.9)
    assert float(sampled[1]) == pytest.approx(0.3 / 0.9)


def test_min_p_and_greedy_limits() -> None:
    probabilities = jnp.asarray([0.5, 0.2, 0.2, 0.1])
    logits = jnp.log(probabilities)
    sampled = sampler_probabilities(logits, "min_p", 1.0, min_p=0.2)
    assert np.all((np.asarray(sampled) == 0) | (np.asarray(sampled) >= 0.2 * 0.5 - 1e-7))
    np.testing.assert_allclose(
        np.asarray(sampler_probabilities(logits, "temperature", 0.0001)),
        [1.0, 0.0, 0.0, 0.0],
    )


def test_count_sampler_never_returns_the_end_token() -> None:
    tokens = np.asarray([2, 10, 11, 12, 2, 10, 11, 13], dtype=np.int32)
    full_keys, full_counts, context_keys, context_counts = ngram_statistics(tokens, 2)
    table = {
        "full_keys": full_keys,
        "full_counts": full_counts,
        "context_keys": context_keys,
        "context_counts": context_counts,
        "k": 0.01,
    }
    sample = sample_ngram(
        table,
        2,
        60,
        8,
        initial_history=np.asarray([2], dtype=np.int64),
    )
    assert len(sample) == 60
    assert 2 not in sample


def test_scaling_law_and_gradient_at_the_fit() -> None:
    parameters = np.asarray([1.8172101, 477.8, 2143.0, 0.3473, 0.3672])
    n = np.asarray([1e8, 1e9, 1e10])
    d = np.asarray([2e9, 2e10, 2e11])
    log_loss = np.log(law(parameters, n, d))
    log_parameters = np.asarray(
        [
            np.log(parameters[1]),
            np.log(parameters[2]),
            np.log(parameters[0]),
            parameters[3],
            parameters[4],
        ]
    )
    loss, gradient = objective_and_gradient_log(
        log_parameters, n, d, log_loss, delta=0.001
    )
    assert loss == pytest.approx(0.0, abs=1e-20)
    np.testing.assert_allclose(gradient, np.zeros(5), atol=1e-12)


def test_compute_optimal_allocation_formula() -> None:
    parameters = np.asarray([1.8172101, 477.8, 2143.0, 0.3473, 0.3672])
    n_star, d_star = optimal(parameters, np.asarray([5.76e23]))
    ratio = d_star[0] / n_star[0]
    assert ratio == pytest.approx(17.92, abs=0.2)
    assert n_star[0] == pytest.approx(7.32e10, rel=0.02)
    assert d_star[0] == pytest.approx(1.31e12, rel=0.02)


def test_manual_packed_code_fits_in_uint64() -> None:
    assert 512**6 < 2**64
    assert 512**6 > 2**53
    assert math.log2(512**6) == 54

