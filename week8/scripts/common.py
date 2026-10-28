from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
INPUTS = ROOT / "inputs"

EVIDENCE.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


def load_settings() -> dict:
    with (ROOT / "settings.toml").open("rb") as handle:
        return tomllib.load(handle)


def path_from(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(name: str, value: object) -> None:
    with (ARTIFACTS / name).open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.item() if isinstance(item, np.generic) else item,
        )
        handle.write("\n")


def read_stories() -> list[str]:
    text = (DATA / "TinyStoriesV2-GPT4-valid.txt").read_text(encoding="utf-8")
    stories = [story.strip() for story in text.split("<|endoftext|>")]
    return [story for story in stories if story]


def split_stories(stories: list[str], fraction: float) -> tuple[list[str], list[str]]:
    cut = int(fraction * len(stories))
    return stories[:cut], stories[cut:]


def encoded_stream(sp, stories: list[str]) -> tuple[np.ndarray, int]:
    eos = int(sp.eos_id())
    pieces: list[np.ndarray] = [np.asarray([eos], dtype=np.int32)]
    for story in stories:
        ids = np.asarray(sp.encode(story, out_type=int), dtype=np.int32)
        pieces.append(ids)
        pieces.append(np.asarray([eos], dtype=np.int32))
    stream = np.concatenate(pieces)
    return stream, sum(len(story) for story in stories)


def encode_token_cache() -> dict:
    import sentencepiece as spm

    settings = load_settings()
    stories = read_stories()
    train_stories, valid_stories = split_stories(
        stories, float(settings["part2"]["train_fraction"])
    )
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "tok512.model"))
    train_tokens, train_chars = encoded_stream(sp, train_stories)
    valid_tokens, valid_chars = encoded_stream(sp, valid_stories)
    mismatches = sum(
        sp.decode(sp.encode(story, out_type=int)) != story for story in valid_stories
    )

    cache_path = path_from(settings["paths"]["token_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train=train_tokens,
        valid=valid_tokens,
        vocab_size=np.asarray([sp.vocab_size()], dtype=np.int32),
        train_stories=np.asarray([len(train_stories)]),
        valid_stories=np.asarray([len(valid_stories)]),
        train_chars=np.asarray([train_chars], dtype=np.int64),
        valid_chars=np.asarray([valid_chars], dtype=np.int64),
        decode_mismatches=np.asarray([mismatches]),
    )
    return {
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
        "train_chars": train_chars,
        "valid_chars": valid_chars,
        "vocab_size": sp.vocab_size(),
        "train_stories": len(train_stories),
        "valid_stories": len(valid_stories),
        "decode_mismatches": mismatches,
    }


def load_token_cache() -> dict:
    settings = load_settings()
    path = path_from(settings["paths"]["token_cache"])
    if not path.exists():
        return encode_token_cache()
    with np.load(path) as data:
        return {
            "train_tokens": data["train"].astype(np.int32),
            "valid_tokens": data["valid"].astype(np.int32),
            "train_chars": int(data["train_chars"][0]),
            "valid_chars": int(data["valid_chars"][0]),
            "vocab_size": int(data["vocab_size"][0]),
            "train_stories": int(data["train_stories"][0]),
            "valid_stories": int(data["valid_stories"][0]),
            "decode_mismatches": int(data["decode_mismatches"][0]),
        }


def ngram_codes(tokens: np.ndarray, length: int) -> np.ndarray:
    """Packed base-V codes for every consecutive window of `length` tokens."""
    tokens = np.asarray(tokens, dtype=np.uint64)
    if length < 1 or length > len(tokens):
        return np.empty(0, dtype=np.uint64)
    vocab = np.uint64(512)
    out = np.zeros(len(tokens) - length + 1, dtype=np.uint64)
    for offset in range(length):
        out *= vocab
        out += tokens[offset : offset + len(out)]
    return out


def ngram_statistics(
    tokens: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Training full-n counts, context counts, and target codes.

    There is one prediction for every token after the leading end-of-sequence
    context.  Contexts with n=1 are represented by the single empty code zero.
    """
    if n == 1:
        context = np.zeros(len(tokens) - 1, dtype=np.uint64)
        full = np.asarray(tokens[1:], dtype=np.uint64)
    else:
        context = ngram_codes(tokens[:-1], n - 1)
        full = ngram_codes(tokens, n)
    full_unique, full_counts = np.unique(full, return_counts=True)
    context_unique, context_counts = np.unique(context, return_counts=True)
    return full_unique, full_counts, context_unique, context_counts


def lookup_counts(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(keys, query)
    idx_clip = np.minimum(idx, max(len(keys) - 1, 0))
    found = (idx < len(keys)) & (keys[idx_clip] == query) if len(keys) else np.zeros_like(
        query, dtype=bool
    )
    return np.where(found, values[idx_clip] if len(keys) else 0, 0)


def ngram_probabilities(
    tokens: np.ndarray,
    n: int,
    full_keys: np.ndarray,
    full_counts: np.ndarray,
    context_keys: np.ndarray,
    context_counts: np.ndarray,
    k: float,
    vocab_size: int = 512,
) -> np.ndarray:
    if n == 1:
        contexts = np.zeros(len(tokens) - 1, dtype=np.uint64)
        full_queries = np.asarray(tokens[1:], dtype=np.uint64)
    else:
        contexts = ngram_codes(tokens[:-1], n - 1)
        full_queries = ngram_codes(tokens, n)
    numerators = lookup_counts(full_keys, full_counts, full_queries).astype(np.float64)
    denominators = lookup_counts(context_keys, context_counts, contexts).astype(np.float64)
    return (numerators + k) / (denominators + k * vocab_size)


def capacity_crossing(memory: np.ndarray, ratio: np.ndarray, target: float = 1.0) -> float:
    """Interpolate the first crossing on log(M), log(R)."""
    memory = np.asarray(memory, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    for i in range(1, len(memory)):
        if ratio[i - 1] <= target < ratio[i]:
            x0, x1 = np.log(memory[i - 1]), np.log(memory[i])
            y0, y1 = np.log(max(ratio[i - 1], 1e-15)), np.log(ratio[i])
            t = (np.log(target) - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0)))
        if ratio[i - 1] > target >= ratio[i]:
            x0, x1 = np.log(memory[i - 1]), np.log(memory[i])
            y0, y1 = np.log(ratio[i - 1]), np.log(max(ratio[i], 1e-15))
            t = (np.log(target) - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0)))
    return float("nan")

