#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/data" "$root/inputs"

stories_rev=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
bpe_rev=0bd21da
potion_rev=bf8b056
chinchilla_rev=92258837425e1b5f2851d624287f0120583a3d0e

curl -L --fail --retry 3 -o "$root/data/TinyStoriesV2-GPT4-valid.txt" \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/$stories_rev/TinyStoriesV2-GPT4-valid.txt"
curl -L --fail --retry 3 -o "$root/data/tok512.model" \
  "https://huggingface.co/karpathy/tinyllamas/resolve/$bpe_rev/stories260K/tok512.model"
curl -L --fail --retry 3 -o "$root/data/chinchilla-runs.csv" \
  "https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/$chinchilla_rev/data/svg_extracted_data.csv"
curl -L --fail --retry 3 -o "$root/inputs/model.safetensors" \
  "https://huggingface.co/minishlab/potion-base-8M/resolve/$potion_rev/model.safetensors"
curl -L --fail --retry 3 -o "$root/inputs/tokenizer.json" \
  "https://huggingface.co/minishlab/potion-base-8M/resolve/$potion_rev/tokenizer.json"

printf '%s\n' "Pinned inputs downloaded."

