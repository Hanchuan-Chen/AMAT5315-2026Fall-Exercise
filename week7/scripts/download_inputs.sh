#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p ../week6/inputs/cifar10 inputs/cifar-model

if [[ ! -f inputs/images.npz ]]; then
  curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz \
    -o inputs/images.npz
fi

if [[ ! -f ../week6/inputs/cifar10/cifar-10-binary.tar.gz ]]; then
  curl -fL \
    https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz \
    -o ../week6/inputs/cifar10/cifar-10-binary.tar.gz
fi

if [[ ! -f inputs/cifar-model/config.json \
  || ! -f inputs/cifar-model/diffusion_pytorch_model.safetensors ]]; then
  .venv/bin/hf download google/ddpm-cifar10-32 \
    config.json diffusion_pytorch_model.safetensors \
    --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
    --local-dir inputs/cifar-model
fi
