#!/usr/bin/env bash
# Part 1's download commands: MNIST from the CVDF mirror and the CIFAR-10
# binary archive from the authors' mirror, into the git-ignored inputs/.
# Each command appends its effective download URL to week6/README.md, as the
# sheet asks.  A URL already recorded in README.md is not appended again, so
# re-running the script (or `make reproduce`) leaves the committed README
# unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p inputs/mnist inputs/cifar10

record_url() {
  local url="$1"
  if ! grep -Fqx -- "$url" README.md 2>/dev/null; then
    printf '%s\n' "$url" >>README.md
  fi
}

mnist_url=https://storage.googleapis.com/cvdf-datasets/mnist
for split in train t10k; do
  for kind in images-idx3-ubyte labels-idx1-ubyte; do
    name="$split-$kind.gz"
    url=$(curl -fL "$mnist_url/$name" -o "inputs/mnist/$name" \
      --write-out "%{url_effective}")
    record_url "$url"
  done
done

cifar_host=https://mindspore-website.obs.cn-north-4.myhuaweicloud.com
url=$(curl -fL "$cifar_host/notebook/datasets/cifar-10-binary.tar.gz" \
  -o inputs/cifar10/cifar-10-binary.tar.gz \
  --write-out "%{url_effective}")
record_url "$url"
