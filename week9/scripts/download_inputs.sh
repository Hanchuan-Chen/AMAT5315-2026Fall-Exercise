#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

asia_base=https://raw.githubusercontent.com/TensorBFS/TensorInference.jl
asia_rev=935ed86f8e473f523cac9461c45a1f470e74f0f9
if [ ! -f asia.uai ]; then
  curl --fail --location \
    "$asia_base/$asia_rev/examples/asia-network/model.uai" --output asia.uai
fi

prom_base=https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition
prom_rev=22005844423f5bca62798548df808cc418fc0d7d
mkdir -p promedus
for item in PR/Promedus_14.uai PR/Promedus_14.uai.evid \
            PR/Promedus_14.uai.PR MAR/Promedus_14.uai.MAR; do
  name=$(basename "$item")
  if [ ! -f "promedus/$name" ]; then
    curl --fail --location "$prom_base/$prom_rev/uai2014/$item" \
      --output "promedus/$name"
  fi
done
