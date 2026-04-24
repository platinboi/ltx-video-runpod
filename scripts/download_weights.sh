#!/usr/bin/env bash
# Downloads every LTX-2.3 weight + the Gemma encoder required by the pipelines
# onto a RunPod network volume mounted at /workspace. Run this once on a cheap
# CPU or GPU pod (with the volume attached) before building the serverless image.
#
# Requires:
#   - huggingface_hub installed (`pip install -U huggingface_hub`)
#   - HF_TOKEN exported (needed for the gated Gemma repo; also lifts LTX rate limits)
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Export your HuggingFace token first." >&2
  exit 1
fi

ROOT="${LTX_WEIGHTS_ROOT:-/workspace/models}"
mkdir -p "$ROOT" "$ROOT/gemma-3-12b"

LTX_FILES=(
  "ltx-2.3-22b-dev.safetensors"
  "ltx-2.3-22b-distilled.safetensors"
  "ltx-2.3-22b-distilled-lora-384.safetensors"
  "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
  "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
)

echo "==> Downloading LTX-2.3 weights to $ROOT"
huggingface-cli download \
  Lightricks/LTX-2.3 \
  "${LTX_FILES[@]}" \
  --local-dir "$ROOT" \
  --token "$HF_TOKEN"

for f in "${LTX_FILES[@]}"; do
  size=$(du -h "$ROOT/$f" | cut -f1)
  echo "  [ok] $f  ($size)"
done

echo "==> Downloading Gemma-3-12B (QAT Q4 unquantized) to $ROOT/gemma-3-12b"
huggingface-cli download \
  google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "$ROOT/gemma-3-12b" \
  --token "$HF_TOKEN"

gemma_size=$(du -sh "$ROOT/gemma-3-12b" | cut -f1)
echo "  [ok] gemma-3-12b/  ($gemma_size)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Running verification"
bash "$SCRIPT_DIR/verify_weights.sh"
