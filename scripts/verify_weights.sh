#!/usr/bin/env bash
# Verifies that every LTX-2.3 weight expected by handler.py is present and
# non-empty on the attached network volume. Exits non-zero on any missing file.
set -uo pipefail

ROOT="${LTX_WEIGHTS_ROOT:-/workspace/models}"

# name : min-bytes (rough lower bounds to catch truncated downloads)
declare -a EXPECTED=(
  "ltx-2.3-22b-dev.safetensors:20000000000"
  "ltx-2.3-22b-distilled.safetensors:20000000000"
  "ltx-2.3-22b-distilled-lora-384.safetensors:100000000"
  "ltx-2.3-spatial-upscaler-x2-1.1.safetensors:100000000"
  "ltx-2.3-temporal-upscaler-x2-1.0.safetensors:100000000"
)

fail=0
printf "%-50s %15s %15s %s\n" "FILE" "ACTUAL" "MIN-EXPECTED" "STATUS"
printf -- "----------------------------------------------------------------------------------------------\n"

for entry in "${EXPECTED[@]}"; do
  name="${entry%%:*}"
  min="${entry##*:}"
  path="$ROOT/$name"
  if [[ ! -f "$path" ]]; then
    printf "%-50s %15s %15s %s\n" "$name" "-" "$min" "MISSING"
    fail=1
    continue
  fi
  actual=$(stat -c '%s' "$path" 2>/dev/null || stat -f '%z' "$path")
  if (( actual < min )); then
    printf "%-50s %15s %15s %s\n" "$name" "$actual" "$min" "TOO_SMALL"
    fail=1
  else
    printf "%-50s %15s %15s %s\n" "$name" "$actual" "$min" "ok"
  fi
done

gemma="$ROOT/gemma-3-12b"
if [[ ! -d "$gemma" ]] || [[ -z "$(ls -A "$gemma" 2>/dev/null)" ]]; then
  printf "%-50s %15s %15s %s\n" "gemma-3-12b/" "-" "dir" "MISSING"
  fail=1
else
  gemma_bytes=$(du -sb "$gemma" 2>/dev/null | cut -f1 || du -sk "$gemma" | awk '{print $1 * 1024}')
  printf "%-50s %15s %15s %s\n" "gemma-3-12b/" "$gemma_bytes" "dir" "ok"
fi

if (( fail != 0 )); then
  echo ""
  echo "FAIL: one or more weights missing or truncated."
  exit 1
fi
echo ""
echo "OK: all LTX-2.3 weights present."
