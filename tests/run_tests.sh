#!/usr/bin/env bash
# run_tests.sh
set -euo pipefail

DLL="${DLL:-build/CipherCore_OpenCl.dll}"
GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"
MARKERS="${MARKERS:-}"

export STREAMLIT_HEADLESS="true"

# ENV nur setzen, wenn nicht bereits vorhanden
: "${CIPHERCORE_DLL:=$DLL}"
: "${CIPHERCORE_GPU:=$GPU}"

echo "Using CIPHERCORE_DLL=$CIPHERCORE_DLL"
echo "Using CIPHERCORE_GPU=$CIPHERCORE_GPU"

ARGS=("-rs")
if [[ "$SMOKE" == "1" ]]; then
  ARGS+=("-m" "smoke")
elif [[ -n "$MARKERS" ]]; then
  ARGS+=("-m" "$MARKERS")
fi

# CLI-Overrides (siehe conftest.py)
ARGS+=("--dll" "$CIPHERCORE_DLL" "--gpu" "$CIPHERCORE_GPU")

python -m pytest "${ARGS[@]}"
