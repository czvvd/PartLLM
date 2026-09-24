#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_ENV=${VLLM_ENV:-$ROOT/.venv-vllm}

python -m venv "$VLLM_ENV"
"$VLLM_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VLLM_ENV/bin/python" -m pip install "vllm==0.11.2" "transformers==4.57.6"
"$VLLM_ENV/bin/python" -m pip install -e "$ROOT/recipe/part_seg/vllm_bridge"

echo "vLLM environment installed at $VLLM_ENV"
