#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <full_shape|text_guided|interactive|refine> <model_path> <mesh_path> <output_dir> [options]" >&2
  exit 2
fi

MODE=$1
MODEL_PATH=$2
INPUT_PATH=$3
OUTPUT_DIR=$4
shift 4

case "$MODE" in
  full_shape)
    [[ -d "$INPUT_PATH" ]] || { echo "Full-shape segmentation expects a directory: $INPUT_PATH" >&2; exit 2; }
    ;;
  text_guided|interactive) ;;
  refine)
    [[ -f "$INPUT_PATH" ]] || { echo "Refine expects one mesh file: $INPUT_PATH" >&2; exit 2; }
    ;;
  *) echo "Unsupported task: $MODE" >&2; exit 2 ;;
esac

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TORCH=0
export USE_TF=0
exec python -m partllm_jittor.infer \
  --task "$MODE" \
  --model_path "$MODEL_PATH" \
  --config_path "${CONFIG_PATH:-$MODEL_PATH}" \
  --mesh_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
