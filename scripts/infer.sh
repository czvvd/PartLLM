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

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${CONFIG_PATH:-$MODEL_PATH}

case "$MODE" in
  full_shape)
    if [[ ! -d "$INPUT_PATH" ]]; then
      echo "Full-shape segmentation expects a directory of meshes: $INPUT_PATH" >&2
      exit 2
    fi
    INPUT_ARGS=(--glb_dir "$INPUT_PATH")
    ENTRY="$ROOT/recipe/part_seg/tools/inference/test_mask_pred.py"
    ;;
  text_guided)
    INPUT_ARGS=(--mesh_path "$INPUT_PATH")
    ENTRY="$ROOT/recipe/part_seg/tools/inference/infer_grounding.py"
    ;;
  interactive)
    INPUT_ARGS=(--mesh_path "$INPUT_PATH")
    ENTRY="$ROOT/recipe/part_seg/tools/inference/infer_promptable.py"
    ;;
  refine)
    if [[ ! -f "$INPUT_PATH" ]]; then
      echo "Refine mode expects one mesh file: $INPUT_PATH" >&2
      exit 2
    fi
    INPUT_ARGS=(--mesh_path "$INPUT_PATH" --require_initial_mask)
    ENTRY="$ROOT/recipe/part_seg/tools/inference/infer_promptable.py"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

cd "$ROOT"

VLLM_PYTHON=${VLLM_PYTHON:-$ROOT/.venv-vllm/bin/python}
VLLM_BACKEND=${USE_VLLM:-auto}
if [[ "$VLLM_BACKEND" == "auto" ]]; then
  if [[ -x "$VLLM_PYTHON" ]]; then
    VLLM_BACKEND=1
  else
    VLLM_BACKEND=0
  fi
fi

if [[ "$VLLM_BACKEND" == "1" ]]; then
  [[ -x "$VLLM_PYTHON" ]] || {
    echo "Missing vLLM environment. Run: bash scripts/install_vllm.sh" >&2
    exit 2
  }
  [[ -d "$MODEL_PATH" ]] || {
    echo "vLLM acceleration requires a local checkpoint directory: $MODEL_PATH" >&2
    exit 2
  }
  BRIDGE_ROOT="$ROOT/recipe/part_seg/vllm_bridge"
  VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-$ROOT/.cache/partllm-vllm-model}
  export PYTHONPATH="$BRIDGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  export VLLM_PLUGINS=partllm_vllm_bridge
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  if [[ ! -f "$VLLM_MODEL_PATH/bridge_manifest.json" ]]; then
    mkdir -p "$(dirname "$VLLM_MODEL_PATH")"
    python "$BRIDGE_ROOT/tools/prepare_model_view.py" \
      --checkpoint "$MODEL_PATH" \
      --config "$CONFIG_PATH" \
      --output "$VLLM_MODEL_PATH"
  fi
  exec python "$BRIDGE_ROOT/tools/run_hybrid_inference.py" \
    --inference_mode "$MODE" \
    --repo_root "$ROOT" \
    --vllm_python "$VLLM_PYTHON" \
    --vllm_model_path "$VLLM_MODEL_PATH" \
    --vllm_mode "${VLLM_MODE:-stable_compiled}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.30}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --vllm_timing_output "$OUTPUT_DIR/vllm_timing.jsonl" \
    --config /dev/null \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    "${INPUT_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
fi

if [[ "$VLLM_BACKEND" != "0" ]]; then
  echo "USE_VLLM must be auto, 1, or 0." >&2
  exit 2
fi

exec python "$ENTRY" \
  --config /dev/null \
  --model_path "$MODEL_PATH" \
  --config_path "$CONFIG_PATH" \
  "${INPUT_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
