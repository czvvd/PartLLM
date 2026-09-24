#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ $# -eq 4 ]]; then
  MODEL_PATH=$1
  TRAIN_PATH=$2
  VAL_PATH=$3
  OUTPUT_DIR=$4
elif [[ $# -eq 0 ]]; then
  MODEL_PATH=${MODEL_PATH:-Czvvd/PartLLM}
  TRAIN_PATH=${TRAIN_PATH:-$ROOT/datasets/PartNeXt/parquet/train.parquet}
  VAL_PATH=${VAL_PATH:-$ROOT/datasets/PartNeXt/parquet/test.parquet}
  OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/checkpoints/partllm_sft}
else
  echo "Usage: $0 [<model_path> <train.parquet> <val.parquet> <output_dir>]" >&2
  exit 2
fi

for path in "$TRAIN_PATH" "$VAL_PATH"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

NPROC_PER_NODE=${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}
[[ "$NPROC_PER_NODE" -gt 0 ]] || { echo "No CUDA devices found" >&2; exit 2; }
NNODES=${NNODES:-1}
[[ "$NNODES" -gt 0 ]] || { echo "NNODES must be positive" >&2; exit 2; }
FSDP_SIZE=${FSDP_SIZE:-$NPROC_PER_NODE}
[[ "$FSDP_SIZE" -le "$NPROC_PER_NODE" ]] || {
  echo "FSDP_SIZE must not exceed NPROC_PER_NODE; FSDP groups stay within each node." >&2
  exit 2
}
(( NPROC_PER_NODE % FSDP_SIZE == 0 )) || {
  echo "NPROC_PER_NODE must be divisible by FSDP_SIZE." >&2
  exit 2
}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$WORLD_SIZE}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
CONFIG_PATH=${CONFIG_PATH:-$ROOT/recipe/part_seg/config_seg}
LOGGER=${LOGGER:-"['console']"}

cd "$ROOT"
python scripts/patch_transformers.py --check
export UTONIA_LOAD_FROM_HF_CKPT=${UTONIA_LOAD_FROM_HF_CKPT:-1}

if [[ "$NNODES" -eq 1 ]]; then
  TORCHRUN_ARGS=(--standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE")
else
  : "${NODE_RANK:?Set NODE_RANK to the zero-based rank of this node}"
  : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node address}"
  MASTER_PORT=${MASTER_PORT:-29500}
  RDZV_ID=${RDZV_ID:-partllm}
  TORCHRUN_ARGS=(
    --nnodes="$NNODES"
    --node-rank="$NODE_RANK"
    --rdzv-id="$RDZV_ID"
    --rdzv-backend=c10d
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
    --nproc-per-node="$NPROC_PER_NODE"
  )
fi

exec torchrun "${TORCHRUN_ARGS[@]}" \
  -m verl.trainer.sft_trainer \
  data.train_files="['$TRAIN_PATH']" \
  data.val_files="['$VAL_PATH']" \
  data.train_batch_size="$GLOBAL_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="${MAX_LENGTH:-15240}" \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU:-15240}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.use_dynamic_bsz=False \
  data.custom_cls.path="$ROOT/recipe/part_seg/partnext_dataset.py" \
  data.custom_cls.name=PartNeXtPoint3DDataset \
  model.path="$MODEL_PATH" \
  model.hf_config_path="$CONFIG_PATH" \
  model.tokenizer_path="$CONFIG_PATH" \
  model.trust_remote_code=True \
  +model.drop_modules=visual \
  model.use_remove_padding=True \
  engine=fsdp \
  optim=fsdp \
  engine.strategy=fsdp2 \
  engine.fsdp_size="$FSDP_SIZE" \
  engine.ulysses_sequence_parallel_size=1 \
  optim.lr="${LR:-1e-5}" \
  optim.lr_encoder="${LR_ENCODER:-1e-5}" \
  optim.lr_mapping="${LR_MAPPING:-2e-5}" \
  optim.lr_llm="${LR_LLM:-1e-5}" \
  optim.lr_decoder="${LR_DECODER:-2e-5}" \
  trainer.logger="$LOGGER" \
  trainer.project_name=PartLLM \
  trainer.experiment_name="${EXP_NAME:-public_sft}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.test_freq="${TEST_FREQ:-500}" \
  trainer.save_freq="${SAVE_FREQ:-500}" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  checkpoint.save_contents='[model,optimizer,extra]' \
  +data.augmentation.enabled=true \
  +data.augmentation.rotate_y.enabled=true \
  +data.augmentation.rotate_y.angle='[-3.1415926536,3.1415926536]' \
  +data.augmentation.rotate_y.p=0.5 \
  +data.augmentation.chromatic_jitter.enabled=true \
  +data.augmentation.chromatic_jitter.std=0.01 \
  +data.augmentation.chromatic_jitter.p=0.1
