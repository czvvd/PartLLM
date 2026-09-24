# Training

## Data Preparation

PartLLM provides converters for PartNeXt, 3DCoMPaT200, PartVerse, and HY3D-Bench. The converters only write annotations and derived mesh files; the original datasets remain subject to their own licenses and access terms.

### PartNeXt

Download the official [annotations](https://huggingface.co/datasets/AuWang/PartNeXt) and [meshes](https://huggingface.co/datasets/AuWang/PartNeXt_mesh):

```bash
hf download AuWang/PartNeXt --repo-type dataset \
  --local-dir datasets/PartNeXt/annotations

hf download AuWang/PartNeXt_mesh --repo-type dataset \
  --local-dir datasets/PartNeXt/meshes
```

The annotation directory must contain `state.json`, `dataset_info.json`, and the Arrow shards. Meshes are stored under `meshes/glbs/` using the official Objaverse-style subdirectories.

```bash
python recipe/part_seg/tools/prepare_partnext_dataset.py \
  --annotation_dir datasets/PartNeXt/annotations \
  --glb_dir datasets/PartNeXt/meshes/glbs \
  --output_dir datasets/PartNeXt/parquet \
  --prompt_mode all \
  --depth_mode all \
  --num_proc 16
```

This produces `train.parquet`, `test.parquet`, per-mode training files, and a reproducible `split.json`.

### 3DCoMPaT200

Download `Compat200.zip` from the official [3DCoMPaT200 dataset repository](https://huggingface.co/datasets/CoMPaT/3DCoMPaT200), then clone the official [3DCoMPaT-v2 code](https://github.com/Vision-CAIR/3DCoMPaT-v2) for its metadata and GLTF loader:

```bash
hf download CoMPaT/3DCoMPaT200 --repo-type dataset \
  --include Compat200.zip \
  --local-dir datasets/3DCoMPaT200/data

git clone https://github.com/Vision-CAIR/3DCoMPaT-v2.git \
  datasets/3DCoMPaT200/3DCoMPaT-v2
```

Convert the official archive to the shared PartLLM schema:

```bash
python recipe/part_seg/tools/prepare_compat200_dataset.py \
  --zip_path datasets/3DCoMPaT200/data/Compat200.zip \
  --meta_dir datasets/3DCoMPaT200/3DCoMPaT-v2/metadata \
  --loader_dir datasets/3DCoMPaT200/3DCoMPaT-v2/loaders/3D \
  --glb_dir datasets/3DCoMPaT200/glbs \
  --output_dir datasets/3DCoMPaT200/parquet \
  --num_proc 16 \
  --verify
```

This produces `train.parquet`, `test.parquet`, and per-mode training files. The converted GLBs in `glbs/` are referenced by the Parquet rows and must remain available during training.

### PartVerse

Download the official [PartVerse release](https://huggingface.co/datasets/dscdyc/partverse). PartLLM uses `anno_infos`, `textured_part_glbs`, and `text_captions.json`; `normalized_glbs` is not required by the converter.

```bash
hf download dscdyc/partverse --repo-type dataset \
  --include "anno_infos.tar.gz" \
  --include "text_captions.json" \
  --include "textured_part_glbs.tar.gz.*" \
  --local-dir datasets/PartVerse/download

mkdir -p datasets/PartVerse/raw
tar -xzf datasets/PartVerse/download/anno_infos.tar.gz \
  -C datasets/PartVerse/raw
cat datasets/PartVerse/download/textured_part_glbs.tar.gz.* | \
  tar -xz -C datasets/PartVerse/raw
cp datasets/PartVerse/download/text_captions.json \
  datasets/PartVerse/raw/text_captions.json
```

The extracted directory contains:

```text
datasets/PartVerse/raw/
├── anno_infos/
├── textured_part_glbs/
└── text_captions.json
```

Run the converter:

```bash
python recipe/part_seg/tools/prepare_partverse_dataset.py \
  --data_root datasets/PartVerse/raw \
  --glb_dir datasets/PartVerse/glbs \
  --output_dir datasets/PartVerse/parquet \
  --num_proc 16 \
  --test_size 0.05
```

The converter assembles the per-part files into one GLB per object and writes `train.parquet`, `test.parquet`, per-mode training files, and `split.json`.

### HY3D-Bench

Download the required mesh shards from the official [HY3D-Bench dataset](https://huggingface.co/datasets/tencent/HY3D-Bench).

Download and extract one shard:

```bash
hf download tencent/HY3D-Bench --repo-type dataset \
  --include "part/meshes/00.tar.gz" \
  --local-dir datasets/HY3D-Bench/download

mkdir -p datasets/HY3D-Bench/part/meshes
tar -xzf datasets/HY3D-Bench/download/part/meshes/00.tar.gz \
  -C datasets/HY3D-Bench/part/meshes
```

To prepare all shards, replace the include pattern with `part/meshes/*.tar.gz` and extract each archive into `datasets/HY3D-Bench/part/meshes/`. The extracted shard directories contain the `.npz` objects used by the converter.

```bash
python recipe/part_seg/tools/prepare_hy3dbench_dataset.py \
  --data_root datasets/HY3D-Bench/part \
  --output_dir datasets/HY3D-Bench/parquet \
  --num_proc 16 \
  --chunk_size 50 \
  --skip_watertight \
  --test_size 0.05
```

With `--skip_watertight`, the converter reads the per-part PLY entries from each NPZ and writes resumable shards, `train.parquet`, `test.parquet`, and per-mode training files. Add `--limit 100` to process the first 100 objects.

### Prepared Data Layout

Keep the generated Parquet files and their referenced geometry in the following layout:

```text
datasets/
├── PartNeXt/
│   ├── meshes/glbs/<type_id>/<model_id>.glb
│   └── parquet/
│       ├── train.parquet
│       ├── test.parquet
│       ├── train_*.parquet
│       └── split.json
├── 3DCoMPaT200/
│   ├── glbs/<shape_id>.glb
│   └── parquet/
│       ├── train.parquet
│       ├── test.parquet
│       └── train_*.parquet
├── PartVerse/
│   ├── glbs/<uid>.glb
│   └── parquet/
│       ├── train.parquet
│       ├── test.parquet
│       ├── train_*.parquet
│       └── split.json
└── HY3D-Bench/
    ├── part/meshes/<shard>/<uid>.npz
    └── parquet/
        ├── train.parquet
        ├── test.parquet
        ├── train_*.parquet
        └── shards/shard_*.parquet
```

Each Parquet row stores paths to the corresponding GLB or NPZ geometry. Run training from the repository root and keep these paths unchanged after conversion. Regenerate the Parquet files if the geometry directories are moved.

PartNeXt uses the default paths in `scripts/train.sh`:

```bash
bash scripts/train.sh
```

## Training Options

| Variable | Default | Description |
|---|---:|---|
| `MODEL_PATH` | `Czvvd/PartLLM` | Local checkpoint path or Hugging Face model ID. |
| `TRAIN_PATH` | `datasets/PartNeXt/parquet/train.parquet` | Training Parquet file. |
| `VAL_PATH` | `datasets/PartNeXt/parquet/test.parquet` | Validation Parquet file. |
| `OUTPUT_DIR` | `checkpoints/partllm_sft` | Checkpoint output directory. |
| `NPROC_PER_NODE` | Visible GPU count | Training processes and GPUs per node. |
| `NNODES` | `1` | Number of nodes; run `scripts/train.sh` on every node. |
| `NODE_RANK` | Required for multi-node | Unique rank from `0` to `NNODES-1` on each node. |
| `MASTER_ADDR` | Required for multi-node | Address of the rank-0 node, shared by all nodes. |
| `MASTER_PORT` | `29500` | Rendezvous port. |
| `RDZV_ID` | `partllm` | c10d rendezvous job ID, shared by all nodes. |
| `FSDP_SIZE` | `NPROC_PER_NODE` | FSDP group size within each node. |
| `GLOBAL_BATCH_SIZE` | Global GPU count | Global batch size across all nodes. |
| `MICRO_BATCH_SIZE_PER_GPU` | `1` | Per-GPU micro batch size. |
| `TOTAL_EPOCHS` | `1` | Number of SFT epochs. |
| `LR` | `1e-5` | Base learning rate. |
| `MAX_LENGTH` | `15240` | Maximum sequence length. |
| `SAVE_FREQ` | `500` | Checkpoint interval in steps. |
| `TEST_FREQ` | `500` | Validation interval in steps. |
| `RESUME_MODE` | `disable` | Trainer resume mode. |
