# PartLLM Inference

The unified wrapper exposes the three part segmentation tasks studied in the paper:

```bash
bash scripts/infer.sh \
  <full_shape|text_guided|interactive|refine> \
  <model_path> \
  <mesh_path> \
  <output_dir> \
  [options]
```

`CONFIG_PATH` defaults to `model_path`. Set it separately only when the model configuration, processor, or tokenizer is stored in another directory.

Supported mesh formats are `.glb`, `.gltf`, `.ply`, `.obj`, `.stl`, `.off`, and `.fbx`. Full-shape segmentation expects a directory; text-guided and interactive segmentation accept either one mesh or a directory.

## vLLM Acceleration

The optional vLLM backend accelerates autoregressive token generation while keeping the point encoder, hidden-state extraction, and mask decoder on the original inference path. It supports full-shape, text-guided, interactive, and refine commands through the same interface described below.

Install the isolated vLLM environment once:

```bash
bash scripts/install_vllm.sh
```

After installation, every `scripts/infer.sh` command automatically uses vLLM. On the first run, PartLLM creates a lightweight model view under `.cache/`; it links to the downloaded checkpoint and does not duplicate its weights. No changes to the inference commands are required.

```bash
bash scripts/infer.sh full_shape "$MODEL_PATH" /path/to/meshes outputs/full_shape
```

Set `USE_VLLM=0` to use the original Hugging Face backend instead:

```bash
USE_VLLM=0 \
bash scripts/infer.sh full_shape "$MODEL_PATH" /path/to/meshes outputs/full_shape
```

`VLLM_MODE=stable_compiled` is the default and provides the fastest repeated inference after the initial compilation. Use `VLLM_MODE=eager` for debugging. `VLLM_GPU_MEMORY_UTILIZATION` defaults to `0.30`; reduce it if the combined point encoder, Hugging Face model, and vLLM worker run out of memory. `VLLM_MAX_MODEL_LEN` defaults to `32768`. Per-request timing is written to `vllm_timing.jsonl` in the output directory.

## Full-Shape Segmentation

Full-shape segmentation decomposes each input shape into parts at the requested granularity.

```bash
bash scripts/infer.sh full_shape "$MODEL_PATH" /path/to/meshes outputs/full_shape
```

| Option | Description |
|---|---|
| `--sem_mode all|semantic|nosem` | Run semantic prompts, non-semantic prompts, or both. Default: `semantic`. |
| `--prompt_set all|coarse|general|fine` | Run all predefined granularities or one selected granularity. Default: `all`. |
| `--num_parts N` | Run a numeric prompt requesting approximately `N` parts instead of `--prompt_set`. |
| `--cache_dir PATH` | Reuse a cache produced by `preprocess_glb_cache.py`. |

### Full-shape prompts

By default, `--sem_mode semantic --prompt_set all` runs the coarse, general, and fine semantic prompts, which request part masks and names. Non-semantic prompts were introduced to support training on datasets such as HY3D-Bench, where semantic part-name annotations are unavailable; they can also be used at inference time to decompose a shape into geometric parts without naming them.

The built-in prompts are:

| Type | Granularity | Prompt |
|---|---|---|
| Semantic | Coarse | `Please segment and name the main parts of this 3D object in <point_cloud>.` |
| Semantic | General | `Please segment and name all parts of this 3D object in <point_cloud>.` |
| Semantic | Fine | `Please segment and name all the detailed parts of this 3D object in <point_cloud>.` |
| Semantic | Numeric | `Please segment and name about N parts of this 3D object in <point_cloud>.` |
| Non-semantic | Coarse | `Please segment this 3D object into its main geometric parts in <point_cloud>.` |
| Non-semantic | General | `Please segment this 3D object into geometric parts in <point_cloud>.` |
| Non-semantic | Fine | `Please segment this 3D object into detailed geometric parts in <point_cloud>.` |
| Non-semantic | Numeric | `Please segment this 3D object into about N geometric parts in <point_cloud>.` |

`--sem_mode semantic` selects semantic prompts, `--sem_mode nosem` selects non-semantic prompts, and `--sem_mode all` selects both. Select prompts as follows:

| Selection | Prompts selected for each enabled type |
|---|---|
| `--prompt_set all` | Coarse, general, and fine. |
| `--prompt_set coarse` | Coarse only. |
| `--prompt_set general` | General only. |
| `--prompt_set fine` | Fine only. |
| `--num_parts N` | Numeric only, with `N` inserted into the prompt. This takes precedence over `--prompt_set`. |

## Text-Guided Part Segmentation

Text-guided part segmentation segments one or more parts specified by name.

```bash
bash scripts/infer.sh text_guided "$MODEL_PATH" /path/to/mesh.glb \
  outputs/text_guided --part_names "seat" "chair back" "leg"
```

| Option | Description |
|---|---|
| `--part_names NAME [NAME ...]` | Part names to segment. Quote names containing spaces. |

### Text-guided prompt

Part names are deduplicated in their input order and inserted into one prompt. For example,

```bash
--part_names "seat" "chair back" "leg"
```

produces the instruction:

```text
Please segment all the seat; chair back; leg in <point_cloud>.
```

## Interactive Segmentation

Interactive segmentation accepts a positive point in the original mesh coordinate system and predicts an initial part mask.

```bash
bash scripts/infer.sh interactive "$MODEL_PATH" /path/to/mesh.glb \
  outputs/interactive --point 0.12 0.34 -0.08
```

### Refine an existing mask

Refine mode takes an existing binary face mask and one or more corrective points:

```bash
bash scripts/infer.sh refine "$MODEL_PATH" /path/to/mesh.glb \
  outputs/refined \
  --initial_mask /path/to/initial_mask.npy \
  --point 0.18 0.31 -0.04 \
  --negative_point -0.20 0.10 0.06
```

`initial_mask.npy` is a one-dimensional NumPy array with shape `[F]`, where `F` is the number of faces in the input mesh. Values must be binary: `1` or `True` marks the foreground part and `0` or `False` marks the background. The array follows the input mesh face order.

Every interactive or refine round writes a reusable `binary_mask.npy` in its `click_N` directory. This file can be passed directly to a later refine command for the same mesh.

| Option | Description |
|---|---|
| `--initial_mask PATH` | Binary per-face `.npy` mask used by refine mode. |
| `--point X Y Z` | Positive point. May be repeated. |
| `--negative_point X Y Z` | Negative point. May be repeated in refine mode. |
| `--semantic true|false` | Enable or disable part-name prediction. Default: `true`. |

### Point coordinates and prompts

Coordinates passed to `--point` and `--negative_point` are expressed in the original mesh coordinate system. The script applies the same normalization as the point cloud and snaps each coordinate to the nearest sampled surface point.

The first positive click uses one of these prompts:

```text
Please segment and name the part at (x, y, z) in <point_cloud>.
Please segment the geometric part at (x, y, z) in <point_cloud>.
```

The second form is selected by `--semantic false`. In refine mode, the initial mask is encoded as white foreground on a gray background in the point-cloud colors. The refinement prompt lists all positive and negative coordinates and asks the model to include the positive points and exclude the negative points.

## Common Options

| Option | Description |
|---|---|
| `--max_samples N` | Process only the first `N` selected shapes. Default: all. |
| `--seed N` | Control generation and point-cloud sampling. Default: `42`. |
| `--sample_seed N` | Control sample selection independently of generation. Default: `42`. |
| `--max_new_tokens N` | Set the generation limit. Default: `10240`. |
| `--do_sample true|false` | Enable sampling instead of greedy decoding. Default: `false`. |
| `--temperature`, `--top_p`, `--top_k` | Configure sampling when `--do_sample` is enabled. |
| `--resume true|false` | Skip completed outputs or force a rerun. Default: `true`. |
| `--save_mesh true|false` | Enable or disable colored mesh export. Default: `true`. |
| `--save_per_part true|false` | Save each predicted part as a separate PLY file. Default: `false`. |

## Post-processing Options

Structured label completion and boundary graph cut are enabled by default.

| Option | Description |
|---|---|
| `--postprocess true|false` | Enable structured label completion and region cleanup. Default: `true`. |
| `--pp_drop_small true|false` | Remove and refill small isolated regions. Default: `true`. |
| `--pp_rel_area_threshold R` | Set the small-region area threshold relative to its shell. Default: `0.001`. |
| `--graph_cut true|false` | Refine part boundaries with graph cut. Default: `true`. |
