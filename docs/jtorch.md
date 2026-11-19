# PartLLM inference with JTorch

The JTorch launcher uses the existing PartLLM inference entry points for full-shape, text-guided, and interactive segmentation. It expects a working CUDA-enabled Jittor 2.0 Torch frontend in a Python environment with PartLLM and its patched Transformers 4.57.3 installed. The benchmark used an experimental Jittor 2.0 checkout at `f7f5fd2c1` and a local PartLLM checkpoint.

    PARTLLM_USE_JTORCH=1 bash scripts/infer.sh text_guided \
      "$MODEL_PATH" path/to/chair.glb outputs/jtorch --part_names seat "chair back" leg

The launcher adds `compat/jtorch` to the JTorch process's `PYTHONPATH`. Its `torch_cluster.fps` shim handles one FP32 point cloud with a scalar sampling ratio. Its `torch_scatter.segment_csr` shim takes a two-dimensional FP32 or BF16 tensor and one valid, nondecreasing, same-device int64 CSR pointer array. The pointer must start at zero and end at the input row count. The supported reductions are `sum`, `add`, `mean`, `min`, and `max`. Both implementations have CPU and CUDA paths. Their original license notices are included alongside the shims.

| Optional flag | Behavior and requirements |
| --- | --- |
| `PARTLLM_DETERMINISTIC_FPS_CACHE=1` | Reuse fixed-start FPS within a single request when the geometry's storage versions match. |
| `PARTLLM_SINGLE_TOKEN_OPT=1` | Skip the packed-sequence check for a single CUDA position. The installed Transformers function must match the validated 4.57.3 implementation. |
| `PARTLLM_FAST_DECODER_KNN=1` | Use a matrix multiplication formula for the validated FP32 CUDA, no-gradient, three-neighbor search. Other inputs fall back to the existing search. |
| `PARTLLM_JTORCH_CPU_LOAD=2` | Stage checkpoint weights on CPU and enable the compatible metadata and weight-loading shortcuts. Requires a JTorch Safetensors frontend with BF16 support. |
| `PARTLLM_JTORCH_DEFER_LINEAR_INIT=1` | Enable deferred `Linear` initialization. Requires the runtime's `jittor.compat.torch.checkpoint_linear` implementation. |
| `PARTLLM_PARALLEL_CONSTRUCTOR_NORMAL=1` | Enable the validated CPU draw for this checkpoint's large embedding. Requires CPU staging and deferred `Linear` initialization. |

Prepared CPU initialization draws are optional. Set `PARTLLM_PREPARED_NORMAL_DIR` to an existing directory reserved for this run. Use `PARTLLM_PREPARED_NORMAL_MODE=prepare` on a first run and `reuse` on later runs with the same runtime, checkpoint, and initial seed. Cache keys include the runtime's cache identity, the RNG state, shape, and dtype. Missing or invalid entries are regenerated during initialization. The benchmark prepared this cache before timing. A generic Jittor installation was not tested for the full benchmark profile.

For comparison with PyTorch, use a separate environment containing the original `torch-cluster` and `torch-scatter` extensions and run the same command with `USE_VLLM=0`. The benchmarks used one RTX 4090 and tested inference only.
