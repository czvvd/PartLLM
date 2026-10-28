# Inference benchmarks

Run these scripts with the Jittor environment on an otherwise idle NVIDIA GPU.
They never import PyTorch. Model checkpoints, input NPZ files, and benchmark
outputs should remain outside version control (`artifacts/` is ignored).

`benchmark_qwen_cache.py` times Qwen prefill and greedy decoding with fixed
zero embeddings. `benchmark_qwen_prompt.py` includes the real prompt and
SEG/BG hidden-state recomputation using saved point embeddings.
`benchmark_inference_stages.py` times point encoding and prediction.
Run the individual scripts from the repository root with module syntax, for
example `python -m benchmarks.benchmark_qwen_cache --help`.

For a paired end-to-end cold/warm comparison, keep an unmodified checkout of
the original `jittor` commit (`908bf4a`) alongside the optimized checkout.
Use two new Jittor cache names that have never been used before:

```bash
python -m benchmarks.compare_end_to_end \
  --baseline-root /path/to/original-checkout \
  --optimized-root /path/to/optimized-checkout \
  --model-dir /path/to/checkpoint \
  --points-npz /path/to/fixed-input.npz \
  --output-dir /path/to/new-results-directory \
  --baseline-cache unique_baseline_cache_name \
  --optimized-cache unique_optimized_cache_name
```

The script runs baseline cold, optimized cold, baseline warm, and optimized
warm serially. Cold runs include Jittor compilation; warm runs reuse only their
own checkout's compiled cache. It records wall time, stage timings, generated
tokens, GPU peak-memory increment, and point-label agreement in
`measurements.json`. The GPU memory sample is process-external and has a 0.2 s
sampling interval; short-lived peaks may be missed. One sample is not enough
to establish a general performance advantage over another framework.
