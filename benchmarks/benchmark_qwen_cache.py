"""Isolated, reproducible Qwen prefill/decode timing with real checkpoint weights.

The all-zero input fixes the Qwen shape and avoids geometry/tokenizer time. This
is a kernel-path benchmark, not a PartLLM segmentation accuracy measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jittor as jt

from partllm_jittor.hf_weights import LazySafeTensorStore
from partllm_jittor.qwen3_text import JittorQwen3Text, Qwen3TextConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if min(args.input_tokens, args.new_tokens, args.repeat) <= 0:
        parser.error("token counts and repeat must be positive")

    jt.flags.use_cuda = 1
    config = Qwen3TextConfig.from_hf_config(args.model_dir / "config.json")
    model = JittorQwen3Text(config, LazySafeTensorStore(args.model_dir, dtype="bfloat16"))
    inputs = jt.zeros((1, args.input_tokens, config.hidden_size), dtype="bfloat16")
    for run in range(args.repeat):
        start = time.perf_counter()
        with jt.no_grad():
            generated = model.generate_greedy(inputs, args.new_tokens)
        jt.sync_all(device_sync=True)
        print(json.dumps({"run": run + 1, "seconds": round(time.perf_counter() - start, 4),
                          "generated_ids": generated}), flush=True)


if __name__ == "__main__":
    main()
