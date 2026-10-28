"""Measure the released checkpoint's point encoder and prediction separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jittor as jt

from partllm_jittor.geometry import load_points_npz
from partllm_jittor.infer import Inference, build_parser
from partllm_jittor.prompts import SEMANTIC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--points-npz", type=Path, required=True)
    args = parser.parse_args()
    inference_args = build_parser().parse_args([
        "--model_path", str(args.model_dir), "--points_npz", str(args.points_npz),
        "--output_dir", "/tmp/partllm_jittor_stage_benchmark", "--max_new_tokens", "64",
        "--prompt_set", "general", "--resume", "false",
    ])
    runtime = Inference(inference_args)
    coords, colors, normals = load_points_npz(args.points_npz)
    with jt.no_grad():
        start = time.perf_counter()
        encoded = runtime.encode(coords, colors, normals)
        encoded_at = time.perf_counter()
        text, logits = runtime.predict(encoded, SEMANTIC["general"])
        jt.sync_all(device_sync=True)
        predicted_at = time.perf_counter()
    print(json.dumps({"encode_seconds": round(encoded_at - start, 4),
                      "predict_seconds": round(predicted_at - encoded_at, 4),
                      "logits_shape": list(logits.shape), "response": text}), flush=True)


if __name__ == "__main__":
    main()
