"""Cold/warm end-to-end Jittor comparison for two PartLLM checkouts.

Each checkout gets its own, initially nonexistent Jittor ``cache_name``. The
same fixed NPZ, checkpoint, seed, prompt and generation limit are used in all
runs. The script records GPU memory and timestamps from existing CLI messages;
it does not use PyTorch or change either model implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import numpy as np


WRAPPER = """
import json
from partllm_jittor.pipeline import PartSegPrompt
_original = PartSegPrompt.generate
def _timed_generate(self, *args, **kwargs):
    result = _original(self, *args, **kwargs)
    print('[Bench] generated_ids=' + json.dumps(result.generated_ids), flush=True)
    print('[Bench] prompt_generate_done', flush=True)
    return result
PartSegPrompt.generate = _timed_generate
from partllm_jittor.infer import main
main()
"""


def gpu_sample() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Expected one visible GPU, got {lines!r}")
    used, utilization = lines[0].split(",")
    return int(used.strip()), int(utilization.strip())


def measure(root: Path, cache_name: str, model_dir: Path, points_npz: Path,
            output: Path, python: Path) -> dict:
    baseline, baseline_util = gpu_sample()
    if baseline > 2048:
        raise RuntimeError(f"GPU already uses {baseline} MiB; refusing a confounded run")
    output.mkdir(parents=True, exist_ok=False)
    command = [str(python), "-u", "-c", WRAPPER, "--model_path", str(model_dir),
               "--points_npz", str(points_npz), "--task", "full_shape", "--prompt_set", "general",
               "--max_new_tokens", "64", "--seed", "0", "--output_dir", str(output),
               "--resume", "false"]
    env = os.environ.copy()
    env["cache_name"] = cache_name
    env["PYTHONPATH"] = str(root)
    events: dict[str, float] = {}
    generated_tokens = None
    generated_ids = None
    log_path = output / "run.log"
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_log() -> None:
        nonlocal generated_tokens, generated_ids
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                elapsed = time.perf_counter() - start
                log.write(line)
                log.flush()
                if "[Encode]" in line and "surface points" in line:
                    events.setdefault("encode_start", elapsed)
                elif "[Encode]" in line and "point tokens ready" in line:
                    events.setdefault("encode_end", elapsed)
                elif "[Generate] prefill" in line:
                    events.setdefault("generate_start", elapsed)
                elif "[Generate] finished with" in line:
                    events.setdefault("generate_end", elapsed)
                    match = re.search(r"finished with (\d+) tokens", line)
                    if match:
                        generated_tokens = int(match.group(1))
                elif "[Bench] prompt_generate_done" in line:
                    events.setdefault("prompt_end", elapsed)
                elif "[Bench] generated_ids=" in line:
                    generated_ids = json.loads(line.split("[Bench] generated_ids=", 1)[1])
                elif "[OK]" in line:
                    events.setdefault("output_done", elapsed)

    reader = threading.Thread(target=read_log, daemon=True)
    reader.start()
    samples = []
    try:
        while process.poll() is None:
            now = time.perf_counter() - start
            used, utilization = gpu_sample()
            samples.append((now, used, utilization))
            time.sleep(0.2)
        return_code = process.wait()
        reader.join()
    except BaseException:
        process.terminate()
        process.wait()
        reader.join()
        raise
    elapsed = time.perf_counter() - start
    if return_code:
        raise RuntimeError(f"Inference failed with {return_code}; see {log_path}")
    last_used, last_utilization = gpu_sample()
    samples.append((elapsed, last_used, last_utilization))
    labels_path = output / points_npz.stem / "general" / "point_labels.npy"
    labels = np.load(labels_path, allow_pickle=False)
    with np.load(points_npz, allow_pickle=False) as points:
        expected_points = len(points["coords"])
    if labels.shape != (expected_points,) or generated_tokens is None:
        raise RuntimeError(f"Unexpected output labels or missing token count: {labels_path}")

    def interval(left: str, right: str) -> float | None:
        if left not in events or right not in events:
            return None
        return round(events[right] - events[left], 4)

    return {
        "checkout": str(root), "cache_name": cache_name, "command": command,
        "wall_seconds": round(elapsed, 4), "generated_tokens": generated_tokens,
        "generated_ids": generated_ids,
        "tokens_per_second_end_to_end": round(generated_tokens / elapsed, 4),
        "objects_per_hour": round(3600 / elapsed, 3),
        "gpu_baseline_mib": baseline, "gpu_baseline_util_pct": baseline_util,
        "gpu_peak_increment_mib": max(sample[1] for sample in samples) - baseline,
        "gpu_mean_util_pct": round(sum(sample[2] for sample in samples) / len(samples), 2),
        "gpu_samples": len(samples), "host_load_1m_after": round(os.getloadavg()[0], 2),
        "stage_start_seconds": {key: round(value, 4) for key, value in events.items()},
        "stage_seconds": {
            "encode": interval("encode_start", "encode_end"),
            "prompt_preparation": interval("encode_end", "generate_start"),
            "qwen_greedy_generate": interval("generate_start", "generate_end"),
            "seg_bg_hidden_recompute": interval("generate_end", "prompt_end"),
            "decoder_and_export": interval("prompt_end", "output_done"),
        },
        "labels_path": str(labels_path), "response_path": str(labels_path.parent / "response.txt"),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--optimized-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--points-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--optimized-cache", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if args.baseline_cache == args.optimized_cache:
        parser.error("baseline and optimized cache names must differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results_path = args.output_dir / "measurements.json"
    results = {}
    runs = (("baseline_cold", args.baseline_root, args.baseline_cache),
            ("optimized_cold", args.optimized_root, args.optimized_cache),
            ("baseline_warm", args.baseline_root, args.baseline_cache),
            ("optimized_warm", args.optimized_root, args.optimized_cache))
    for name, root, cache_name in runs:
        print(f"[benchmark] {name}", flush=True)
        row = measure(root.resolve(), cache_name, args.model_dir.resolve(), args.points_npz.resolve(),
                      args.output_dir / name, args.python.resolve())
        results[name] = row
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run": name, "wall_seconds": row["wall_seconds"],
                          "gpu_peak_increment_mib": row["gpu_peak_increment_mib"],
                          "stage_seconds": row["stage_seconds"]}), flush=True)
    reference = np.load(results["baseline_warm"]["labels_path"], allow_pickle=False)
    for name in ("baseline_cold", "optimized_cold", "optimized_warm"):
        labels = np.load(results[name]["labels_path"], allow_pickle=False)
        results[name]["point_label_agreement_vs_baseline_warm"] = round(float((labels == reference).mean()), 8)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] saved {results_path}", flush=True)


if __name__ == "__main__":
    main()
