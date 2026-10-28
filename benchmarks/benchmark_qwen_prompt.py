"""Time the released PartLLM prompt path using saved point embeddings.

The supplied NPZ must have ``hidden_states`` shaped [1, point_tokens, hidden_size].
It excludes Utonia and mask decoding, but includes the current second full
Qwen pass used to recover SEG/BG hidden states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jittor as jt
import numpy as np

from partllm_jittor.hf_weights import LazySafeTensorStore
from partllm_jittor.pipeline import PartSegPrompt
from partllm_jittor.qwen3_text import JittorQwen3Text, Qwen3TextConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    with np.load(args.embeddings_npz, allow_pickle=False) as archive:
        point_embeddings = np.asarray(archive["hidden_states"], dtype=np.float32)[0]
    prompt = PartSegPrompt(args.model_dir, point_token_count=len(point_embeddings))
    model = JittorQwen3Text(
        Qwen3TextConfig.from_hf_config(args.model_dir / "config.json"),
        LazySafeTensorStore(args.model_dir, dtype="bfloat16"),
    )
    text = "Please segment and name all parts of this 3D object in <point_cloud>."
    with jt.no_grad():
        start = time.perf_counter()
        input_ids = prompt.input_ids(text)
        inputs_embeds = prompt.inject_point_embeddings(model, input_ids, point_embeddings)
        jt.sync([inputs_embeds])
        prepared = time.perf_counter()
        generated = model.generate_greedy(
            inputs_embeds, args.new_tokens, prompt.tokenizer.eos_token_id,
            stop_token_ids={prompt.bg_token_id}, seg_token_id=prompt.seg_token_id,
        )
        jt.sync_all(device_sync=True)
        generated_at = time.perf_counter()
        if prompt.bg_token_id not in generated:
            generated.append(prompt.bg_token_id)
        full_ids = np.concatenate([input_ids, np.asarray(generated, dtype=np.int32)[None, :]], axis=1)
        final_embeddings = prompt.inject_point_embeddings(model, full_ids, point_embeddings)
        hidden, _ = model.forward(final_embeddings)
        token_positions = input_ids.shape[1] + np.flatnonzero(np.isin(generated, [prompt.seg_token_id, prompt.bg_token_id]))
        token_hidden = hidden.float32().numpy()[:, token_positions, :]
        completed = time.perf_counter()
    print(json.dumps({"prepare_seconds": round(prepared - start, 4),
                      "generate_seconds": round(generated_at - prepared, 4),
                      "second_forward_seconds": round(completed - generated_at, 4),
                      "generated_ids": generated, "token_hidden_shape": list(token_hidden.shape)}), flush=True)


if __name__ == "__main__":
    main()
