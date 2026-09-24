#!/usr/bin/env python3
"""PartLLM hybrid vLLM entry point without modifying test_mask_pred.py.

Execution stages:
1. PyTorch and Utonia produce point embeddings and decoder geometry features.
2. A persistent vLLM engine performs autoregressive token generation.
3. The HF language model performs one teacher-forced forward pass.
4. The original mask decoder predicts point masks from the HF hidden states.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from partllm_vllm.constants import BG_TOKEN_ID, SEG_TOKEN_ID
from partllm_vllm.contracts import SamplingConfig
from partllm_vllm.ipc import PointVLLMProcess


class FatalVLLMBackendError(BaseException):
    """Stop the outer mesh loop immediately if the vLLM backend fails."""


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--inference_mode",
        choices=("full_shape", "text_guided", "interactive", "refine"),
        default="full_shape",
    )
    parser.add_argument("--vllm_python", type=Path, required=True)
    parser.add_argument("--vllm_model_path", type=Path, required=True)
    parser.add_argument(
        "--vllm_mode",
        choices=("eager", "stable_compiled"),
        default="stable_compiled",
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.30)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--vllm_timing_output", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    return parser.parse_known_args()


def cuda_timed(callable_):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


class HybridInferenceController:
    def __init__(self, wrapper_args: argparse.Namespace, inference_module: Any) -> None:
        self.args = wrapper_args
        self.inference = inference_module
        self.case_index = 0
        self._current_geometry_ref = None
        self.timing_path = wrapper_args.vllm_timing_output.expanduser().resolve()
        self.timing_path.parent.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        print(
            "[vLLM] starting isolated point-only worker before HF CUDA model load: "
            f"mode={wrapper_args.vllm_mode}, "
            f"gpu_memory_utilization={wrapper_args.vllm_gpu_memory_utilization}, "
            f"max_model_len={wrapper_args.vllm_max_model_len}"
        )
        bridge_src = Path(__file__).resolve().parents[1] / "src"
        self.engine = PointVLLMProcess(
            python_executable=wrapper_args.vllm_python,
            model_path=wrapper_args.vllm_model_path,
            mode=wrapper_args.vllm_mode,
            gpu_memory_utilization=wrapper_args.vllm_gpu_memory_utilization,
            max_model_len=wrapper_args.vllm_max_model_len,
            capture_batch_size=1,
            env={
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(bridge_src),
                "VLLM_PLUGINS": "partllm_vllm_bridge",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        try:
            self._append_timing(
                {
                    "record_type": "engine",
                    "mode": wrapper_args.vllm_mode,
                    "model": str(self.engine.model_path),
                    "engine_load_seconds": self.engine.load_seconds,
                    "max_model_len": wrapper_args.vllm_max_model_len,
                    "gpu_memory_utilization": wrapper_args.vllm_gpu_memory_utilization,
                }
            )
            print(f"[vLLM] engine ready in {self.engine.load_seconds:.3f}s")
        except BaseException:
            self.engine.close()
            raise

    def validate_checkpoint(self, checkpoint: str | Path) -> None:
        manifest_path = self.engine.model_path / "bridge_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"vLLM model view is missing {manifest_path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = Path(str(manifest.get("checkpoint", ""))).resolve(strict=True)
        actual = Path(checkpoint).expanduser().resolve(strict=True)
        if expected != actual:
            raise ValueError(
                f"vLLM model view checkpoint={expected} does not match HF checkpoint={actual}"
            )

    def _append_timing(self, payload: dict[str, Any]) -> None:
        with self.timing_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def run_inference(self, model, inputs, utonia_point_dict, processor, args):
        from verl.models.transformers.qwen3_vl import _get_input_embeds

        total_started = time.perf_counter()
        prompt_ids = inputs["input_ids"]
        if prompt_ids.shape[0] != 1:
            raise ValueError("hybrid inference currently supports request batch size 1 only")
        prompt_len = int(prompt_ids.shape[1])



        geometry_ref = (
            utonia_point_dict
            if utonia_point_dict is not None
            else inputs.get("point_clouds")
        )
        if geometry_ref is not self._current_geometry_ref:
            self._current_geometry_ref = geometry_ref
            model.model._partllm_fps_indices_cache = {"__seed__": int(args.seed)}

        def encode_point_and_prompt():
            return _get_input_embeds(
                model.model,
                prompt_ids,
                attention_mask=inputs.get("attention_mask"),
                point_clouds=inputs.get("point_clouds"),
                utonia_point_dict=utonia_point_dict,
            )

        embedding_data, point_encode_seconds = cuda_timed(encode_point_and_prompt)
        prompt_embeddings = embedding_data["inputs_embeds"]
        point_embeddings = embedding_data["point_cloud_embeddings"]
        if point_embeddings is None:
            raise RuntimeError("Utonia/point encoder did not return point embeddings")

        stop_token_ids: set[int] = {
            int(getattr(model.config, "bg_token_id", BG_TOKEN_ID))
        }
        tokenizer = getattr(processor, "tokenizer", None)
        for value in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(model.generation_config, "eos_token_id", None),
        ):
            if isinstance(value, int) and value >= 0:
                stop_token_ids.add(int(value))
            elif isinstance(value, (list, tuple)):
                stop_token_ids.update(
                    int(item) for item in value if isinstance(item, int) and item >= 0
                )

        sampling = SamplingConfig(
            max_tokens=int(args.max_new_tokens),
            temperature=(float(args.temperature) if args.do_sample else 0.0),
            top_p=(float(args.top_p) if args.do_sample else 1.0),
            top_k=(
                int(args.top_k)
                if args.do_sample and args.top_k is not None and args.top_k > 0
                else -1
            ),
            seed=int(args.seed),
            max_seg_tokens=int(args.max_generated_parts),
            stop_token_ids=tuple(sorted(stop_token_ids)),
        )
        vllm_roundtrip_started = time.perf_counter()
        try:
            generation = self.engine.generate(
                expanded_prompt_ids=prompt_ids[0].detach().to("cpu").tolist(),
                point_embeddings=point_embeddings,
                sampling=sampling,
            )
        except Exception as exc:
            raise FatalVLLMBackendError(
                "vLLM generation failed; stopping instead of retrying every remaining sample"
            ) from exc
        vllm_roundtrip_seconds = time.perf_counter() - vllm_roundtrip_started

        generated_ids = torch.tensor(
            [
                prompt_ids[0].detach().to("cpu").tolist()
                + list(generation.output_token_ids)
            ],
            dtype=torch.long,
            device=model.device,
        )
        generated_ids_for_mask, added_bg = self.inference.ensure_bg_token_for_joint_mask(
            generated_ids,
            prompt_len,
            processor,
            model,
            args,
        )
        response = processor.decode(
            generated_ids_for_mask[0, prompt_len:], skip_special_tokens=False
        )

        generated_tail = generated_ids_for_mask[:, prompt_len:]
        language_model = model.model.language_model
        generated_embeddings = language_model.embed_tokens(generated_tail)
        full_embeddings = torch.cat([prompt_embeddings, generated_embeddings], dim=1)
        full_attention_mask = torch.ones(
            full_embeddings.shape[:2],
            dtype=torch.long,
            device=full_embeddings.device,
        )

        def teacher_forward():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return language_model(
                    input_ids=None,
                    inputs_embeds=full_embeddings,
                    attention_mask=full_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

        teacher_outputs, teacher_seconds = cuda_timed(teacher_forward)
        full_ids = generated_ids_for_mask[0]
        seg_mask = full_ids == int(getattr(model.config, "seg_token_id", SEG_TOKEN_ID))
        bg_mask = full_ids == int(getattr(model.config, "bg_token_id", BG_TOKEN_ID))
        num_parts = int(seg_mask.sum().item())

        pca_capture = None
        mask_logits = None
        mask_seconds = 0.0
        if num_parts > 0:
            hidden = teacher_outputs.last_hidden_state[0]
            seg_hidden = hidden[seg_mask]
            if not bool(bg_mask.any()):
                raise ValueError("joint mask decoding requires a BG token")
            bg_hidden = hidden[bg_mask][-1:, :]
            encoder_hidden = torch.cat([seg_hidden, bg_hidden], dim=0).unsqueeze(0)

            pc_info = embedding_data.get("pc_info")
            if not pc_info:
                raise RuntimeError("point encoder did not return pc_info/centers")
            centers = pc_info[0]
            coords_raw = embedding_data.get("utonia_coords")
            if coords_raw is None:
                raise RuntimeError("Utonia did not return original point coordinates")
            coords = coords_raw.unsqueeze(0)
            feat_raw_value = embedding_data.get("utonia_feat_raw")
            feat_raw = feat_raw_value.unsqueeze(0) if feat_raw_value is not None else None

            pca_capture = (
                self.inference._PCAFeatureCapture.attach(model)
                if self.inference._pca_enabled()
                else None
            )

            def decode_mask():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return model.model.mask_decoder(
                        hidden_states=point_embeddings.unsqueeze(0),
                        encoder_hidden_states=encoder_hidden,
                        centers=centers,
                        coords=coords,
                        feat_raw=feat_raw,
                    )

            try:
                mask_logits, mask_seconds = cuda_timed(decode_mask)
            finally:
                if pca_capture is not None:
                    pca_capture.detach()

            expected_classes = num_parts + 1
            if int(mask_logits.shape[1]) != expected_classes:
                raise RuntimeError(
                    f"mask decoder classes={mask_logits.shape[1]}, expected {expected_classes}"
                )

        total_seconds = time.perf_counter() - total_started
        timing = {
            "record_type": "request",
            "case_index": self.case_index,
            "mode": self.args.vllm_mode,
            "prompt_tokens": prompt_len,
            "generated_tokens": int(generated_tail.shape[1]),
            "seg_tokens": num_parts,
            "bg_tokens": int(bg_mask.sum().item()),
            "auto_added_bg": bool(added_bg),
            "hit_seg_limit": bool(generation.hit_seg_limit),
            "vllm_prompt_match": bool(generation.prompt_match),
            "max_tokens_used": generation.max_tokens_used,
            "point_encode_seconds": point_encode_seconds,
            "vllm_warmup_seconds": generation.warmup_seconds,
            "vllm_generate_seconds": generation.generate_seconds,
            "vllm_roundtrip_seconds": vllm_roundtrip_seconds,
            "hf_teacher_seconds": teacher_seconds,
            "mask_decoder_seconds": mask_seconds,
            "hybrid_total_seconds": total_seconds,
        }
        self._append_timing(timing)
        self.case_index += 1
        print(
            "[vLLM] "
            f"parts={num_parts} tokens={generated_tail.shape[1]} "
            f"point={point_encode_seconds:.3f}s "
            f"generate={generation.generate_seconds:.3f}s "
            f"roundtrip={vllm_roundtrip_seconds:.3f}s "
            f"teacher={teacher_seconds:.3f}s "
            f"mask={mask_seconds:.3f}s total={total_seconds:.3f}s"
        )

        del (
            teacher_outputs,
            full_embeddings,
            generated_embeddings,
            prompt_embeddings,
            embedding_data,
            generated_ids,
            generated_ids_for_mask,
        )
        return response, mask_logits, pca_capture

    def close(self) -> None:
        self.engine.close()


def main() -> None:
    wrapper_args, inference_args = parse_wrapper_args()
    repo_root = wrapper_args.repo_root.expanduser().resolve(strict=True)
    bridge_root = Path(__file__).resolve().parents[1]
    inference_dir = repo_root / "recipe" / "part_seg" / "tools" / "inference"
    sys.path.insert(0, str(bridge_root / "src"))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(inference_dir))

    os.environ.setdefault("VLLM_PLUGINS", "partllm_vllm_bridge")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    inference_core = importlib.import_module("test_mask_pred")
    controller = HybridInferenceController(wrapper_args, inference_core)
    try:
        original_load_model = inference_core.load_model

        def load_model_with_checkpoint_gate(args):
            controller.validate_checkpoint(args.model_path)
            return original_load_model(args)

        inference_core.load_model = load_model_with_checkpoint_gate
        inference_core.run_inference = controller.run_inference

        entrypoints = {
            "full_shape": ("test_mask_pred", "test_mask_pred.py"),
            "text_guided": ("infer_grounding", "infer_grounding.py"),
            "interactive": ("infer_promptable", "infer_promptable.py"),
            "refine": ("infer_promptable", "infer_promptable.py"),
        }
        module_name, filename = entrypoints[wrapper_args.inference_mode]
        inference_entry = (
            inference_core
            if module_name == "test_mask_pred"
            else importlib.import_module(module_name)
        )
        sys.argv = [str(inference_dir / filename), *inference_args]
        inference_entry.main()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
