"""Runtime bridge between PartLLM PyTorch inference and vLLM generation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .constants import DEFAULT_STOP_TOKEN_IDS, NUM_POINT_TOKENS, POINT_TOKEN_ID
from .contracts import SamplingConfig, preserve_stop_token, truncate_on_max_seg


def collapse_point_prompt(
    prompt_token_ids: Sequence[int],
    *,
    point_token_id: int = POINT_TOKEN_ID,
    expected_point_tokens: int = NUM_POINT_TOKENS,
) -> tuple[list[int], list[int]]:
    """Collapse contiguous point tokens into one vLLM multimodal placeholder."""

    expanded = [int(token_id) for token_id in prompt_token_ids]
    point_positions = [
        index for index, token_id in enumerate(expanded) if token_id == point_token_id
    ]
    if len(point_positions) != expected_point_tokens:
        raise ValueError(
            f"expected {expected_point_tokens} point tokens, got {len(point_positions)}"
        )
    expected_positions = list(
        range(point_positions[0], point_positions[0] + expected_point_tokens)
    )
    if point_positions != expected_positions:
        raise ValueError("point tokens must form one contiguous prompt span")
    collapsed = (
        expanded[: point_positions[0]]
        + [point_token_id]
        + expanded[point_positions[-1] + 1 :]
    )
    return collapsed, point_positions


@dataclass(frozen=True)
class VLLMGeneration:
    output_token_ids: tuple[int, ...]
    finish_reason: str | None
    stop_reason: int | str | None
    prompt_match: bool
    generate_seconds: float
    warmup_seconds: float
    hit_seg_limit: bool
    max_tokens_used: int


class PointVLLMEngine:
    """Persistent point-only vLLM engine for LLM token generation."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        mode: str = "stable_compiled",
        gpu_memory_utilization: float = 0.30,
        max_model_len: int = 32768,
        capture_batch_size: int = 1,
    ) -> None:
        if mode not in {"eager", "stable_compiled"}:
            raise ValueError(f"unsupported vLLM mode: {mode}")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        self.mode = mode
        self.max_model_len = int(max_model_len)
        self._warmed = False

        os.environ.setdefault("VLLM_PLUGINS", "partllm_vllm_bridge")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if mode == "stable_compiled":
            from .runtime import stable_compile_execution_kwargs

            execution_kwargs = stable_compile_execution_kwargs(
                capture_batch_size=capture_batch_size
            )
        else:
            execution_kwargs = {"enforce_eager": True}

        load_started = time.perf_counter()
        self.llm = LLM(
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            trust_remote_code=True,
            dtype="bfloat16",
            enable_mm_embeds=True,
            limit_mm_per_prompt={"point": 1, "image": 0, "video": 0},
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=capture_batch_size,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            mm_processor_cache_gb=0,
            disable_log_stats=True,
            **execution_kwargs,
        )
        self.load_seconds = time.perf_counter() - load_started

    def _build_request(
        self,
        expanded_prompt_ids: Sequence[int],
        point_embeddings: torch.Tensor,
    ) -> tuple[dict[str, Any], list[int]]:
        collapsed_ids, _ = collapse_point_prompt(expanded_prompt_ids)
        prompt = self.tokenizer.decode(
            collapsed_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        roundtrip_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if roundtrip_ids != collapsed_ids:
            raise RuntimeError("collapsed point prompt does not round-trip through the tokenizer")
        if point_embeddings.ndim != 2:
            raise ValueError(
                f"point_embeddings must have shape [L,H], got {tuple(point_embeddings.shape)}"
            )
        if point_embeddings.shape[0] != NUM_POINT_TOKENS:
            raise ValueError(
                f"expected {NUM_POINT_TOKENS} point embeddings, got {point_embeddings.shape[0]}"
            )
        request = {
            "prompt": prompt,
            "multi_modal_data": {
                "point": point_embeddings.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            },
        }
        return request, [int(value) for value in expanded_prompt_ids]

    def generate(
        self,
        *,
        expanded_prompt_ids: Sequence[int],
        point_embeddings: torch.Tensor,
        sampling: SamplingConfig,
    ) -> VLLMGeneration:
        from vllm import SamplingParams

        request, expected_prompt_ids = self._build_request(
            expanded_prompt_ids, point_embeddings
        )
        remaining_context = self.max_model_len - len(expected_prompt_ids)
        if remaining_context <= 0:
            raise ValueError(
                f"prompt length {len(expected_prompt_ids)} reaches max_model_len={self.max_model_len}"
            )
        max_tokens = int(sampling.max_tokens)
        if max_tokens <= 0:
            raise ValueError("maximum generation tokens must be positive")
        if max_tokens > remaining_context:
            raise ValueError(
                f"prompt={len(expected_prompt_ids)} + max_tokens={max_tokens} exceeds "
                f"vLLM max_model_len={self.max_model_len}; refusing silent truncation"
            )

        warmup_seconds = 0.0
        if not self._warmed:
            warmup_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                ignore_eos=True,
                seed=sampling.seed,
                detokenize=False,
            )
            warmup_started = time.perf_counter()
            self.llm.generate([request], warmup_params, use_tqdm=False)
            warmup_seconds = time.perf_counter() - warmup_started
            self._warmed = True

        sampling_kwargs: dict[str, Any] = {
            "temperature": sampling.temperature if not sampling.greedy else 0.0,
            "max_tokens": max_tokens,
            "stop_token_ids": [int(value) for value in sampling.stop_token_ids],
            "seed": sampling.seed,
            "detokenize": False,
        }
        if not sampling.greedy:
            sampling_kwargs["top_p"] = sampling.top_p
            sampling_kwargs["top_k"] = sampling.top_k
        params = SamplingParams(**sampling_kwargs)

        started = time.perf_counter()
        request_output = self.llm.generate([request], params, use_tqdm=False)[0]
        generate_seconds = time.perf_counter() - started
        candidate = request_output.outputs[0]
        output_ids = preserve_stop_token(
            candidate.token_ids,
            candidate.stop_reason,
            sampling.stop_token_ids,
        )
        output_ids, hit_seg_limit = truncate_on_max_seg(
            output_ids,
            max_seg_tokens=sampling.max_seg_tokens,
        )
        actual_prompt_ids = (
            [int(value) for value in request_output.prompt_token_ids]
            if request_output.prompt_token_ids is not None
            else None
        )
        prompt_match = actual_prompt_ids == expected_prompt_ids
        if not prompt_match:
            raise RuntimeError("vLLM expanded prompt token IDs do not match the HF prompt")
        return VLLMGeneration(
            output_token_ids=output_ids,
            finish_reason=(
                None if candidate.finish_reason is None else str(candidate.finish_reason)
            ),
            stop_reason=candidate.stop_reason,
            prompt_match=prompt_match,
            generate_seconds=generate_seconds,
            warmup_seconds=warmup_seconds,
            hit_seg_limit=hit_seg_limit,
            max_tokens_used=max_tokens,
        )
