"""Explicit contracts between the PyTorch point path and vLLM language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .constants import (
    BG_TOKEN_ID,
    DEFAULT_STOP_TOKEN_IDS,
    HIDDEN_SIZE,
    NUM_POINT_TOKENS,
    POINT_TOKEN_ID,
    SEG_TOKEN_ID,
)


@dataclass(frozen=True)
class EmbeddingOverride:
    """Point tokens whose standard token embeddings must be replaced."""

    positions: torch.Tensor
    values: torch.Tensor

    def validate(self, prompt_length: int) -> None:
        if self.positions.dtype != torch.long or self.positions.ndim != 1:
            raise ValueError("positions must be a one-dimensional torch.long tensor")
        if tuple(self.values.shape) != (self.positions.numel(), HIDDEN_SIZE):
            raise ValueError(
                f"values must have shape [K,{HIDDEN_SIZE}], got {tuple(self.values.shape)}"
            )
        if self.positions.numel() != NUM_POINT_TOKENS:
            raise ValueError(
                f"expected {NUM_POINT_TOKENS} point tokens, got {self.positions.numel()}"
            )
        if self.positions.numel() and (
            int(self.positions.min()) < 0 or int(self.positions.max()) >= prompt_length
        ):
            raise ValueError("embedding override positions exceed the prompt length")


@dataclass(frozen=True)
class LLMRequest:
    """An LLM request that preserves token IDs for scheduling, RoPE, and stopping."""

    prompt_token_ids: tuple[int, ...]
    embedding_override: EmbeddingOverride
    seg_token_id: int = SEG_TOKEN_ID
    bg_token_id: int = BG_TOKEN_ID
    stop_token_ids: tuple[int, ...] = DEFAULT_STOP_TOKEN_IDS

    def validate(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        self.embedding_override.validate(len(self.prompt_token_ids))
        point_positions = tuple(
            index for index, token_id in enumerate(self.prompt_token_ids) if token_id == POINT_TOKEN_ID
        )
        if point_positions != tuple(int(value) for value in self.embedding_override.positions.tolist()):
            raise ValueError("point token positions do not match embedding overrides")


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 25240
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 66
    max_seg_tokens: int = 64
    stop_token_ids: tuple[int, ...] = DEFAULT_STOP_TOKEN_IDS

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass(frozen=True)
class GenerationResult:
    output_token_ids: tuple[int, ...]
    finish_reason: str | None
    stop_reason: int | str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class SelectedHidden:
    full_token_ids: tuple[int, ...]
    seg_positions: tuple[int, ...]
    bg_position: int
    seg_hidden: torch.Tensor
    bg_hidden: torch.Tensor

    def validate(self) -> None:
        if tuple(self.seg_hidden.shape) != (len(self.seg_positions), HIDDEN_SIZE):
            raise ValueError("SEG hidden-state shape mismatch")
        if tuple(self.bg_hidden.shape) != (1, HIDDEN_SIZE):
            raise ValueError("BG hidden-state shape mismatch")
        if self.full_token_ids[self.bg_position] != BG_TOKEN_ID:
            raise ValueError("bg_position does not point to a BG token")


def preserve_stop_token(
    output_token_ids: Sequence[int],
    stop_reason: int | str | None,
    stop_token_ids: Sequence[int] = DEFAULT_STOP_TOKEN_IDS,
) -> tuple[int, ...]:
    """Restore a stop token when the active vLLM version omits it from token IDs."""

    result = tuple(int(token_id) for token_id in output_token_ids)
    if isinstance(stop_reason, int) and stop_reason in {
        int(token_id) for token_id in stop_token_ids
    }:
        if not result or result[-1] != stop_reason:
            result += (stop_reason,)
    return result


def truncate_on_max_seg(
    output_token_ids: Sequence[int], seg_token_id: int = SEG_TOKEN_ID, max_seg_tokens: int = 64
) -> tuple[tuple[int, ...], bool]:
    """Truncate at the configured SEG-token limit."""

    if max_seg_tokens <= 0:
        return tuple(int(token_id) for token_id in output_token_ids), False
    count = 0
    for index, token_id in enumerate(output_token_ids):
        if int(token_id) == seg_token_id:
            count += 1
            if count >= max_seg_tokens:
                return tuple(int(value) for value in output_token_ids[: index + 1]), True
    return tuple(int(token_id) for token_id in output_token_ids), False


def ensure_joint_bg(
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    seg_token_id: int = SEG_TOKEN_ID,
    bg_token_id: int = BG_TOKEN_ID,
    end_token_ids: Sequence[int] = DEFAULT_STOP_TOKEN_IDS[:-1],
) -> tuple[tuple[int, ...], bool]:
    """Apply the inference auto-add-BG behavior and return the full sequence."""

    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    output = tuple(int(token_id) for token_id in output_token_ids)
    if seg_token_id not in output or bg_token_id in output:
        return prompt + output, False

    insert_at = len(output)
    end_ids = set(int(token_id) for token_id in end_token_ids)
    for index, token_id in enumerate(output):
        if token_id in end_ids:
            insert_at = index
            break
    fixed_output = output[:insert_at] + (bg_token_id,) + output[insert_at:]
    return prompt + fixed_output, True
