"""Filter a mixed PartLLM checkpoint into weights supported by vLLM Qwen3-VL."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

SUPPORTED_WEIGHT_PREFIXES = (
    "model.language_model.",
    "lm_head.",
)

SKIPPED_WEIGHT_PREFIXES = (
    "model.mask_decoder.",
    "model.utonia_model.",
    "model.utonia_proj.",
)


def is_supported_weight(name: str) -> bool:
    return name.startswith(SUPPORTED_WEIGHT_PREFIXES)


def iter_supported_weights(weights: Iterable[tuple[str, T]]) -> Iterator[tuple[str, T]]:
    for name, tensor in weights:
        if is_supported_weight(name):
            yield name, tensor
