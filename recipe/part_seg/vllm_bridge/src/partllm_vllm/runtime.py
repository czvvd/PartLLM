"""Numerically validated PartLLM execution settings for vLLM 0.11.2."""

from __future__ import annotations

from typing import Any


def stable_compile_execution_kwargs(
    *, capture_batch_size: int = 1
) -> dict[str, Any]:
    """Return ``vllm.LLM`` arguments for stable and accelerated decoding.

    Keeping all vLLM custom operations opaque prevents Inductor from changing
    greedy tokens in BF16 near-ties. CUDA Graph covers decoding only and uses
    the actual request batch size.
    """

    if capture_batch_size <= 0:
        raise ValueError("capture_batch_size must be positive")

    from vllm.config.compilation import (
        CUDAGraphMode,
        CompilationConfig,
        CompilationMode,
    )

    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="inductor",
        custom_ops=["all"],
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        compile_sizes=[],
        cudagraph_capture_sizes=[capture_batch_size],
        max_cudagraph_capture_size=capture_batch_size,
    )
    return {
        "enforce_eager": False,
        "compilation_config": compilation_config,
    }


def stable_compile_profile(*, capture_batch_size: int = 1) -> dict[str, Any]:
    """Return a serializable description for diagnostics and logs."""

    if capture_batch_size <= 0:
        raise ValueError("capture_batch_size must be positive")
    return {
        "enforce_eager": False,
        "compilation_mode": "VLLM_COMPILE",
        "backend": "inductor",
        "custom_ops": ["all"],
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "compile_sizes": [],
        "cudagraph_capture_sizes": [capture_batch_size],
        "max_cudagraph_capture_size": capture_batch_size,
    }
