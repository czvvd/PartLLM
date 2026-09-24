#!/usr/bin/env python3
"""Point-generation worker running in an isolated vLLM environment.

stdout and stderr remain available for vLLM logs. Machine-readable control
messages use only the Unix socket passed through ``--control-fd``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from partllm_vllm.ipc import IPCProtocolError, recv_frame, send_frame


@dataclass(frozen=True)
class WorkerSamplingConfig:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    max_seg_tokens: int
    stop_token_ids: tuple[int, ...]

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("eager", "stable_compiled"), default="stable_compiled"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--capture-batch-size", type=int, default=1)
    return parser.parse_args()


def _error_frame(
    exc: BaseException,
    *,
    phase: str,
    request_id: int | None = None,
) -> dict[str, Any]:
    return {
        "type": "error",
        "phase": phase,
        "request_id": request_id,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _parse_sampling(payload: Any) -> WorkerSamplingConfig:
    if not isinstance(payload, dict):
        raise IPCProtocolError("sampling must be an object")
    return WorkerSamplingConfig(
        max_tokens=int(payload["max_tokens"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        top_k=int(payload["top_k"]),
        seed=int(payload["seed"]),
        max_seg_tokens=int(payload["max_seg_tokens"]),
        stop_token_ids=tuple(int(value) for value in payload["stop_token_ids"]),
    )


def _decode_point_embeddings(message: dict[str, Any], raw: bytes):
    metadata = message.get("point_embeddings")
    if not isinstance(metadata, dict):
        raise IPCProtocolError("generate frame is missing point_embeddings metadata")
    if metadata.get("logical_dtype") != "bfloat16":
        raise IPCProtocolError("only logical_dtype=bfloat16 is supported")
    if metadata.get("storage_dtype") != "uint16":
        raise IPCProtocolError("only storage_dtype=uint16 is supported")
    if metadata.get("byteorder") != sys.byteorder:
        raise IPCProtocolError(
            f"parent and worker byte order differ: {metadata.get('byteorder')} != {sys.byteorder}"
        )
    shape_value = metadata.get("shape")
    if not isinstance(shape_value, list) or len(shape_value) != 2:
        raise IPCProtocolError("point embedding shape must be [L,H]")
    shape = tuple(int(value) for value in shape_value)
    if any(value <= 0 for value in shape):
        raise IPCProtocolError(f"invalid point embedding shape: {shape}")
    expected = shape[0] * shape[1] * 2
    if len(raw) != expected:
        raise IPCProtocolError(
            f"point embedding payload must contain {expected} bytes, got {len(raw)}"
        )

    import torch


    storage = torch.frombuffer(bytearray(raw), dtype=torch.uint16).reshape(shape)
    return storage.view(torch.bfloat16)


def _shutdown_engine(engine: Any) -> None:
    llm = getattr(engine, "llm", None)
    candidates = [
        getattr(llm, "shutdown", None),
        getattr(getattr(llm, "llm_engine", None), "shutdown", None),
    ]
    for candidate in candidates:
        if callable(candidate):
            try:
                candidate()
            except Exception:
                traceback.print_exc(file=sys.stderr)
            return


def run(args: argparse.Namespace) -> int:
    control = socket.socket(fileno=args.control_fd)
    os.set_inheritable(args.control_fd, False)
    engine: Any = None
    try:
        try:


            from partllm_vllm.hybrid import PointVLLMEngine

            engine = PointVLLMEngine(
                model_path=args.model_path,
                mode=args.mode,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                capture_batch_size=args.capture_batch_size,
            )
            send_frame(
                control,
                {
                    "type": "ready",
                    "load_seconds": float(engine.load_seconds),
                    "model_path": str(engine.model_path),
                },
            )
        except BaseException as exc:
            try:
                send_frame(control, _error_frame(exc, phase="startup"))
            except Exception:
                traceback.print_exc(file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 1

        while True:
            try:
                message, raw = recv_frame(control)
            except EOFError:

                break
            message_type = message.get("type")
            if message_type == "close":
                if raw:
                    send_frame(
                        control,
                        {
                            "type": "error",
                            "phase": "close",
                            "error_type": "IPCProtocolError",
                            "message": "close frame must not contain a raw payload",
                            "traceback": "",
                        },
                    )
                else:
                    send_frame(control, {"type": "closed"})
                break
            if message_type != "generate":
                exc = IPCProtocolError(f"unknown message type: {message_type!r}")
                try:
                    raise exc
                except IPCProtocolError as caught:
                    send_frame(control, _error_frame(caught, phase="request"))
                continue

            request_id_value = message.get("request_id")
            request_id = (
                int(request_id_value)
                if isinstance(request_id_value, int)
                and not isinstance(request_id_value, bool)
                else None
            )
            try:
                if request_id is None:
                    raise IPCProtocolError("generate frame is missing a valid request_id")
                prompt_ids = message.get("expanded_prompt_ids")
                if not isinstance(prompt_ids, list) or not prompt_ids:
                    raise IPCProtocolError("expanded_prompt_ids must be a non-empty list")
                point_embeddings = _decode_point_embeddings(message, raw)
                sampling = _parse_sampling(message.get("sampling"))
                generation = engine.generate(
                    expanded_prompt_ids=[int(value) for value in prompt_ids],
                    point_embeddings=point_embeddings,
                    sampling=sampling,
                )
                result = dataclasses.asdict(generation)
                result["output_token_ids"] = list(generation.output_token_ids)
                send_frame(
                    control,
                    {
                        "type": "generation",
                        "request_id": request_id,
                        "result": result,
                    },
                )
            except BaseException as exc:
                send_frame(
                    control,
                    _error_frame(
                        exc,
                        phase="generate",
                        request_id=request_id,
                    ),
                )
    finally:
        if engine is not None:
            _shutdown_engine(engine)
        try:
            control.close()
        except OSError:
            pass
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
