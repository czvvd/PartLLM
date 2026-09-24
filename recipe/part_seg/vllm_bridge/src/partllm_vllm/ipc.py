"""Binary IPC between the HF parent and the isolated vLLM environment.

Control messages use length-prefixed JSON. Point embeddings follow as raw
BF16 bits in uint16 storage. The protocol uses neither pickle nor torch
objects across the two Python environments.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from .constants import DEFAULT_STOP_TOKEN_IDS

if TYPE_CHECKING:
    from .hybrid import VLLMGeneration


_JSON_LENGTH = struct.Struct("!Q")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_RAW_BYTES = 2 * 1024 * 1024 * 1024


class IPCProtocolError(RuntimeError):
    """An IPC frame is incomplete or violates the protocol."""


class RemoteVLLMError(RuntimeError):
    """A remote vLLM worker error with its complete traceback."""

    def __init__(
        self,
        *,
        message: str,
        error_type: str | None = None,
        remote_traceback: str | None = None,
        phase: str | None = None,
        request_id: int | None = None,
    ) -> None:
        self.message = message
        self.error_type = error_type
        self.remote_traceback = remote_traceback
        self.phase = phase
        self.request_id = request_id
        prefix = "vLLM worker error"
        if phase:
            prefix += f" ({phase})"
        if error_type:
            prefix += f" [{error_type}]"
        detail = f"{prefix}: {message}"
        if remote_traceback:
            detail += f"\n--- remote traceback ---\n{remote_traceback.rstrip()}"
        super().__init__(detail)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise IPCProtocolError(f"invalid read length: {size}")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            received = size - remaining
            raise EOFError(
                f"IPC socket closed early: expected {size} bytes, received {received}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket,
    message: Mapping[str, Any],
    raw_payload: bytes | bytearray | memoryview = b"",
) -> None:
    """Send an 8-byte JSON length, JSON body, and raw payload."""

    raw_view = memoryview(raw_payload).cast("B")
    if len(raw_view) > _MAX_RAW_BYTES:
        raise IPCProtocolError(f"raw payload is too large: {len(raw_view)} bytes")
    body = dict(message)
    declared = body.get("raw_nbytes")
    if declared is not None and int(declared) != len(raw_view):
        raise IPCProtocolError(
            f"raw_nbytes={declared} does not match payload={len(raw_view)}"
        )
    body["raw_nbytes"] = len(raw_view)
    try:
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IPCProtocolError(f"JSON serialization failed: {exc}") from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise IPCProtocolError(f"JSON frame is too large: {len(encoded)} bytes")
    sock.sendall(_JSON_LENGTH.pack(len(encoded)))
    sock.sendall(encoded)
    if raw_view:
        sock.sendall(raw_view)


def recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    """Receive one frame and validate JSON and raw payload lengths."""

    (json_size,) = _JSON_LENGTH.unpack(_recv_exact(sock, _JSON_LENGTH.size))
    if json_size > _MAX_JSON_BYTES:
        raise IPCProtocolError(f"JSON frame is too large: {json_size} bytes")
    encoded = _recv_exact(sock, json_size)
    try:
        message = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IPCProtocolError(f"failed to parse JSON frame: {exc}") from exc
    if not isinstance(message, dict):
        raise IPCProtocolError("JSON frame root must be an object")
    raw_size = message.get("raw_nbytes", 0)
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise IPCProtocolError(f"invalid raw_nbytes: {raw_size!r}")
    if raw_size > _MAX_RAW_BYTES:
        raise IPCProtocolError(f"raw payload is too large: {raw_size} bytes")
    return message, _recv_exact(sock, raw_size)


def _sampling_payload(sampling: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "max_tokens": 25240,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 66,
        "max_seg_tokens": 64,
        "stop_token_ids": DEFAULT_STOP_TOKEN_IDS,
    }
    if isinstance(sampling, Mapping):
        source = sampling
        values = {key: source.get(key, default) for key, default in defaults.items()}
    else:
        values = {
            key: getattr(sampling, key, default) for key, default in defaults.items()
        }
    payload = {
        "max_tokens": int(values["max_tokens"]),
        "temperature": float(values["temperature"]),
        "top_p": float(values["top_p"]),
        "top_k": int(values["top_k"]),
        "seed": int(values["seed"]),
        "max_seg_tokens": int(values["max_seg_tokens"]),
        "stop_token_ids": [int(value) for value in values["stop_token_ids"]],
    }
    if payload["max_tokens"] <= 0:
        raise ValueError("max_tokens must be positive")
    return payload


def _tensor_to_bf16_uint16(point_embeddings: Any) -> tuple[tuple[int, int], bytes]:
    """Convert to a CPU BF16 byte stream without serializing torch objects."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - inference requires torch
        raise RuntimeError("generate() requires PyTorch in the parent environment") from exc
    if not isinstance(point_embeddings, torch.Tensor):
        raise TypeError("point_embeddings must be a torch.Tensor")
    tensor = (
        point_embeddings.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
    )
    if tensor.ndim != 2:
        raise ValueError(
            f"point_embeddings must have shape [L,H], got {tuple(tensor.shape)}"
        )

    raw = tensor.view(torch.int16).numpy().tobytes(order="C")
    return (int(tensor.shape[0]), int(tensor.shape[1])), raw


def _remote_error(message: Mapping[str, Any]) -> RemoteVLLMError:
    request_id = message.get("request_id")
    return RemoteVLLMError(
        message=str(message.get("message", "unknown remote error")),
        error_type=(
            None if message.get("error_type") is None else str(message["error_type"])
        ),
        remote_traceback=(
            None if message.get("traceback") is None else str(message["traceback"])
        ),
        phase=None if message.get("phase") is None else str(message["phase"]),
        request_id=(
            int(request_id) if isinstance(request_id, int) and not isinstance(request_id, bool) else None
        ),
    )


def _signal_process_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError, PermissionError):
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _force_cleanup(
    control_socket: socket.socket | None,
    process: subprocess.Popen[Any] | None,
    timeout: float = 5.0,
) -> None:
    if control_socket is not None:
        try:
            control_socket.close()
        except OSError:
            pass
    if process is None or process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(process, signal.SIGKILL)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:

        pass


class PointVLLMProcess:
    """Launch a vLLM worker while preserving the PointVLLMEngine interface."""

    def __init__(
        self,
        *,
        python_executable: str | Path,
        model_path: str | Path,
        worker_path: str | Path | None = None,
        mode: str = "stable_compiled",
        gpu_memory_utilization: float = 0.30,
        max_model_len: int = 32768,
        capture_batch_size: int = 1,
        startup_timeout: float = 1800.0,
        request_timeout: float | None = 3600.0,
        close_timeout: float = 15.0,
        env: Mapping[str, str] | None = None,
        worker_stdout: int | BinaryIO | None = None,
        worker_stderr: int | BinaryIO | None = None,
    ) -> None:
        if mode not in {"eager", "stable_compiled"}:
            raise ValueError(f"unsupported vLLM mode: {mode}")
        if startup_timeout <= 0 or close_timeout <= 0:
            raise ValueError("startup_timeout and close_timeout must be positive")
        if request_timeout is not None and request_timeout <= 0:
            raise ValueError("request_timeout must be positive or None")



        self.python_executable = Path(
            os.path.abspath(os.path.expanduser(str(python_executable)))
        )
        if not self.python_executable.is_file() or not os.access(
            self.python_executable, os.X_OK
        ):
            raise FileNotFoundError(
                f"vLLM Python does not exist or is not executable: {self.python_executable}"
            )
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        default_worker = Path(__file__).resolve().parents[2] / "tools" / "vllm_worker.py"
        self.worker_path = Path(worker_path or default_worker).expanduser().resolve(strict=True)
        self.mode = mode
        self.request_timeout = request_timeout
        self.close_timeout = float(close_timeout)
        self._lock = threading.Lock()
        self._next_request_id = 1
        self._closed = False
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen[Any] | None = None
        self._finalizer: weakref.finalize | None = None

        parent_socket, child_socket = socket.socketpair()
        process: subprocess.Popen[Any] | None = None
        try:
            child_fd = child_socket.fileno()
            command = [
                str(self.python_executable),
                str(self.worker_path),
                "--control-fd",
                str(child_fd),
                "--model-path",
                str(self.model_path),
                "--mode",
                mode,
                "--gpu-memory-utilization",
                str(float(gpu_memory_utilization)),
                "--max-model-len",
                str(int(max_model_len)),
                "--capture-batch-size",
                str(int(capture_batch_size)),
            ]
            worker_env = os.environ.copy()
            if env is not None:
                worker_env.update({str(key): str(value) for key, value in env.items()})
            worker_env.setdefault("PYTHONUNBUFFERED", "1")
            process = subprocess.Popen(
                command,
                pass_fds=(child_fd,),
                stdin=subprocess.DEVNULL,
                stdout=worker_stdout,
                stderr=worker_stderr,
                env=worker_env,
                close_fds=True,
                start_new_session=True,
            )
            child_socket.close()
            parent_socket.settimeout(float(startup_timeout))
            message, raw = recv_frame(parent_socket)
            if raw:
                raise IPCProtocolError("ready frame must not contain a raw payload")
            if message.get("type") == "error":
                raise _remote_error(message)
            if message.get("type") != "ready":
                raise IPCProtocolError(
                    f"expected a ready frame, received {message.get('type')!r}"
                )
            ready_model = Path(str(message.get("model_path", ""))).expanduser().resolve()
            if ready_model != self.model_path:
                raise IPCProtocolError(
                    f"worker model_path={ready_model} does not match requested={self.model_path}"
                )
            self.load_seconds = float(message["load_seconds"])
            self._socket = parent_socket
            self._process = process
            self._socket.settimeout(request_timeout)
            self._finalizer = weakref.finalize(
                self, _force_cleanup, self._socket, self._process
            )
        except BaseException:
            try:
                child_socket.close()
            except OSError:
                pass
            _force_cleanup(parent_socket, process)
            raise

    @property
    def pid(self) -> int:
        if self._process is None:
            raise RuntimeError("vLLM worker did not start")
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def _exchange(
        self, message: Mapping[str, Any], raw_payload: bytes = b""
    ) -> dict[str, Any]:
        if self._closed or self._socket is None:
            raise RuntimeError("vLLM worker is closed")
        try:
            send_frame(self._socket, message, raw_payload)
            response, response_raw = recv_frame(self._socket)
        except BaseException:
            self._abort()
            raise
        if response_raw:
            self._abort()
            raise IPCProtocolError("worker response must not contain a raw payload")
        if response.get("type") == "error":
            raise _remote_error(response)
        return response

    def generate(
        self,
        *,
        expanded_prompt_ids: Sequence[int],
        point_embeddings: Any,
        sampling: Any,
    ) -> "VLLMGeneration":
        shape, raw = _tensor_to_bf16_uint16(point_embeddings)
        return self.generate_raw(
            expanded_prompt_ids=expanded_prompt_ids,
            point_embeddings_shape=shape,
            point_embeddings_bf16_uint16=raw,
            sampling=sampling,
        )

    def generate_raw(
        self,
        *,
        expanded_prompt_ids: Sequence[int],
        point_embeddings_shape: Sequence[int],
        point_embeddings_bf16_uint16: bytes | bytearray | memoryview,
        sampling: Any,
    ) -> "VLLMGeneration":
        """Low-level torch-free entry point for protocol tests and other callers."""

        shape = tuple(int(value) for value in point_embeddings_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(f"point_embeddings_shape must be positive [L,H]: {shape}")
        raw_view = memoryview(point_embeddings_bf16_uint16).cast("B")
        expected_nbytes = shape[0] * shape[1] * 2
        if len(raw_view) != expected_nbytes:
            raise ValueError(
                f"BF16 payload must contain {expected_nbytes} bytes, got {len(raw_view)}"
            )
        prompt_ids = [int(value) for value in expanded_prompt_ids]
        if not prompt_ids:
            raise ValueError("expanded_prompt_ids must not be empty")

        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            message = {
                "type": "generate",
                "request_id": request_id,
                "expanded_prompt_ids": prompt_ids,
                "point_embeddings": {
                    "shape": list(shape),
                    "logical_dtype": "bfloat16",
                    "storage_dtype": "uint16",
                    "byteorder": sys.byteorder,
                },
                "sampling": _sampling_payload(sampling),
            }
            response = self._exchange(message, raw_view.tobytes())
        if response.get("type") != "generation":
            raise IPCProtocolError(
                f"expected a generation frame, received {response.get('type')!r}"
            )
        if response.get("request_id") != request_id:
            raise IPCProtocolError(
                f"response request_id={response.get('request_id')} does not match {request_id}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise IPCProtocolError("generation response is missing a result object")
        from .hybrid import VLLMGeneration

        return VLLMGeneration(
            output_token_ids=tuple(int(value) for value in result["output_token_ids"]),
            finish_reason=(
                None
                if result.get("finish_reason") is None
                else str(result["finish_reason"])
            ),
            stop_reason=result.get("stop_reason"),
            prompt_match=bool(result["prompt_match"]),
            generate_seconds=float(result["generate_seconds"]),
            warmup_seconds=float(result["warmup_seconds"]),
            hit_seg_limit=bool(result["hit_seg_limit"]),
            max_tokens_used=int(result["max_tokens_used"]),
        )

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sock = self._socket
            process = self._process
            self._socket = None
            try:
                if sock is not None and process is not None and process.poll() is None:
                    sock.settimeout(self.close_timeout)
                    send_frame(sock, {"type": "close"})
                    response, raw = recv_frame(sock)
                    if raw or response.get("type") != "closed":
                        raise IPCProtocolError("worker did not confirm closure")
            except (OSError, EOFError, TimeoutError, IPCProtocolError):

                pass
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if process is not None:
                try:
                    process.wait(timeout=self.close_timeout)
                except subprocess.TimeoutExpired:
                    _force_cleanup(None, process)
            if self._finalizer is not None and self._finalizer.alive:
                self._finalizer.detach()

    def _abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        _force_cleanup(self._socket, self._process)
        self._socket = None
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()

    def __enter__(self) -> "PointVLLMProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
