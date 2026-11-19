"""Opt-in preparation and replay of exact checkpoint initialization draws."""

from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import tempfile
import threading

import numpy as np


HEADER_LIMIT = 4096
DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": np.uint16}


def install(jt, torch, counts):
    from jittor._core.dtypes import dtype_name
    from jittor.compat.torch.frontend import tensor_frontend
    from jittor.compat.torch.installers import nn_init
    from transformers.modeling_utils import PreTrainedModel

    root = Path(os.environ["PARTLLM_PREPARED_NORMAL_DIR"])
    mode = os.environ["PARTLLM_PREPARED_NORMAL_MODE"]
    identity = os.environ["PARTLLM_PREPARED_NORMAL_ID"]
    assert mode in ("prepare", "reuse") and root.is_dir()
    active = ContextVar("partllm_prepared_checkpoint_normal", default=False)
    original = PreTrainedModel._initialize_missing_keys
    original_normal = nn_init._normal_values

    def borrow(path, shape, dtype, target_dtype):
        try:
            stream = path.open("rb")
        except OSError:
            return None
        with stream:
            if os.fstat(stream.fileno()).st_size < 4:
                return None
            try:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
            except (OSError, ValueError):
                return None
        offset = 0
        try:
            size = struct.unpack_from("<I", mapping)[0]
            if size > HEADER_LIMIT or 4 + size > len(mapping):
                return None
            try:
                header = json.loads(mapping[4:4 + size])
            except (ValueError, UnicodeDecodeError):
                return None
            if not isinstance(header, list) or len(header) != 2 or not all(
                isinstance(value, str) for value in header
            ):
                return None
            state, expected = header
            start = 4 + size
            if (start % 64 or
                    len(mapping) - start != math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize):
                return None
            checksum = hashlib.sha256(state.encode("utf-8"))
            with memoryview(mapping)[start:] as payload:
                checksum.update(payload)
            if checksum.hexdigest() != expected:
                return None
            array = np.ndarray(shape, dtype=DTYPES[dtype], buffer=mapping, offset=start)
            offset = start  # The NumPy/DLPack owner now keeps the mapping alive.
            with tensor_frontend(torch.Tensor, device="cpu"):
                value = jt.from_dlpack(array, copy=False)
            if dtype == "bfloat16":
                value = value.view(target_dtype)
            value.requires_grad_(False)
            return value, state
        finally:
            if not offset:
                mapping.close()

    def prepare(path, value, dtype, state):
        if dtype == "bfloat16":
            value = value.view(torch.uint16)
        array = value.numpy()
        with memoryview(array).cast("B") as payload:
            checksum = hashlib.sha256(state.encode("utf-8"))
            checksum.update(payload)
            header = json.dumps([state, checksum.hexdigest()]).encode("utf-8")
            header += b" " * (-(len(header) + 4) % 64)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(struct.pack("<I", len(header)))
                    stream.write(header)
                    stream.write(payload)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    @wraps(original_normal)
    def normal(tensor, mean=0.0, std=1.0, generator=None):
        if (not active.get() or generator is not None or tensor.device.type != "cpu"
                or tensor.numel() < 262144 or threading.active_count() != 1
                or not math.isfinite(mean) or not math.isfinite(std) or std < 0):
            return original_normal(tensor, mean, std, generator)
        dtype = dtype_name(tensor.dtype)
        if dtype not in DTYPES:
            return original_normal(tensor, mean, std, generator)
        with jt.no_grad():
            jt.sync_all()
            shape = tuple(tensor.shape)
            start = jt.core.rng_state("cpu")
            fingerprint = json.dumps([identity, start, shape, dtype, float(mean).hex(), float(std).hex()])
            path = root / (hashlib.sha256(fingerprint.encode("utf-8")).hexdigest() + ".bin")
            if mode == "reuse":
                saved = borrow(path, shape, dtype, tensor.dtype)
                if saved is not None:
                    value, end = saved
                    # An intervening draw invalidates this replay.
                    if jt.core.rng_state("cpu") == start:
                        jt.core.set_rng_state("cpu", end)
                        counts["prepared_normal_hits"] = counts.get("prepared_normal_hits", 0) + 1
                        return value
                counts["prepared_normal_misses"] = counts.get("prepared_normal_misses", 0) + 1
                return original_normal(tensor, mean, std, generator)
            value = original_normal(tensor, mean, std, generator)
            value.sync()
            end = jt.core.rng_state("cpu")
            prepare(path, value, dtype, end)
            counts["prepared_normal_writes"] = counts.get("prepared_normal_writes", 0) + 1
            return value

    @wraps(original)
    def initialize(model, *args, **kwargs):
        token = active.set(True)
        try:
            return original(model, *args, **kwargs)
        finally:
            active.reset(token)

    nn_init._normal_values = normal
    PreTrainedModel._initialize_missing_keys = initialize
