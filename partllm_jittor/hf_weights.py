"""PyTorch-free lazy access to sharded Hugging Face Safetensors weights."""

from __future__ import annotations

from pathlib import Path

import jittor as jt
import ml_dtypes
import numpy as np
from safetensors import safe_open

from .checkpoint import load_weight_map


class LazySafeTensorStore:
    """Load checkpoint tensors lazily at FP16, FP32, or Jittor BF16 precision."""

    def __init__(self, model_dir: str | Path, dtype: str = "float16"):
        self.model_dir = Path(model_dir)
        self.weight_map = load_weight_map(self.model_dir)
        self.dtype = dtype
        self._handles = {}
        self._cache = {}

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return [name for name in self.weight_map if name.startswith(prefix)]

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.model_dir / shard, framework="np", backend="pread")
        return self._handles[shard]

    def tensor(self, name: str) -> jt.Var:
        if name not in self._cache:
            try:
                array = self._handle(self.weight_map[name]).get_tensor(name)
            except KeyError as exc:
                raise KeyError(f"Checkpoint lacks tensor {name}") from exc


            if self.dtype == "bfloat16":
                self._cache[name] = jt.array(np.asarray(array, dtype=np.float32)).astype("bfloat16")
            else:
                self._cache[name] = jt.array(np.asarray(array, dtype=self.dtype))
        return self._cache[name]

    def release_prefix(self, prefix: str) -> None:
        """Release GPU tensors that will not be reused in this forward pass."""
        for name in [name for name in self._cache if name.startswith(prefix)]:
            del self._cache[name]
        jt.gc()

    def release_all(self) -> None:
        """Release all materialized Jittor tensors while retaining shard metadata."""
        self._cache.clear()
        jt.gc()
