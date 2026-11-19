"""FP32 Euclidean distances for PartLLM's no-gradient mask interpolation."""

from functools import wraps
import hashlib
import inspect


def install_decoder_knn():
    """Install the optimization for the validated Transformers implementation."""
    import torch
    import transformers
    from transformers.models.qwen3_vl import mask_decoder

    if not hasattr(torch, "_torch_compat_install_context"):
        raise RuntimeError("PartLLM decoder optimization requires the Jittor Torch frontend")
    original = mask_decoder.knn_points
    if getattr(original, "_partllm_decoder_knn_optimized", False):
        return
    expected = "a235ac97d8c058f24b1ed5325ecb253b2efc334284574053f9b99f83fe8d30ef"
    source = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if transformers.__version__ != "4.57.3" or source != expected:
        raise RuntimeError("Unsupported PartLLM nearest-neighbor implementation")
    mask_decoder.knn_points = wrap_knn_points(original)


def _padded_euclidean_mm(torch, matmul, query, key):
    # PyTorch 2.8 Distance.cpp::_euclidean_dist. Keep the norms in the
    # matmul rather than adding them afterward, where rounding can differ.
    query_norm = query.square().sum(dim=-1, keepdim=True)
    key_norm = key.square().sum(dim=-1, keepdim=True)
    left = torch.cat((-2.0 * query, query_norm, torch.ones_like(query_norm)), dim=-1)
    right = torch.cat((key, torch.ones_like(key_norm), key_norm), dim=-1)
    return matmul(left, right.transpose(-2, -1)).clamp_min(0).sqrt()


def wrap_knn_points(original):
    """Use the CUDA matrix-multiply path for the validated k=3 call."""
    import torch

    if not hasattr(torch, "_torch_compat_install_context"):
        return original

    import jittor as jt
    from jittor.compat.torch.frontend import tensor_frontend

    set_precision = jt.core._set_float32_precision
    reset_precision = jt.core._reset_float32_precision
    counts = {"mm": 0, "fallback": 0}

    @wraps(original)
    def knn_points(query, key, k, sorted=False, transpose=False):
        if (transpose or type(k) is not int or k != 3 or torch.is_grad_enabled()
                or query.ndim != 3 or key.ndim != 3
                or query.shape[-1] != 3 or key.shape[-1] != 3
                or not query.shape[0] or query.shape[0] != key.shape[0]
                or not query.shape[1] or key.shape[1] < k
                or max(query.shape[1], key.shape[1]) <= 25
                or query.dtype != torch.float32 or key.dtype != torch.float32
                or query.device.type != "cuda" or query.device != key.device
                or not query.is_contiguous() or not key.is_contiguous()):
            counts["fallback"] += 1
            return original(query, key, k, sorted=sorted, transpose=transpose)

        counts["mm"] += 1
        with (torch.autocast(device_type="cuda", enabled=False),
              jt.flag_scope(amp_reg=0), tensor_frontend(torch.Tensor, like=query)):
            # The native tiers are highest=0, high=1, medium=2. Use IEEE
            # FP32 for this GEMM even if the caller permits TF32 elsewhere.
            token = set_precision(0, 0)
            try:
                distance = _padded_euclidean_mm(torch, jt.matmul, query, key)
            finally:
                reset_precision(token)
            values, indices = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
        return values, indices

    knn_points.counts = counts
    knn_points._partllm_decoder_knn_optimized = True
    return knn_points
