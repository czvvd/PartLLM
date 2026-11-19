"""Optional, narrowly scoped optimizations for PartLLM's Jittor inference."""
from functools import wraps


def wrap_packed_sequence_check(original, torch):
    """Avoid evaluating a tautology for a valid single-token integer position.

    The upstream check subtracts the sole value from itself and reduces it;
    for a 1x1 int32/int64 tensor the predicate is necessarily false. Every
    other input is delegated unchanged to the original Transformers function.
    """
    @wraps(original)
    def check(position_ids, batch_size):
        if (position_ids is not None and batch_size == 1
                and getattr(position_ids, 'ndim', None) == 2
                and tuple(position_ids.shape) == (1, 1)
                and position_ids.dtype in (torch.int32, torch.int64)
                and position_ids.device.type == 'cuda'):
            return False
        return original(position_ids, batch_size)
    check._partllm_single_token_optimized = True
    return check


def install():
    import hashlib
    import inspect
    import os
    import torch
    if not hasattr(torch, '_torch_compat_install_context'):
        raise RuntimeError('PartLLM runtime optimization requires the Jittor Torch frontend')
    if os.environ.get('PARTLLM_FAST_DECODER_KNN', '0') == '1':
        from partllm_mask_decoder_knn import install_decoder_knn
        install_decoder_knn()
    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers
    original = flash_utils._is_packed_sequence
    if getattr(original, '_partllm_single_token_optimized', False):
        return
    expected = '638885760ad4134b6d253d8aa5af54cc78b5eae6bc805d53120286865a43fbfc'
    actual = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if transformers.__version__ != '4.57.3' or actual != expected:
        raise RuntimeError('unsupported Transformers packed-sequence implementation; disable the optional optimization')
    flash_utils._is_packed_sequence = wrap_packed_sequence_check(original, torch)
