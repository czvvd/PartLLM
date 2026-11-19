import sys
mode = sys.argv[1]
if mode == 'jittor':
    import jittor
import torch
assert hasattr(torch, '_torch_compat_install_context') == (mode == 'jittor')
from transformers.modeling_flash_attention_utils import _is_packed_sequence as original
from partllm_runtime_optimizations import wrap_packed_sequence_check
calls = [0]
def observed(pos, batch):
    calls[0] += 1
    return original(pos, batch)
fast = wrap_packed_sequence_check(observed, torch)
tests = [
    (None, 1, True),
    (torch.tensor([[0]], dtype=torch.int64, device='cuda'), 1, False),
    (torch.tensor([[-7]], dtype=torch.int64, device='cuda'), 1, False),
    (torch.tensor([[2**62]], dtype=torch.int64, device='cuda'), 1, False),
    (torch.tensor([[17]], dtype=torch.int32, device='cuda'), 1, False),
    (torch.tensor([[17]], dtype=torch.int64, device='cpu'), 1, True),
    (torch.tensor([[0]], dtype=torch.int64, device='cuda'), 2, True),
    (torch.tensor([[0, 1, 2]], dtype=torch.int64, device='cuda'), 1, True),
    (torch.tensor([[0, 1, 0]], dtype=torch.int64, device='cuda'), 1, True),
    (torch.tensor([[0], [1]], dtype=torch.int64, device='cuda'), 1, True),
    (torch.tensor([[float('nan')]], dtype=torch.float32, device='cuda'), 1, True),
]
for i, (pos, batch, must_delegate) in enumerate(tests):
    expected = bool(original(pos, batch))
    before = calls[0]
    actual = bool(fast(pos, batch))
    if must_delegate:
        assert actual == expected, (mode, i, actual, expected)
    else:
        # PyTorch's mathematical result for a single integer position is false;
        # Jittor's unoptimized check can miscompute it for very large int64.
        assert actual is False, (mode, i, actual)
        if expected:
            print('ORIGINAL_JITTOR_DIFFERS_FROM_TORCH', i, flush=True)
    assert (calls[0] != before) == must_delegate, (mode, i, 'unexpected routing')
    print('CASE', mode, i, 'value', actual, 'delegated', must_delegate, flush=True)
print('ALL_CASES_PASSED', mode, len(tests), flush=True)
from partllm_runtime_optimizations import install
if mode == 'jittor':
    import transformers.modeling_flash_attention_utils as flash_utils
    install()
    installed = flash_utils._is_packed_sequence
    install()
    assert flash_utils._is_packed_sequence is installed
    assert not bool(installed(torch.tensor([[4]], dtype=torch.int64, device='cuda'), 1))
    print('INSTALL_AND_IDEMPOTENCE_PASSED', flush=True)
else:
    try:
        install()
    except RuntimeError:
        print('ORACLE_INSTALL_REJECTED', flush=True)
    else:
        raise AssertionError('optimization must reject real PyTorch')
