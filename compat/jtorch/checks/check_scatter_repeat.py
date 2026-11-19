import sys
import numpy as np
mode, output = sys.argv[1:]
if mode == 'jittor':
    import jittor
import torch
assert hasattr(torch, '_torch_compat_install_context') == (mode == 'jittor')
from torch_scatter import segment_csr
rng = np.random.default_rng(55)
n, channels, groups = 81920, 3, 8192
src = rng.standard_normal((n, channels), dtype=np.float32)
indptr = np.linspace(0, n, groups + 1, dtype=np.int64)
out = {}
for dtype_name, dtype in [('f32', torch.float32), ('bf16', torch.bfloat16)]:
    x = torch.tensor(src, device='cuda', dtype=dtype)
    ptr = torch.tensor(indptr, device='cuda')
    for reduce in ('sum', 'mean'):
        results = []
        with torch.no_grad():
            for _ in range(5):
                y = segment_csr(x, ptr, reduce=reduce)
                results.append(y.float().cpu().numpy())
        out[f'{dtype_name}-{reduce}'] = np.stack(results)
        for i in range(1, 5):
            print('REPEAT', mode, dtype_name, reduce, i,
                  'mismatch', np.count_nonzero(results[0] != results[i]),
                  'max', np.max(np.abs(results[0] - results[i])), flush=True)
np.savez(output + '/' + mode + '.npz', **out)
if mode == 'jittor':
    ref = np.load(output + '/torch.npz')
    errors = []
    for key, values in out.items():
        diff = np.abs(values[0] - ref[key][0])
        print('CROSS', key, 'mismatch', np.count_nonzero(diff), 'max', diff.max(),
              'mean', diff.mean(), flush=True)
        if not np.array_equal(values[0], ref[key][0]) or any(not np.array_equal(values[0], v) for v in values[1:]):
            errors.append(key)
    print('REPEAT_AND_CROSS', 'pass' if not errors else 'fail', errors, flush=True)
    if errors:
        raise SystemExit(1)
