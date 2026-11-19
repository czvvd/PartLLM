import os
import sys
import numpy as np
mode, out_dir = sys.argv[1:]
if mode == 'jittor':
    import jittor
import torch
assert hasattr(torch, '_torch_compat_install_context') == (mode == 'jittor')
from torch_cluster import fps
from torch_scatter import segment_csr
arrays = {}
for device in ('cpu', 'cuda'):
    for n in (5, 17, 260, 1024):
        rng = np.random.default_rng(n)
        a = rng.standard_normal((n, 6), dtype=np.float32)
        x = torch.tensor(a, device=device)[:, ::2]
        for ratio in (.1, .5, 1.):
            y = fps(x, ratio=ratio, random_start=False)
            arrays[f'fps-{device}-{n}-{ratio}'] = y.cpu().numpy()
    xdata = np.array([[-2., -3.], [-4., -1.], [5., 2.], [5., 2.], [-1., -2.]], dtype=np.float32)
    for name, pointers in [('with-empty', [0, 2, 2, 5]), ('ties', [0, 2, 4, 5])]:
        for reduce in ('sum', 'mean', 'min', 'max'):
            x = torch.tensor(xdata, device=device, requires_grad=True)
            ptr = torch.tensor(pointers, device=device)
            y = segment_csr(x, ptr, reduce=reduce)
            grad = torch.autograd.grad(y.sum(), x)[0]
            arrays[f'seg-{device}-{name}-{reduce}'] = y.detach().cpu().numpy()
            arrays[f'grad-{device}-{name}-{reduce}'] = grad.detach().cpu().numpy()
    for reduce in ('sum', 'mean', 'min', 'max'):
        x = torch.tensor(xdata, device=device, dtype=torch.bfloat16, requires_grad=True)
        ptr = torch.tensor([0, 2, 2, 5], device=device)
        y = segment_csr(x, ptr, reduce=reduce)
        grad = torch.autograd.grad(y.sum(), x)[0]
        arrays[f'bf16-seg-{device}-{reduce}'] = y.detach().cpu().float().numpy()
        arrays[f'bf16-grad-{device}-{reduce}'] = grad.detach().cpu().float().numpy()
    with torch.no_grad():
        vals = np.array([[np.nan, -np.inf, -0., 0.], [1., -np.inf, 0., -0.],
                         [-2., 3., 0., -5.]], dtype=np.float32)
        x = torch.tensor(vals, device=device)
        ptr = torch.tensor([0, 2, 2, 3], device=device)
        arrays[f'no-grad-special-{device}'] = segment_csr(x, ptr, reduce='max').cpu().numpy()
        arrays[f'no-grad-sign-{device}'] = np.signbit(arrays[f'no-grad-special-{device}'])
        x = torch.tensor(vals, device=device, dtype=torch.bfloat16)
        arrays[f'no-grad-bf16-special-{device}'] = segment_csr(x, ptr, reduce='max').cpu().float().numpy()
        arrays[f'noograd-bf16-sign-{device}'] = np.signbit(arrays[f'no-grad-bf16-special-{device}'])
        rng = np.random.default_rng(321)
        x = torch.tensor(rng.standard_normal((257, 34), dtype=np.float32), device=device)[:, ::2]
        ptr = torch.tensor([0, 0, 1, 32, 197, 257], device=device)
        arrays[f'no-grad-random-view-{device}'] = segment_csr(x, ptr, reduce='max').cpu().numpy()
        for dtype_name, dtype in (('f32', torch.float32), ('bf16', torch.bfloat16)):
            for reduce in ('sum', 'mean'):
                values = rng.standard_normal((257, 3), dtype=np.float32)
                values = torch.tensor(values, device=device, dtype=dtype)
                arrays[f'no-grad-{dtype_name}-{reduce}-{device}'] = segment_csr(
                    values, ptr, reduce=reduce).cpu().float().numpy()
np.savez(os.path.join(out_dir, mode + '.npz'), **arrays)
print('FINISHED', mode, len(arrays), flush=True)
if mode == 'jittor':
    reference = np.load(os.path.join(out_dir, 'torch.npz'))
    failures = []
    for key, a in arrays.items():
        b = reference[key]
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6):
            failures.append(key)
            print('DIFFERENT', key, 'actual', a.tolist(), 'expected', b.tolist(), flush=True)
    print('COMPARISON', len(arrays) - len(failures), '/', len(arrays), 'passed', flush=True)
    if failures:
        raise SystemExit(1)
