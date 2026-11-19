import numpy as np
import pytest
import torch
from torch_scatter import segment_csr
from torch_cluster import fps


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('reduce,expected,expected_grad', [
    ('sum', [[-6., -4.], [0., 0.], [5., 2.]], [[1., 1.], [1., 1.], [1., 1.]]),
    ('mean', [[-3., -2.], [0., 0.], [5., 2.]], [[.5, .5], [.5, .5], [1., 1.]]),
    ('min', [[-4., -3.], [0., 0.], [5., 2.]], [[0., 1.], [1., 0.], [1., 1.]]),
    ('max', [[-2., -1.], [0., 0.], [5., 2.]], [[1., 0.], [0., 1.], [1., 1.]]),
])
def test_segment_forward_and_grad(device, reduce, expected, expected_grad):
    x = torch.tensor([[-2., -3.], [-4., -1.], [5., 2.]], device=device, requires_grad=True)
    ptr = torch.tensor([0, 2, 2, 3], device=device)
    out = segment_csr(x, ptr, reduce=reduce)
    grad = torch.autograd.grad(out.sum(), x)[0]
    np.testing.assert_allclose(out.detach().cpu().numpy(), expected)
    np.testing.assert_allclose(grad.detach().cpu().numpy(), expected_grad)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_segment_empty_and_invalid_arguments(device):
    x = torch.tensor([[1., -3.], [2., -4.]], device=device)
    ptr = torch.tensor([0], device=device)
    assert tuple(segment_csr(x, ptr).shape) == (0, 2)
    with pytest.raises(NotImplementedError):
        segment_csr(x, ptr, out=torch.empty((0, 2), device=device))
    with pytest.raises(ValueError):
        segment_csr(x, ptr, reduce='prod')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fps_reference_and_random_start(device):
    points = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [2., 2.], [1., 1.]], device=device)
    actual = fps(points, ratio=.8, random_start=False).cpu().tolist()
    assert actual == [0, 3, 1, 2]
    random_indices = fps(points, ratio=.8, random_start=True).cpu().tolist()
    assert len(random_indices) == 4
    assert len(set(random_indices)) == 4
    assert all(0 <= index < len(points) for index in random_indices)
    with pytest.raises(NotImplementedError):
        fps(points, batch=torch.zeros(5, device=device, dtype=torch.int64))
