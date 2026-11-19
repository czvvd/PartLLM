"""Explicit Jittor implementation of the segment_csr subset used by PartLLM.

Supports unbatched CSR pointers, two-dimensional float32/bfloat16 inputs and
sum, mean, min and max. Unsupported torch-scatter APIs fail explicitly.
"""
import jittor as jt
import torch

__version__ = '0.1.0+jittor'


def _segment_max_inference(src, indptr, groups, channels):
    # Each thread owns one output channel, preserving CSR order and ties.
    cpu_src = r'''
    for (int group = 0; group < out0_shape0; ++group) {
      auto begin = in1_p[group], end = in1_p[group + 1];
      for (int channel = 0; channel < out0_shape1; ++channel) {
        float value = -3.402823466e+38F;
        for (auto row = begin; row < end; ++row) {
          float current = in0_p[row * out0_shape1 + channel];
          if (current > value) value = current;
        }
        out0_p[group * out0_shape1 + channel] = begin == end ? 0.f : value;
      }
    }
    '''
    cuda_src = r'''
    __global__ void partllm_csr_max(const in0_type* x, const long long* ptr,
                                    out0_type* out, int groups, int channels) {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= groups * channels) return;
      int group = index / channels, channel = index % channels;
      long long begin = ptr[group], end = ptr[group + 1];
      if (begin == end) { out[index] = 0.f; return; }
      float value = INIT_VALUE;
      for (long long row = begin; row < end; ++row) {
        float current = x[row * channels + channel];
        if (current > value) value = current;
      }
      out[index] = value;
    }
    partllm_csr_max<<<(out0_shape0*out0_shape1+255)/256,256>>>(
        in0_p, (const long long*)in1_p, out0_p, out0_shape0, out0_shape1);
    '''
    init_value = ('-3.3895313892515355e+38F' if str(src.dtype) in ('bfloat16', 'torch.bfloat16')
                  else '-3.402823466e+38F')
    cuda_src = cuda_src.replace('INIT_VALUE', init_value)
    return jt.code([groups, channels], src.dtype, [src, indptr],
                   cpu_src=cpu_src, cuda_src=cuda_src)


def _segment_sum_inference(src, indptr, groups, channels, mean):
    # torch-scatter accumulates CPU BF16 in float32 but CUDA BF16 in BF16.
    # Process each CSR group in order to avoid nondeterministic atomics.
    cpu_src = r'''
    for (int group = 0; group < out0_shape0; ++group) {
      auto begin = in1_p[group], end = in1_p[group + 1];
      for (int channel = 0; channel < out0_shape1; ++channel) {
        float total = 0.f;
        for (auto row = begin; row < end; ++row)
          total += (float)in0_p[row * out0_shape1 + channel];
        out0_type result = (out0_type)total;
        if (DO_MEAN) result = result / (out0_type)(end > begin ? end - begin : 1);
        out0_p[group * out0_shape1 + channel] = result;
      }
    }
    '''
    cuda_src = r'''
    __global__ void partllm_csr_sum(const in0_type* x, const long long* ptr,
                                    out0_type* out, int groups, int channels) {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= groups * channels) return;
      int group = index / channels, channel = index % channels;
      long long begin = ptr[group], end = ptr[group + 1];
      out0_type result = (out0_type)0;
      for (long long row = begin; row < end; ++row)
        result = result + x[row * channels + channel];
      if (DO_MEAN) result = result / (out0_type)(end > begin ? end - begin : 1);
      out[index] = result;
    }
    partllm_csr_sum<<<(out0_shape0*out0_shape1+255)/256,256>>>(
        in0_p, (const long long*)in1_p, out0_p, out0_shape0, out0_shape1);
    '''
    flag = '1' if mean else '0'
    return jt.code([groups, channels], src.dtype, [src, indptr],
                   cpu_src=cpu_src.replace('DO_MEAN', flag),
                   cuda_src=cuda_src.replace('DO_MEAN', flag))


def segment_csr(src, indptr, out=None, reduce='sum'):
    if out is not None:
        raise NotImplementedError('segment_csr out= is not supported')
    if src.ndim != 2 or indptr.ndim != 1:
        raise NotImplementedError('segment_csr supports 2-D src and 1-D indptr')
    if str(src.dtype) not in ('float32', 'torch.float32', 'bfloat16', 'torch.bfloat16'):
        raise NotImplementedError('segment_csr unsupported input dtype: ' + str(src.dtype))
    if str(indptr.dtype) not in ('int64', 'torch.int64'):
        raise TypeError('segment_csr requires int64 CSR pointers')
    if src.device != indptr.device:
        raise ValueError('src and indptr must be on the same device')
    if reduce not in ('sum', 'add', 'mean', 'min', 'max'):
        raise ValueError('unsupported segment_csr reduction: ' + str(reduce))
    if int(indptr.shape[0]) < 1:
        raise ValueError('indptr must contain at least one entry')
    n, channels = map(int, src.shape)
    groups = int(indptr.shape[0]) - 1
    if groups == 0:
        return src[:0]
    if channels == 0:
        raise NotImplementedError('segment_csr requires at least one channel')
    if (reduce == 'max' and n and jt.flags.no_grad
            and str(indptr.dtype) in ('int64', 'torch.int64')
            and src.ndim == 2):
        return _segment_max_inference(src.contiguous(), indptr.contiguous(), groups, channels)
    if reduce in ('sum', 'add', 'mean') and n and jt.flags.no_grad:
        return _segment_sum_inference(src.contiguous(), indptr.contiguous(),
                                      groups, channels, reduce == 'mean')
    counts = indptr[1:] - indptr[:-1]
    if n == 0:
        return torch.zeros((groups, channels), device=src.device, dtype=src.dtype) + src.sum() * 0
    indices = torch.arange(n, device=src.device, dtype=torch.int64)
    ids = jt.searchsorted(indptr[1:], indices, right=True, out_int32=True)
    operation = {'sum': 'add', 'add': 'add', 'mean': 'add',
                 'min': 'minimum', 'max': 'maximum'}[reduce]
    result = src.reindex_reduce(operation, [groups, channels],
                                ['@e0(i0)', 'i1'], extras=[ids])
    if reduce in ('min', 'max') and not jt.flags.no_grad:
        # torch_scatter routes gradient through the first matching extremum,
        # while reindex_reduce routes it to every tied extremum.
        positions = indices.reshape(n, 1).expand(n, channels)
        candidates = torch.where(src == result[ids], positions,
                                 torch.full_like(positions, n))
        winner = candidates.reindex_reduce('minimum', [groups, channels],
                                           ['@e0(i0)', 'i1'], extras=[ids])
        columns = torch.arange(channels, device=src.device, dtype=torch.int64)
        result = src[winner.clamp(min=0, max=n - 1), columns.reshape(1, channels)]
    if reduce == 'mean':
        result = result / counts.clamp(min=1).to(dtype=src.dtype).reshape(groups, 1)
    return torch.where(counts.reshape(groups, 1) > 0, result, torch.zeros_like(result))
