"""Jittor FPS subset used by PartLLM; see LICENSE.torch-cluster.

Supports one implicit point cloud, 2-D float32 input and a scalar ratio.
The random-start distribution is supported, but the Jittor and PyTorch random
number generators are not bitwise interchangeable.
"""
import math
import numpy as np
import jittor as jt
import torch

__version__ = '0.1.0+jittor'


def fps(src, batch=None, ratio=None, random_start=True, batch_size=None, ptr=None):
    if batch is not None or ptr is not None or batch_size is not None:
        raise NotImplementedError('fps supports a single implicit point cloud')
    if src.ndim != 2 or str(src.dtype) not in ('float32', 'torch.float32'):
        raise NotImplementedError('fps supports a 2-D float32 point matrix')
    ratio = .5 if ratio is None else ratio
    if not isinstance(ratio, float) or not 0 < ratio <= 1:
        raise NotImplementedError('fps supports a scalar ratio in (0, 1]')
    n, dim = map(int, src.shape)
    if n == 0 or dim == 0:
        raise NotImplementedError('fps requires nonempty points and features')
    count = int(math.ceil(float(np.float32(n) * np.float32(ratio))))
    src = src.contiguous()
    start = ((torch.rand((1,), device=src.device) * n).long() if random_start
             else torch.zeros((1,), dtype=torch.int64, device=src.device))
    cpu_src = r'''
    std::vector<float> dist(in0_shape0, std::numeric_limits<float>::infinity());
    out0_p[0] = in1_p[0];
    for (int m=1; m<out0_shape0; ++m) {
      auto old = out0_p[m-1];
      float best = -1.0f; long long idx = 0;
      for (int i=0; i<in0_shape0; ++i) {
        float dd = 0.0f;
        for (int d=0; d<in0_shape1; ++d) {
          float v = in0_p[in0_shape1*old+d] - in0_p[in0_shape1*i+d];
          dd += v*v;
        }
        dist[i] = std::min(dist[i], dd);
        if (dist[i] > best) {best = dist[i]; idx = i;}
      }
      out0_p[m] = idx;
    }
    '''
    cuda_src = r'''
    __global__ void fps_subset(const float* src, long long* out, float* dist, int n, int k, int dim, const long long* start) {
      __shared__ float best_dist[256]; __shared__ long long best_idx[256];
      int tid=threadIdx.x;
      for (int i=tid; i<n; i+=256) dist[i]=5e4f;
      if (tid==0) out[0]=start[0];
      for (int m=1; m<k; ++m) {
        __syncthreads(); long long old=out[m-1];
        float best=-1.f; long long idx=0;
        for (int i=tid; i<n; i+=256) {
          float dd=0.f;
          for (int d=0; d<dim; ++d) {float v=src[dim*old+d]-src[dim*i+d]; dd=fmaf(v,v,dd);}
          dd=fminf(dist[i],dd); dist[i]=dd;
          if (dd>best) {best=dd; idx=i;}
        }
        best_dist[tid]=best; best_idx[tid]=idx;
        for (int step=1; step<256; step*=2) {
          __syncthreads();
          if (tid+step<256 && best_dist[tid]<best_dist[tid+step]) {
            best_dist[tid]=best_dist[tid+step]; best_idx[tid]=best_idx[tid+step];
          }
        }
        __syncthreads(); if (tid==0) out[m]=best_idx[0];
      }
    }
    fps_subset<<<1,256>>>(in0_p, (long long*)out0_p, out1_p, in0_shape0, out0_shape0, in0_shape1, (const long long*)in1_p);
    '''
    result, _ = jt.code([(count,), (n,)], ['int64', 'float32'], [src, start],
                        cpu_header='#include <vector>\n#include <limits>',
                        cpu_src=cpu_src, cuda_src=cuda_src)
    return result
