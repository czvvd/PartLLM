"""Jittor inference port of the encoder-only Utonia PointTransformerV3.

The implementation replaces ``spconv.pytorch`` with a sparse 3x3x3 neighbour
lookup and replaces ``torch_scatter.segment_csr`` with Jittor reindex_reduce.
The encoder configuration uses 54/108/216/432/576 channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jittor as jt
from jittor import nn
import numpy as np

from .geometry import UtoniaPoints
from .hf_weights import LazySafeTensorStore


@dataclass
class _Level:
    feat: jt.Var
    coord: np.ndarray
    grid: np.ndarray


@dataclass(frozen=True)
class UtoniaEncoding:
    point_embeddings: jt.Var
    centers: jt.Var
    coords: jt.Var
    feat_raw: jt.Var


def _utonia_key(name: str) -> str:
    """Expand compact Utonia module paths to their checkpoint key prefix."""
    if name.startswith(("enc.", "embedding.")):
        return f"model.utonia_model.{name}"
    return name


def _linear(x: jt.Var, store: LazySafeTensorStore, name: str) -> jt.Var:
    name = _utonia_key(name)
    output = jt.matmul(x, store.tensor(f"{name}.weight").transpose())
    bias_name = f"{name}.bias"
    if bias_name in store.weight_map:
        output = output + store.tensor(bias_name)
    return output.astype(x.dtype)


def _layer_norm(x: jt.Var, store: LazySafeTensorStore, name: str, eps: float = 1e-5) -> jt.Var:
    name = _utonia_key(name)
    mean = x.float32().mean(dim=-1, keepdims=True)
    variance = ((x.float32() - mean) ** 2).mean(dim=-1, keepdims=True)
    normalised = ((x.float32() - mean) / jt.sqrt(variance + eps)).astype(x.dtype)
    return normalised * store.tensor(f"{name}.weight") + store.tensor(f"{name}.bias")


def _morton_code(grid: np.ndarray, transpose_xy: bool = False) -> np.ndarray:
    coords = np.asarray(grid, dtype=np.int64)
    if transpose_xy:
        coords = coords[:, [1, 0, 2]]
    code = np.zeros(len(coords), dtype=np.uint64)
    for bit in range(16):
        mask = np.uint64(1 << bit)
        code |= ((coords[:, 0].astype(np.uint64) & mask) << np.uint64(2 * bit + 2))
        code |= ((coords[:, 1].astype(np.uint64) & mask) << np.uint64(2 * bit + 1))
        code |= ((coords[:, 2].astype(np.uint64) & mask) << np.uint64(2 * bit))
    return code


def _serialization_order(grid: np.ndarray, order_index: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(_morton_code(grid, transpose_xy=bool(order_index % 2)), kind="stable").astype(np.int32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    return order, inverse


def _padding_indices(num_points: int, patch_size: int) -> np.ndarray:
    if num_points <= patch_size:
        return np.arange(num_points, dtype=np.int32)
    padded_size = int(math.ceil(num_points / patch_size) * patch_size)
    pad = np.arange(padded_size, dtype=np.int32)
    remainder = num_points % patch_size
    if remainder:

        pad[num_points:] = pad[num_points - patch_size : num_points - remainder]
    return pad


def _neighbour_indices(grid: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(grid, dtype=np.int64)
    coordinates = coordinates - coordinates.min(axis=0) + 1
    extent = coordinates.max(axis=0) + 2
    stride_x, stride_y = int(extent[1] * extent[2]), int(extent[2])
    keys = coordinates[:, 0] * stride_x + coordinates[:, 1] * stride_y + coordinates[:, 2]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    indices = np.full((len(grid), 27), len(grid), dtype=np.int32)
    offset_index = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                target = keys + dx * stride_x + dy * stride_y + dz
                found = np.searchsorted(sorted_keys, target)
                valid = found < len(grid)
                valid &= sorted_keys[np.minimum(found, len(grid) - 1)] == target
                indices[valid, offset_index] = order[found[valid]]
                offset_index += 1
    return indices


def _sparse_cpe(feat: jt.Var, grid: np.ndarray, store: LazySafeTensorStore, name: str) -> jt.Var:
    """Submanifold 3x3x3 convolution matching spconv's [Cout, K, K, K, Cin] weight layout."""
    name = _utonia_key(name)
    weight = store.tensor(f"{name}.weight")
    bias = store.tensor(f"{name}.bias")
    num_points, channels = feat.shape
    padded_feat = jt.concat([feat, jt.zeros((1, channels), dtype=feat.dtype)], dim=0)
    neighbours = jt.array(_neighbour_indices(grid))
    output = jt.zeros((num_points, weight.shape[0]), dtype=feat.dtype) + bias
    offset_index = 0
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                gathered = padded_feat[neighbours[:, offset_index]]
                output = output + jt.matmul(gathered, weight[:, dx, dy, dz, :].transpose()).astype(feat.dtype)
                offset_index += 1
    return output


def _rope(q: jt.Var, k: jt.Var, coords: np.ndarray, inv_freq: jt.Var) -> tuple[jt.Var, jt.Var]:

    coord_var = jt.array(coords.astype(np.float32))
    frequencies = coord_var.unsqueeze(-1) * inv_freq.reshape(1, 1, -1)
    phase = jt.concat([frequencies, frequencies], dim=-1).astype(q.dtype)
    chunk = q.shape[-1] // 3
    q_out, k_out = [], []
    for axis in range(3):
        cos = jt.cos(phase[:, axis, :]).unsqueeze(1)
        sin = jt.sin(phase[:, axis, :]).unsqueeze(1)
        q_part, k_part = q[:, :, axis * chunk : (axis + 1) * chunk], k[:, :, axis * chunk : (axis + 1) * chunk]
        half = chunk // 2
        q_rot = jt.concat([-q_part[:, :, half:], q_part[:, :, :half]], dim=-1)
        k_rot = jt.concat([-k_part[:, :, half:], k_part[:, :, :half]], dim=-1)
        q_out.append(q_part * cos + q_rot * sin)
        k_out.append(k_part * cos + k_rot * sin)
    return jt.concat(q_out, dim=-1), jt.concat(k_out, dim=-1)


class JittorUtoniaEncoder:
    """Utonia encoder with 54/108/216/432/576 channels."""

    channels = (54, 108, 216, 432, 576)
    depths = (3, 3, 3, 12, 3)
    heads = (3, 6, 12, 24, 32)
    patch_sizes = (1024, 1024, 1024, 1024, 1024)

    def __init__(self, weights: LazySafeTensorStore, attention_group_chunk: int = 4):
        self.weights = weights
        self.attention_group_chunk = attention_group_chunk

    def _detach_inference(self, value: jt.Var) -> jt.Var:
        if self.weights.dtype == "float32":
            return jt.array(value.numpy())
        return value

    def _attention(self, level: _Level, name: str, heads: int, patch_size: int, order_index: int) -> jt.Var:
        order, inverse = _serialization_order(level.grid, order_index)


        patch_size = min(patch_size, len(order))
        pad = _padding_indices(len(order), patch_size)
        ordered = order[pad]
        feat = level.feat[jt.array(ordered)]
        coords = level.coord[ordered]
        channels = feat.shape[-1]
        head_dim = channels // heads
        qkv = _linear(feat, self.weights, f"{name}.qkv").reshape(-1, 3, heads, head_dim)
        q, k, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = _rope(q, k, coords, self.weights.tensor(f"{_utonia_key(name)}.rope.inv_freq"))


        attention_dtype = "bfloat16" if self.weights.dtype == "float32" else q.dtype
        q, k, value = q.astype(attention_dtype), k.astype(attention_dtype), value.astype(attention_dtype)
        groups = len(ordered) // patch_size
        q = q.reshape(groups, patch_size, heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(groups, patch_size, heads, head_dim).permute(0, 2, 1, 3)
        value = value.reshape(groups, patch_size, heads, head_dim).permute(0, 2, 1, 3)
        output_chunks = []
        for start in range(0, groups, self.attention_group_chunk):
            end = min(start + self.attention_group_chunk, groups)
            scores = jt.matmul(q[start:end], k[start:end].permute(0, 1, 3, 2)) * (head_dim ** -0.5)
            attended = jt.matmul(nn.softmax(scores, dim=-1).astype(value.dtype), value[start:end])
            output_chunks.append(attended.numpy())
        output = jt.array(np.concatenate(output_chunks, axis=0)).permute(0, 2, 1, 3).reshape(-1, channels).astype(level.feat.dtype)
        output = _linear(output, self.weights, f"{name}.proj")
        return output[jt.array(inverse)]

    def _block(self, level: _Level, prefix: str, heads: int, patch_size: int, order_index: int) -> _Level:
        feat = level.feat
        cpe = _sparse_cpe(feat, level.grid, self.weights, f"{prefix}.cpe.0")
        cpe = _linear(cpe, self.weights, f"{prefix}.cpe.1")
        feat = feat + _layer_norm(cpe, self.weights, f"{prefix}.cpe.2")
        shortcut = feat
        attention_input = _layer_norm(feat, self.weights, f"{prefix}.norm1.0")
        level = _Level(attention_input, level.coord, level.grid)
        feat = shortcut + self._attention(level, f"{prefix}.attn", heads, patch_size, order_index)
        shortcut = feat
        mlp_input = _layer_norm(feat, self.weights, f"{prefix}.norm2.0")
        mlp = nn.gelu(_linear(mlp_input, self.weights, f"{prefix}.mlp.0.fc1"))
        feat = shortcut + _linear(mlp, self.weights, f"{prefix}.mlp.0.fc2")
        feat.sync()
        self.weights.release_prefix(f"model.utonia_model.{prefix}.")
        return _Level(self._detach_inference(feat), level.coord, level.grid)

    def _pool(self, level: _Level, prefix: str) -> tuple[_Level, np.ndarray]:
        projected = _linear(level.feat, self.weights, f"{prefix}.proj")
        pooled_grid, inverse, counts = np.unique(level.grid // 2, axis=0, return_inverse=True, return_counts=True)
        cluster = jt.array(inverse.astype(np.int32))


        pooled_feat = projected.float32().reindex_reduce(
            "maximum", [len(pooled_grid), projected.shape[1]], ["@e0(i0)", "i1"], extras=[cluster]
        ).astype(projected.dtype)
        coordinate_sum = jt.array(level.coord.astype(np.float32)).reindex_reduce(
            "add", [len(pooled_grid), 3], ["@e0(i0)", "i1"], extras=[cluster]
        )
        coord = coordinate_sum.numpy() / counts[:, None].astype(np.float32)
        pooled_feat = _layer_norm(pooled_feat, self.weights, f"{prefix}.norm.0")
        pooled_feat = nn.gelu(pooled_feat)
        pooled_feat.sync()
        self.weights.release_prefix(f"model.utonia_model.{prefix}.")
        return _Level(self._detach_inference(pooled_feat), coord, pooled_grid.astype(np.int32)), inverse.astype(np.int32)

    @staticmethod
    def _fps_indices(coords: np.ndarray, count: int) -> np.ndarray:
        if count > len(coords):
            raise ValueError(f"Requested {count} point tokens from only {len(coords)} voxel points")
        chosen = np.empty(count, dtype=np.int32)
        distance = np.full(len(coords), np.inf, dtype=np.float32)
        farthest = 0
        for index in range(count):
            chosen[index] = farthest
            squared = ((coords - coords[farthest]) ** 2).sum(axis=1)
            distance = np.minimum(distance, squared)
            farthest = int(distance.argmax())
        return chosen

    def encode(self, points: UtoniaPoints, num_tokens: int = 4096) -> UtoniaEncoding:
        feature_dtype = np.float32 if self.weights.dtype == "float32" else np.float16
        feat = jt.array(points.feat.astype(feature_dtype))
        feat = _linear(feat, self.weights, "model.utonia_model.embedding.stem.linear")
        feat = nn.gelu(_layer_norm(feat, self.weights, "model.utonia_model.embedding.stem.norm"))
        feat.sync()
        self.weights.release_prefix("model.utonia_model.embedding.")
        level = _Level(self._detach_inference(feat), points.coord, points.grid_coord)
        levels, inverses = [], []
        for stage, (depth, heads, patch_size) in enumerate(zip(self.depths, self.heads, self.patch_sizes)):
            if stage:
                level, inverse = self._pool(level, f"enc.enc{stage}.down")
                inverses.append(inverse)
            for block in range(depth):
                level = self._block(level, f"enc.enc{stage}.block{block}", heads, patch_size, block % 2)
            levels.append(level)
        full_feat = level.feat
        for stage in range(len(levels) - 2, -1, -1):
            full_feat = jt.concat([levels[stage].feat, full_feat[jt.array(inverses[stage])]], dim=-1)
        full_feat.sync()
        raw_feat = full_feat[jt.array(points.inverse.astype(np.int32))]
        raw_coord = jt.array(points.coord[points.inverse].astype(np.float32))
        fps_indices = self._fps_indices(raw_coord.numpy(), num_tokens)
        centers = raw_coord[jt.array(fps_indices)]
        token_feat = raw_feat[jt.array(fps_indices)]
        point_embeddings = _linear(token_feat, self.weights, "model.utonia_proj")
        return UtoniaEncoding(point_embeddings=point_embeddings, centers=centers, coords=raw_coord, feat_raw=raw_feat)
