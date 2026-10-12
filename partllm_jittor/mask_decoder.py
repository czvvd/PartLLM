"""Jittor implementation of the PartLLM joint mask decoder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import jittor as jt
from jittor import nn
import numpy as np


from .checkpoint import MaskDecoderConfig, read_decoder_weights


class WeightStore:
    """Lazy Jittor tensors backed by NumPy checkpoint weights."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = weights
        self._jittor_weights: Dict[str, jt.Var] = {}

    @classmethod
    def from_npz(cls, path: str | Path) -> "WeightStore":
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({name: archive[name].astype(np.float32, copy=False) for name in archive.files})

    @classmethod
    def from_checkpoint(cls, model_dir: str | Path) -> "WeightStore":
        return cls(read_decoder_weights(model_dir))

    def tensor(self, name: str) -> jt.Var:
        if name not in self._jittor_weights:
            try:
                self._jittor_weights[name] = jt.array(self.weights[name])
            except KeyError as exc:
                raise KeyError(f"Missing decoder checkpoint tensor: {name}") from exc
        return self._jittor_weights[name]

    def release_all(self) -> None:
        self._jittor_weights.clear()
        jt.gc()


def _linear(x: jt.Var, weights: WeightStore, name: str) -> jt.Var:
    return jt.matmul(x, weights.tensor(f"{name}.weight").transpose()) + weights.tensor(f"{name}.bias")


def _layer_norm(x: jt.Var, weights: WeightStore, name: str, eps: float = 1e-5) -> jt.Var:
    mean = x.mean(dim=-1, keepdims=True)
    variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdims=True)
    return (x - mean) / jt.sqrt(variance + eps) * weights.tensor(f"{name}.weight") + weights.tensor(f"{name}.bias")


def _mlp(x: jt.Var, weights: WeightStore, name: str) -> jt.Var:
    x = nn.relu(_linear(x, weights, f"{name}.layers.0"))
    x = nn.relu(_linear(x, weights, f"{name}.layers.1"))
    return _linear(x, weights, f"{name}.layers.2")


def _attention(q: jt.Var, k: jt.Var, v: jt.Var, weights: WeightStore, name: str, num_heads: int) -> jt.Var:
    q = _linear(q, weights, f"{name}.q_proj")
    k = _linear(k, weights, f"{name}.k_proj")
    v = _linear(v, weights, f"{name}.v_proj")
    batch, tokens, channels = q.shape
    head_dim = channels // num_heads
    q = q.reshape(batch, tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(batch, k.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(batch, v.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    scores = jt.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(head_dim)
    attended = jt.matmul(nn.softmax(scores, dim=-1), v)
    attended = attended.permute(0, 2, 1, 3).reshape(batch, tokens, channels)
    return _linear(attended, weights, f"{name}.out_proj")


def _transformer_mlp(x: jt.Var, weights: WeightStore, name: str) -> jt.Var:

    return _linear(nn.relu(_linear(x, weights, f"{name}.lin1")), weights, f"{name}.lin2")


def _interpolate_features(x: jt.Var, indices: jt.Var, interpolation_weights: jt.Var) -> jt.Var:
    """Gather [B, Nk, C] features into [B, Nq, K, C] then interpolate."""
    batch, num_keys, channels = x.shape
    _, num_queries, neighbors = indices.shape
    offsets = jt.arange(batch).reshape(batch, 1, 1) * num_keys
    flat_indices = (indices + offsets).reshape(-1)
    gathered = x.reshape(batch * num_keys, channels)[flat_indices]
    gathered = gathered.reshape(batch, num_queries, neighbors, channels)
    return (gathered * interpolation_weights.unsqueeze(-1)).sum(dim=2)


def compute_interp_weights(
    query: jt.Var, key: jt.Var, k: int = 3, eps: float = 1e-8, chunk_size: int = 8192
) -> tuple[jt.Var, jt.Var]:
    """3-NN inverse-squared-distance interpolation, chunked over query points."""
    batch, num_query, _ = query.shape
    all_indices, all_weights = [], []
    for start in range(0, num_query, chunk_size):
        current = query[:, start : min(start + chunk_size, num_query), :]


        differences = current.unsqueeze(2) - key.unsqueeze(1)
        distances_squared = (differences * differences).sum(dim=3)
        distances_squared, indices = jt.topk(distances_squared, k, dim=2, largest=False)


        inverse_distance = 1.0 / jt.maximum(distances_squared, eps)
        interpolation_weights = inverse_distance / inverse_distance.sum(dim=2, keepdims=True)
        all_indices.append(indices.numpy().astype(np.int32))
        all_weights.append(interpolation_weights.numpy())
    return jt.array(np.concatenate(all_indices, axis=1)), jt.array(np.concatenate(all_weights, axis=1))


class JittorMaskDecoder:
    """Joint inference decoder configured by a PartLLM checkpoint."""

    def __init__(self, config: MaskDecoderConfig, weights: WeightStore):
        if config.transformer_embedding_dim != config.mask_decoder_transformer_dim:
            raise ValueError("This decoder requires equal transformer embedding and decoder dimensions.")
        self.config = config
        self.weights = weights

    def _two_way_transformer(self, pc_embeddings: jt.Var, pc_pe: jt.Var, point_embeddings: jt.Var) -> tuple[jt.Var, jt.Var]:
        queries, keys = point_embeddings, pc_embeddings
        root = "mask_decoder.transformer"
        for layer_index in range(self.config.transformer_depth):
            block = f"{root}.layers.{layer_index}"
            if layer_index == 0:
                queries = _attention(queries, queries, queries, self.weights, f"{block}.self_attn", self.config.transformer_num_heads)
            else:
                queries = queries + _attention(
                    queries + point_embeddings, queries + point_embeddings, queries, self.weights,
                    f"{block}.self_attn", self.config.transformer_num_heads,
                )
            queries = _layer_norm(queries, self.weights, f"{block}.norm1")
            queries = queries + _attention(
                queries + point_embeddings, keys + pc_pe, keys, self.weights,
                f"{block}.cross_attn_token_to_image", self.config.transformer_num_heads,
            )
            queries = _layer_norm(queries, self.weights, f"{block}.norm2")
            queries = _layer_norm(
                queries + _transformer_mlp(queries, self.weights, f"{block}.mlp"), self.weights, f"{block}.norm3"
            )
            keys = _layer_norm(
                keys + _attention(
                    keys + pc_pe, queries + point_embeddings, queries, self.weights,
                    f"{block}.cross_attn_image_to_token", self.config.transformer_num_heads,
                ),
                self.weights,
                f"{block}.norm4",
            )
        queries = _layer_norm(
            queries + _attention(
                queries + point_embeddings, keys + pc_pe, keys, self.weights,
                f"{root}.final_attn_token_to_image", self.config.transformer_num_heads,
            ),
            self.weights,
            f"{root}.norm_final_attn",
        )
        return queries, keys

    def _position_embedding(self, centers: jt.Var) -> jt.Var:
        maximum = jt.maximum(centers.abs().max(), 1e-8)
        normalized = centers / maximum
        gaussian = self.weights.tensor("pe_layer.positional_encoding_gaussian_matrix")
        encoded = 2.0 * math.pi * jt.matmul(normalized, gaussian)
        return jt.concat([jt.sin(encoded), jt.cos(encoded)], dim=-1)

    def __call__(
        self,
        pc_embeddings: jt.Var,
        token_embeddings: jt.Var,
        centers: jt.Var,
        coords: jt.Var,
        feat_raw: Optional[jt.Var] = None,
        interp_indices: Optional[jt.Var] = None,
        interp_weights: Optional[jt.Var] = None,
    ) -> jt.Var:
        """Return joint logits shaped ``[B, num_seg_tokens_plus_bg, num_points]``."""
        if interp_indices is None or interp_weights is None:
            interp_indices, interp_weights = compute_interp_weights(coords, centers)
        pc_embeddings = _mlp(pc_embeddings, self.weights, "mask_decoder.encoder_mapper")
        token_embeddings = _mlp(token_embeddings, self.weights, "mask_decoder.seg_token_mapper")
        pc_pe = self._position_embedding(centers)
        token_outputs, source_outputs = self._two_way_transformer(pc_embeddings, pc_pe, token_embeddings)
        interpolated = _interpolate_features(source_outputs, interp_indices, interp_weights)

        if feat_raw is None:
            upscaled = _linear(interpolated, self.weights, "mask_decoder.output_upscaling.0")
            upscaled = _layer_norm(upscaled, self.weights, "mask_decoder.output_upscaling.1")
            upscaled = nn.gelu(upscaled)
            upscaled = nn.gelu(_linear(upscaled, self.weights, "mask_decoder.output_upscaling.3"))
        else:
            name = "mask_decoder.output_upscaling_with_fusion.0"
            weight = self.weights.tensor(f"{name}.weight")
            channels = interpolated.shape[-1]
            upscaled = jt.matmul(interpolated, weight[:, :channels].transpose())
            upscaled = upscaled + jt.matmul(feat_raw, weight[:, channels:].transpose())
            upscaled = upscaled + self.weights.tensor(f"{name}.bias")
            upscaled = _layer_norm(upscaled, self.weights, "mask_decoder.output_upscaling_with_fusion.1")
            upscaled = nn.gelu(upscaled)
            upscaled = nn.gelu(_linear(upscaled, self.weights, "mask_decoder.output_upscaling_with_fusion.3"))

        hypernetwork = _mlp(token_outputs, self.weights, "mask_decoder.output_hypernetworks_mlps.0")
        return jt.matmul(hypernetwork, upscaled.permute(0, 2, 1))
