"""Jittor Qwen3-VL text inference path for PartLLM.

PartLLM inference injects point-cloud
embeddings into the text sequence and does not require the vision tower.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jittor as jt
import numpy as np

from .hf_weights import LazySafeTensorStore


@dataclass(frozen=True)
class Qwen3TextConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "Qwen3TextConfig":
        config = json.loads(Path(path).read_text(encoding="utf-8"))["text_config"]
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            head_dim=int(config["head_dim"]),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(config["rope_theta"]),
        )


def _linear(x: jt.Var, store: LazySafeTensorStore, name: str) -> jt.Var:
    return jt.matmul(x, store.tensor(f"{name}.weight").transpose()).astype(x.dtype)


def _rms_norm(x: jt.Var, weight: jt.Var, eps: float) -> jt.Var:
    x32 = x.float32()
    output = x32 * (1.0 / jt.sqrt((x32 * x32).mean(dim=-1, keepdims=True) + eps))
    return output.astype(x.dtype) * weight


def _silu(x: jt.Var) -> jt.Var:
    return x * jt.sigmoid(x)


def _rotate_half(x: jt.Var) -> jt.Var:
    half = x.shape[-1] // 2
    return jt.concat([-x[..., half:], x[..., :half]], dim=-1)


def _detach_inference(value: jt.Var) -> jt.Var:
    """Break the lazy graph while keeping the materialized tensor on GPU."""
    value.sync()
    return value.detach()


@dataclass
class KVCache:
    key: jt.Var
    value: jt.Var
    length: int

    @property
    def capacity(self) -> int:
        return self.key.shape[2]


class JittorQwen3Text:
    """Qwen3 text model with dynamic KV cache and hidden-state access."""

    def __init__(
        self,
        config: Qwen3TextConfig,
        weights: LazySafeTensorStore,
        attention_chunk_size: int = 128,
        stream_layer_weights: bool = False,
    ):
        self.config = config
        self.weights = weights
        self.attention_chunk_size = attention_chunk_size
        self.stream_layer_weights = stream_layer_weights

    def embed(self, input_ids: jt.Var) -> jt.Var:
        return self.weights.tensor("model.language_model.embed_tokens.weight")[input_ids]

    def _rope(self, positions: jt.Var) -> tuple[jt.Var, jt.Var]:
        half_dim = self.config.head_dim // 2
        frequency_indices = jt.arange(half_dim).float32() * 2.0 / self.config.head_dim
        inv_freq = 1.0 / (self.config.rope_theta**frequency_indices)
        frequencies = positions.float32().unsqueeze(-1) * inv_freq.reshape(1, 1, -1)
        phase = jt.concat([frequencies, frequencies], dim=-1)
        return jt.cos(phase), jt.sin(phase)

    def _attention(
        self,
        hidden_states: jt.Var,
        layer_index: int,
        positions: jt.Var,
        cache: Optional[KVCache],
    ) -> tuple[jt.Var, KVCache]:
        prefix = f"model.language_model.layers.{layer_index}.self_attn"
        batch, query_length, _ = hidden_states.shape
        head_dim = self.config.head_dim
        query = _linear(hidden_states, self.weights, f"{prefix}.q_proj").reshape(
            batch, query_length, self.config.num_attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        key = _linear(hidden_states, self.weights, f"{prefix}.k_proj").reshape(
            batch, query_length, self.config.num_key_value_heads, head_dim
        ).permute(0, 2, 1, 3)
        value = _linear(hidden_states, self.weights, f"{prefix}.v_proj").reshape(
            batch, query_length, self.config.num_key_value_heads, head_dim
        ).permute(0, 2, 1, 3)
        query = _rms_norm(query, self.weights.tensor(f"{prefix}.q_norm.weight"), self.config.rms_norm_eps)
        key = _rms_norm(key, self.weights.tensor(f"{prefix}.k_norm.weight"), self.config.rms_norm_eps)
        cos, sin = self._rope(positions)
        cos, sin = cos.astype(query.dtype).unsqueeze(1), sin.astype(query.dtype).unsqueeze(1)
        query, key = query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin
        if cache is None:
            new_cache = KVCache(key, value, query_length)
            past_length = 0
        else:
            past_length = cache.length
            end = past_length + query_length
            if end > cache.capacity:
                raise ValueError("KV cache capacity was not reserved before attention")
            cache.key[:, :, past_length:end, :] = key
            cache.value[:, :, past_length:end, :] = value
            cache.length = end
            new_cache = cache
            key, value = cache.key, cache.value
        repeat = self.config.num_attention_heads // self.config.num_key_value_heads
        key_for_attention = key.unsqueeze(2).expand(batch, self.config.num_key_value_heads, repeat, key.shape[2], head_dim)
        value_for_attention = value.unsqueeze(2).expand(batch, self.config.num_key_value_heads, repeat, value.shape[2], head_dim)
        key_for_attention = key_for_attention.reshape(batch, self.config.num_attention_heads, key.shape[2], head_dim)
        value_for_attention = value_for_attention.reshape(batch, self.config.num_attention_heads, value.shape[2], head_dim)
        key_length = key.shape[2]
        key_transposed = key_for_attention.permute(0, 1, 3, 2)
        if query_length == 1:
            scores = jt.matmul(query.float32(), key_transposed.float32()) / math.sqrt(head_dim)
            if new_cache.length < key_length:
                valid = (jt.arange(key_length) < new_cache.length).float32()
                scores = scores + (1.0 - valid).reshape(1, 1, 1, key_length) * -65504.0
            output = jt.matmul(jt.nn.softmax(scores, dim=-1), value_for_attention.float32()).astype(value_for_attention.dtype)
        else:
            key_positions = jt.arange(key_length).reshape(1, key_length)
            outputs = []
            for start in range(0, query_length, self.attention_chunk_size):
                end = min(start + self.attention_chunk_size, query_length)
                scores = jt.matmul(
                    query[:, :, start:end, :].float32(), key_transposed.float32()
                ) / math.sqrt(head_dim)
                query_positions = jt.arange(past_length + start, past_length + end).reshape(end - start, 1)
                causal = (key_positions <= query_positions).astype(scores.dtype)
                penalty = jt.array([-65504.0], dtype="float32").astype(scores.dtype)
                scores = scores + (1.0 - causal).reshape(1, 1, end - start, key_length) * penalty
                chunk = jt.matmul(jt.nn.softmax(scores, dim=-1), value_for_attention.float32()).astype(value_for_attention.dtype)
                outputs.append(_detach_inference(chunk))
            output = jt.concat(outputs, dim=2)
        output = output.permute(0, 2, 1, 3).reshape(
            batch, query_length, self.config.num_attention_heads * head_dim
        )
        return _linear(output, self.weights, f"{prefix}.o_proj"), new_cache

    @staticmethod
    def _reserve_cache(cache: list[KVCache], capacity: int) -> None:
        """Grow each layer's GPU cache once per block, not once per token."""
        for layer in cache:
            if layer.capacity >= capacity:
                continue
            shape = list(layer.key.shape)
            shape[2] = capacity
            key = jt.zeros(shape, dtype=layer.key.dtype)
            value = jt.zeros(shape, dtype=layer.value.dtype)
            key[:, :, :layer.length, :] = layer.key
            value[:, :, :layer.length, :] = layer.value
            jt.sync([key, value])
            layer.key, layer.value = key.detach(), value.detach()

    def forward(
        self,
        inputs_embeds: jt.Var,
        cache: Optional[list[KVCache]] = None,
        start_position: int = 0,
    ) -> tuple[jt.Var, list[KVCache]]:
        batch, sequence_length, _ = inputs_embeds.shape
        if batch != 1:
            raise ValueError("Minimum Jittor inference currently supports batch_size=1.")
        if cache is not None and cache[0].length + sequence_length > cache[0].capacity:
            self._reserve_cache(cache, max(cache[0].capacity + 128, cache[0].length + sequence_length))
        positions = jt.arange(start_position, start_position + sequence_length).reshape(1, sequence_length)
        new_cache = []
        hidden_states = inputs_embeds
        for layer_index in range(self.config.num_hidden_layers):
            layer = f"model.language_model.layers.{layer_index}"
            residual = hidden_states
            hidden_states = _rms_norm(
                hidden_states, self.weights.tensor(f"{layer}.input_layernorm.weight"), self.config.rms_norm_eps
            )
            attended, cached = self._attention(hidden_states, layer_index, positions, None if cache is None else cache[layer_index])
            hidden_states = residual + attended
            residual = hidden_states
            hidden_states = _rms_norm(
                hidden_states, self.weights.tensor(f"{layer}.post_attention_layernorm.weight"), self.config.rms_norm_eps
            )
            mlp = _silu(_linear(hidden_states, self.weights, f"{layer}.mlp.gate_proj"))
            mlp = mlp * _linear(hidden_states, self.weights, f"{layer}.mlp.up_proj")
            hidden_states = residual + _linear(mlp, self.weights, f"{layer}.mlp.down_proj")

            # One explicit sync call per layer: hidden/K/V stay on the GPU
            # instead of making three NumPy round trips.
            jt.sync([hidden_states, cached.key, cached.value])
            hidden_states = hidden_states.detach()
            cached.key, cached.value = cached.key.detach(), cached.value.detach()
            new_cache.append(cached)
            if self.stream_layer_weights:
                self.weights.release_prefix(f"model.language_model.layers.{layer_index}.")
            elif hasattr(self.weights, "detach_prefix"):
                self.weights.detach_prefix(f"model.language_model.layers.{layer_index}.")
            if (layer_index + 1) % 6 == 0:
                jt.gc()
        if hasattr(self.weights, "detach_prefix"):
            self.weights.detach_prefix("model.language_model.embed_tokens.")
        hidden_states = _rms_norm(
            hidden_states, self.weights.tensor("model.language_model.norm.weight"), self.config.rms_norm_eps
        )
        return hidden_states, new_cache

    def logits(self, hidden_states: jt.Var) -> jt.Var:
        return jt.matmul(hidden_states, self.weights.tensor("model.language_model.embed_tokens.weight").transpose())

    def generate_greedy(
        self,
        inputs_embeds: jt.Var,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[set[int]] = None,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        seed: int = 42,
        seg_token_id: Optional[int] = None,
        max_generated_parts: int = 64,
    ) -> list[int]:
        print(f"[Generate] prefill {inputs_embeds.shape[1]} tokens", flush=True)
        hidden_states, cache = self.forward(inputs_embeds)
        self._reserve_cache(cache, cache[0].length + min(max_new_tokens, 128))
        generated = []
        rng = np.random.default_rng(seed)
        for step in range(max_new_tokens):
            scores = self.logits(hidden_states[:, -1:, :])
            if do_sample:
                values = scores.float32().numpy().reshape(-1).astype(np.float64) / temperature
                order = np.argsort(values)[::-1]
                if top_k > 0:
                    order = order[:top_k]
                probabilities = np.exp(values[order] - values[order].max())
                probabilities /= probabilities.sum()
                keep = np.cumsum(probabilities) - probabilities < top_p
                order, probabilities = order[keep], probabilities[keep]
                next_token = int(rng.choice(order, p=probabilities / probabilities.sum()))
            else:
                next_token = int(jt.argmax(scores, dim=-1)[0].numpy()[0, 0])
            generated.append(next_token)
            if len(generated) % 64 == 0:
                print(f"[Generate] {len(generated)} tokens", flush=True)
            if (eos_token_id is not None and next_token == eos_token_id) or (
                stop_token_ids is not None and next_token in stop_token_ids
            ):
                break
            if max_generated_parts > 0 and generated.count(seg_token_id) >= max_generated_parts:
                break
            hidden_states, cache = self.forward(self.embed(jt.array([[next_token]])).astype(inputs_embeds.dtype), cache, inputs_embeds.shape[1] + step)
        print(f"[Generate] finished with {len(generated)} tokens", flush=True)
        return generated
