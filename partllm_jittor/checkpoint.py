"""Checkpoint configuration and NumPy weight access."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open


DECODER_PREFIX = "model.mask_decoder."
ENCODER_CHANNELS = (54, 108, 216, 432, 576)


def load_config(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))


def load_weight_map(model_dir: str | Path) -> dict[str, str]:
    root = Path(model_dir).resolve()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="np") as handle:
            mapping = {key: "model.safetensors" for key in handle.keys()}
    for shard in set(mapping.values()):
        path = (root / shard).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid checkpoint shard: {shard}")
    return mapping


@dataclass(frozen=True)
class MaskDecoderConfig:
    transformer_depth: int
    transformer_embedding_dim: int
    transformer_num_heads: int
    transformer_mlp_dim: int
    transformer_attention_downsample_rate: int
    mask_decoder_transformer_dim: int
    mask_decoder_embedding_input_dim: int
    utonia_enc_channels: tuple = ENCODER_CHANNELS

    @classmethod
    def from_hf_config(cls, config_path: str | Path) -> "MaskDecoderConfig":
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        values = config.get("mask_decoder_config", config)
        required = (
            "transformer_depth", "transformer_embedding_dim", "transformer_num_heads",
            "transformer_mlp_dim", "transformer_attention_downsample_rate",
            "mask_decoder_transformer_dim", "mask_decoder_embedding_input_dim",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"Missing mask decoder configuration: {', '.join(missing)}")
        result = cls(
            **{name: int(values[name]) for name in required},
            utonia_enc_channels=tuple(values.get("utonia_enc_channels", ENCODER_CHANNELS)),
        )
        if any(getattr(result, name) <= 0 for name in required):
            raise ValueError("Mask decoder dimensions and depth must be positive")
        if result.transformer_embedding_dim != result.mask_decoder_transformer_dim:
            raise ValueError("Mask decoder transformer dimensions must match")
        width = result.transformer_embedding_dim
        rate = result.transformer_attention_downsample_rate
        if width % rate or (width // rate) % result.transformer_num_heads or width % 2:
            raise ValueError("Mask decoder attention dimensions are incompatible")
        return result


def read_decoder_weights(model_dir: str | Path) -> dict[str, np.ndarray]:
    root = Path(model_dir)
    mapping = load_weight_map(root)
    selected = {key: shard for key, shard in mapping.items() if key.startswith(DECODER_PREFIX)}
    if not selected:
        raise ValueError("Checkpoint contains no PartLLM mask decoder weights")
    result = {}
    for shard in sorted(set(selected.values())):
        with safe_open(root / shard, framework="np", backend="pread") as handle:
            for key, source in selected.items():
                if source == shard:
                    result[key.removeprefix(DECODER_PREFIX)] = np.asarray(
                        handle.get_tensor(key), dtype=np.float32
                    ).copy()
    return result


def validate_checkpoint(model_dir: str | Path, config_dir: str | Path | None = None) -> dict:
    root = Path(model_dir)
    config_root = Path(config_dir) if config_dir is not None else root
    config = load_config(config_root)
    decoder = MaskDecoderConfig.from_hf_config(config_root / "config.json")
    text = config["text_config"]
    if config.get("decoder_mode", "joint") != "joint":
        raise ValueError("Jittor inference requires a joint mask decoder")
    if config.get("point_cloud_encoder_type") != "utonia":
        raise ValueError("Jittor inference requires the Utonia point encoder")
    if tuple(config.get("utonia_enc_channels", decoder.utonia_enc_channels)) != ENCODER_CHANNELS:
        raise ValueError("Unsupported Utonia encoder channels")
    if decoder.utonia_enc_channels != ENCODER_CHANNELS:
        raise ValueError("Unsupported mask decoder Utonia channels")
    if not text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        raise ValueError("Jittor inference requires tied Qwen input/output embeddings")
    if text.get("attention_bias", False) or text.get("hidden_act", "silu") != "silu":
        raise ValueError("Unsupported Qwen attention bias or activation")
    hidden = int(text["hidden_size"])
    if decoder.mask_decoder_embedding_input_dim != hidden:
        raise ValueError("Qwen hidden size and mask decoder input dimension differ")
    point_tokens = int(config.get("utonia_num_tokens", 4096))
    if point_tokens <= 0:
        raise ValueError("utonia_num_tokens must be positive")
    width = decoder.transformer_embedding_dim
    expected = {
        "model.language_model.embed_tokens.weight": (int(text["vocab_size"]), hidden),
        "model.utonia_proj.weight": (hidden, sum(ENCODER_CHANNELS)),
        f"{DECODER_PREFIX}pe_layer.positional_encoding_gaussian_matrix": (3, width // 2),
        f"{DECODER_PREFIX}mask_decoder.encoder_mapper.layers.0.weight": (width, hidden),
        f"{DECODER_PREFIX}mask_decoder.seg_token_mapper.layers.0.weight": (width, hidden),
        f"{DECODER_PREFIX}mask_decoder.output_upscaling_with_fusion.0.weight":
            (width, width + sum(ENCODER_CHANNELS)),
    }
    for index in range(decoder.transformer_depth):
        layer = f"{DECODER_PREFIX}mask_decoder.transformer.layers.{index}"
        expected[f"{layer}.self_attn.q_proj.weight"] = (width, width)
        attention_width = width // decoder.transformer_attention_downsample_rate
        expected[f"{layer}.cross_attn_token_to_image.q_proj.weight"] = (attention_width, width)
        expected[f"{layer}.mlp.lin1.weight"] = (decoder.transformer_mlp_dim, width)
    mapping = load_weight_map(root)
    with ExitStack() as stack:
        handles = {}
        for key, shape in expected.items():
            if key not in mapping:
                raise ValueError(f"Missing checkpoint tensor: {key}")
            shard = mapping[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(root / shard, framework="np"))
            actual = tuple(handles[shard].get_slice(key).get_shape())
            if actual != shape:
                raise ValueError(f"Checkpoint shape mismatch for {key}: {actual}, expected {shape}")
    return {
        "point_tokens": point_tokens,
        "decoder_depth": decoder.transformer_depth,
        "decoder_width": width,
        "attention_downsample_rate": decoder.transformer_attention_downsample_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_checkpoint(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
