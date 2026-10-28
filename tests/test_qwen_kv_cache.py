"""GPU regressions for the inference-only Qwen KV cache."""

import unittest

import jittor as jt
import numpy as np

from partllm_jittor.qwen3_text import JittorQwen3Text, Qwen3TextConfig


class TinyWeights:
    def __init__(self):
        rng = np.random.default_rng(7)
        self.values = {}

        def add(name, shape, scale=0.05):
            self.values[name] = jt.array(rng.normal(0, scale, shape).astype(np.float32))

        add("model.language_model.embed_tokens.weight", (32, 8))
        add("model.language_model.norm.weight", (8,))
        self.values["model.language_model.norm.weight"] = jt.ones((8,))
        for index in range(2):
            prefix = f"model.language_model.layers.{index}"
            for name in ("input_layernorm", "post_attention_layernorm"):
                self.values[f"{prefix}.{name}.weight"] = jt.ones((8,))
            attention = f"{prefix}.self_attn"
            for name, shape in (("q_proj", (8, 8)), ("k_proj", (4, 8)),
                                ("v_proj", (4, 8)), ("o_proj", (8, 8))):
                add(f"{attention}.{name}.weight", shape)
            self.values[f"{attention}.q_norm.weight"] = jt.ones((4,))
            self.values[f"{attention}.k_norm.weight"] = jt.ones((4,))
            for name, shape in (("gate_proj", (16, 8)), ("up_proj", (16, 8)),
                                ("down_proj", (8, 16))):
                add(f"{prefix}.mlp.{name}.weight", shape)

    def tensor(self, name):
        return self.values[name]

    def detach_prefix(self, prefix):
        for name, value in self.values.items():
            if name.startswith(prefix):
                self.values[name] = value.detach()


class QwenKVCacheTest(unittest.TestCase):
    def test_reserved_and_grown_cache_match_full_causal_forward(self):
        jt.flags.use_cuda = 1
        config = Qwen3TextConfig(
            hidden_size=8, intermediate_size=16, num_attention_heads=2,
            num_key_value_heads=1, num_hidden_layers=2, head_dim=4,
            vocab_size=32, rms_norm_eps=1e-6, rope_theta=10000.0,
        )
        model = JittorQwen3Text(config, TinyWeights(), attention_chunk_size=3)
        inputs = jt.array(np.random.default_rng(11).normal(0, 0.1, (1, 7, 8)).astype(np.float32))
        with jt.no_grad():
            full_hidden, full_cache = model.forward(inputs)
            _, cache = model.forward(inputs[:, :4, :])
            model._reserve_cache(cache, 5)
            self.assertEqual(cache[0].capacity, 5)
            for position in range(4, 7):
                incremental_hidden, cache = model.forward(
                    inputs[:, position:position + 1, :], cache, start_position=position
                )
            self.assertGreater(cache[0].capacity, 5)
            for expected, actual in zip(full_cache, cache):
                self.assertEqual(actual.length, 7)
                np.testing.assert_allclose(
                    actual.key[:, :, :7, :].numpy(), expected.key.numpy(), rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    actual.value[:, :, :7, :].numpy(), expected.value.numpy(), rtol=1e-4, atol=1e-4
                )
            np.testing.assert_allclose(
                incremental_hidden.numpy(), full_hidden[:, -1:, :].numpy(), rtol=1e-4, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main()
