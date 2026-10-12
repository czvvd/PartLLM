"""Prompt construction and Jittor Qwen/decoder orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import jittor as jt
import numpy as np

os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer

from .qwen3_text import JittorQwen3Text


@dataclass
class GenerationResult:
    text: str
    generated_ids: list[int]
    full_ids: np.ndarray
    token_hidden_states: np.ndarray


class PartSegPrompt:
    """Expands one point-cloud placeholder and injects its Jittor embeddings."""

    def __init__(self, tokenizer_dir: str | Path, point_token_count: int = 4096):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True, fix_mistral_regex=True
        )
        self.point_token_count = point_token_count
        self.point_token_id = self.tokenizer.convert_tokens_to_ids("<|point_cloud_pad|>")
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids("<|SEG|>")
        self.bg_token_id = self.tokenizer.convert_tokens_to_ids("<|BG|>")
        for name, token_id in (("<|point_cloud_pad|>", self.point_token_id), ("<|SEG|>", self.seg_token_id), ("<|BG|>", self.bg_token_id)):
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer is missing {name}")

    def input_ids(self, prompt: str) -> np.ndarray:


        prompt = prompt.replace("<point_cloud>", "")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "point_cloud"}]}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if ids.count(self.point_token_id) != 1:
            raise ValueError("Prompt template must contain exactly one point-cloud placeholder before expansion.")
        expanded = []
        for token_id in ids:
            expanded.extend([token_id] * self.point_token_count if token_id == self.point_token_id else [token_id])
        return np.asarray(expanded, dtype=np.int32)[None, :]

    def inject_point_embeddings(self, model: JittorQwen3Text, input_ids: np.ndarray, point_embeddings: np.ndarray) -> jt.Var:
        if point_embeddings.shape != (self.point_token_count, model.config.hidden_size):
            raise ValueError(
                f"point_embeddings must be [{self.point_token_count}, {model.config.hidden_size}], got {point_embeddings.shape}"
            )
        positions = np.flatnonzero(input_ids[0] == self.point_token_id)
        if len(positions) != self.point_token_count:
            raise ValueError(f"Expected {self.point_token_count} expanded point tokens, found {len(positions)}")
        embedded = model.embed(jt.array(input_ids))


        embedding_dtype = embedded.dtype
        token_embeddings = embedded.float32().numpy()
        token_embeddings[0, positions, :] = point_embeddings.astype(np.float32, copy=False)
        return jt.array(token_embeddings).astype(embedding_dtype)

    def generate(
        self,
        model: JittorQwen3Text,
        prompt: str,
        point_embeddings: np.ndarray,
        max_new_tokens: int = 256,
        auto_add_bg: bool = True,
        **generation_options,
    ) -> GenerationResult:
        input_ids = self.input_ids(prompt)
        inputs_embeds = self.inject_point_embeddings(model, input_ids, point_embeddings)


        generated = model.generate_greedy(
            inputs_embeds,
            max_new_tokens,
            self.tokenizer.eos_token_id,
            stop_token_ids={self.bg_token_id},
            seg_token_id=self.seg_token_id,
            **generation_options,
        )


        if self.seg_token_id not in generated:
            raise RuntimeError("Generation produced no part mask tokens. Increase --max_new_tokens or check the checkpoint.")
        if self.bg_token_id not in generated:
            if not auto_add_bg:
                raise RuntimeError("Generation ended without a background query")
            generated.append(self.bg_token_id)
        full_ids = np.concatenate([input_ids, np.asarray(generated, dtype=np.int32)[None, :]], axis=1)

        final_embeddings = self.inject_point_embeddings(model, full_ids, point_embeddings)
        hidden, _ = model.forward(final_embeddings)
        token_positions = input_ids.shape[1] + np.flatnonzero(np.isin(generated, [self.seg_token_id, self.bg_token_id]))
        if len(token_positions) == 0:
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            raise RuntimeError(
                "Generation produced no <|SEG|> or <|BG|> tokens; joint decoder cannot run. "
                f"generated_ids={generated}; text={text!r}"
            )
        token_hidden = hidden.float32().numpy()[:, token_positions, :]
        return GenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            generated_ids=generated,
            full_ids=full_ids,
            token_hidden_states=token_hidden,
        )
