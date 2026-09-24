"""Point-embedding adapter for vLLM's native Qwen3-VL model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    EmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .constants import HIDDEN_SIZE, NUM_POINT_TOKENS
from .weights import iter_supported_weights

POINT_MODALITY = "point"
POINT_PLACEHOLDER = "<|point_cloud_start|><|point_cloud_pad|><|point_cloud_end|>"


class PointDataParser(MultiModalDataParser):
    """Pass through embeddings already produced by Utonia."""

    def _parse_point_data(self, data: object) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            raise ValueError("point embeddings must not be empty")
        if isinstance(data, torch.Tensor):
            if data.ndim == 2:
                data = [data]
            elif data.ndim != 3:
                raise ValueError(
                    "point embedding tensor must have shape [L,H] or [B,L,H], "
                    f"got {tuple(data.shape)}"
                )
        elif isinstance(data, list):
            if not data:
                return None
            if not all(isinstance(item, torch.Tensor) and item.ndim == 2 for item in data):
                raise ValueError("point embedding lists may contain only [L,H] tensors")
        else:
            raise TypeError(f"unsupported point embedding type: {type(data)}")
        return EmbeddingItems(data, POINT_MODALITY)

    def _get_subparsers(self):
        return {POINT_MODALITY: self._parse_point_data}


class PartLLMPointProcessingInfo(Qwen3VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {POINT_MODALITY: 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        config = self.get_hf_config()
        return {POINT_MODALITY: int(getattr(config, "utonia_num_tokens", NUM_POINT_TOKENS))}


class PartLLMPointDummyInputsBuilder(BaseDummyInputsBuilder[PartLLMPointProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return POINT_PLACEHOLDER * mm_counts.get(POINT_MODALITY, 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len, mm_options
        count = mm_counts.get(POINT_MODALITY, 0)
        if count == 0:
            return {}
        config = self.info.get_hf_config()
        num_tokens = int(getattr(config, "utonia_num_tokens", NUM_POINT_TOKENS))
        hidden_size = int(config.text_config.hidden_size)
        return {
            POINT_MODALITY: torch.zeros(
                (count, num_tokens, hidden_size),
                dtype=self.info.ctx.model_config.dtype,
            )
        }


class PartLLMPointMultiModalProcessor(BaseMultiModalProcessor[PartLLMPointProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return PointDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:

        if mm_data:
            raise ValueError(f"received unparsed multimodal data: {set(mm_data)}")
        del mm_kwargs
        tokenizer_kwargs = dict(tok_kwargs)
        tokenizer_kwargs.pop("return_tensors", None)
        encoded = self.info.get_tokenizer()(
            prompt,
            return_tensors="pt",
            **tokenizer_kwargs,
        )
        return BatchFeature(dict(encoded))

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        return {"point_embeds": MultiModalFieldConfig.batched(POINT_MODALITY)}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs
        config = self.info.get_hf_config()
        point_token_id = int(config.point_cloud_token_id)
        hidden_size = int(config.text_config.hidden_size)

        def replacement(item_idx: int) -> list[int]:
            item = out_mm_kwargs[POINT_MODALITY][item_idx]
            point_embeds = item["point_embeds"].data
            if not isinstance(point_embeds, torch.Tensor) or point_embeds.ndim != 2:
                raise ValueError("each point embedding must be an [L,H] tensor")
            if point_embeds.shape[-1] != hidden_size:
                raise ValueError(
                    f"point hidden size must be {hidden_size}, got {point_embeds.shape[-1]}"
                )
            return [point_token_id] * point_embeds.shape[0]

        return [
            PromptReplacement(
                modality=POINT_MODALITY,
                target=[point_token_id],
                replacement=replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    PartLLMPointMultiModalProcessor,
    info=PartLLMPointProcessingInfo,
    dummy_inputs=PartLLMPointDummyInputsBuilder,
)
class PartLLMQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Hold only the Qwen3-VL LLM; Utonia and the mask decoder stay in PyTorch."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.visual is not None:
            raise RuntimeError("the point-only adapter requires image=0 and video=0")
        self.use_deepstack = False
        self.deepstack_num_level = 0
        self.deepstack_input_embeds = None
        self._hidden_capture_targets: dict[int, int] | None = None
        self._hidden_capture_ids: list[torch.Tensor] = []
        self._hidden_capture_positions: list[torch.Tensor] = []
        self._hidden_capture_values: list[torch.Tensor] = []

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith(POINT_MODALITY):
            return POINT_PLACEHOLDER
        raise ValueError("only the point modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        point_embeds = kwargs.get("point_embeds")
        if point_embeds is None:
            return None
        if isinstance(point_embeds, torch.Tensor):
            if point_embeds.ndim == 2:
                items = (point_embeds,)
            elif point_embeds.ndim == 3:
                items = tuple(point_embeds.unbind(0))
            else:
                raise ValueError(
                    "point_embeds must have shape [L,H] or [B,L,H], "
                    f"got {tuple(point_embeds.shape)}"
                )
        elif isinstance(point_embeds, list):
            items = tuple(point_embeds)
        else:
            raise TypeError(f"unsupported point_embeds type: {type(point_embeds)}")

        hidden_size = int(self.config.text_config.hidden_size)
        target_dtype = self.language_model.model.embed_tokens.weight.dtype
        outputs: list[torch.Tensor] = []
        for item in items:
            if not isinstance(item, torch.Tensor) or item.ndim != 2:
                raise ValueError("each point embedding must be an [L,H] tensor")
            if item.shape[-1] != hidden_size:
                raise ValueError(
                    f"point hidden size must be {hidden_size}, got {item.shape[-1]}"
                )
            outputs.append(item.to(dtype=target_dtype))
        return tuple(outputs)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        unsupported = {
            feature.modality for feature in mm_features if feature.modality != POINT_MODALITY
        }
        if unsupported:
            raise ValueError(f"unsupported modality: {unsupported}")
        positions = (
            torch.arange(len(input_tokens), dtype=torch.long)
            .unsqueeze(0)
            .expand(3, -1)
            .contiguous()
        )
        return positions, 0

    def begin_hidden_capture(
        self,
        positions: tuple[int, ...],
        token_ids: tuple[int, ...],
    ) -> dict[str, object]:
        """Begin final-layer hidden-state capture for selected tokens."""

        if len(positions) != len(token_ids):
            raise ValueError("hidden capture positions and token_ids have different lengths")
        self._hidden_capture_targets = {
            int(position): int(token_id)
            for position, token_id in zip(positions, token_ids, strict=True)
        }
        self._hidden_capture_ids.clear()
        self._hidden_capture_positions.clear()
        self._hidden_capture_values.clear()
        return {"capture": "started", "positions": tuple(self._hidden_capture_targets)}

    def drain_hidden_capture(self) -> dict[str, torch.Tensor]:
        """Stop capture and move the selected hidden states to CPU."""

        self._hidden_capture_targets = None
        if self._hidden_capture_values:
            token_ids = torch.cat(self._hidden_capture_ids)
            positions = torch.cat(self._hidden_capture_positions)
            hidden_states = torch.cat(self._hidden_capture_values)
        else:
            token_ids = torch.empty(0, dtype=torch.long)
            positions = torch.empty(0, dtype=torch.long)
            hidden_states = torch.empty((0, HIDDEN_SIZE), dtype=torch.bfloat16)
        self._hidden_capture_ids.clear()
        self._hidden_capture_positions.clear()
        self._hidden_capture_values.clear()
        return {
            "token_ids": token_ids,
            "positions": positions,
            "hidden_states": hidden_states,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Any:
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        capture_targets = self._hidden_capture_targets
        if capture_targets is None or not isinstance(hidden_states, torch.Tensor):
            return hidden_states

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if positions.ndim == 1:
            flat_positions = positions
        else:
            flat_positions = positions.reshape(-1, positions.shape[-1])[0]
        if flat_positions.shape[0] != flat_hidden.shape[0]:
            raise RuntimeError(
                "hidden capture positions and hidden states have different token counts: "
                f"{flat_positions.shape[0]} != {flat_hidden.shape[0]}"
            )
        selected = torch.zeros_like(flat_positions, dtype=torch.bool)
        for target_position in capture_targets:
            selected |= flat_positions == target_position
        if not bool(selected.any()):
            return hidden_states

        selected_positions = flat_positions[selected].detach().to("cpu")
        selected_ids = torch.tensor(
            [capture_targets[int(position)] for position in selected_positions.tolist()],
            dtype=torch.long,
        )
        self._hidden_capture_ids.append(selected_ids)
        self._hidden_capture_positions.append(selected_positions)
        self._hidden_capture_values.append(flat_hidden[selected].detach().to("cpu"))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(iter_supported_weights(weights))
