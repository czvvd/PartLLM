"""vLLM general plugin entry point."""

from .constants import ARCHITECTURE_NAME


def register() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE_NAME not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE_NAME,
            "partllm_vllm.model:PartLLMQwen3VLForConditionalGeneration",
        )
