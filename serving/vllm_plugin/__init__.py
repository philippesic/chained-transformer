from __future__ import annotations

ARCHITECTURE = "ChainedQwen3ForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE, "serving.vllm_plugin.model:ChainedQwen3ForCausalLM"
        )
