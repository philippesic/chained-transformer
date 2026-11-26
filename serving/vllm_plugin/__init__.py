from __future__ import annotations

ARCHITECTURE = "ChainedQwen3ForCausalLM"
MODEL_STATE_MODULE = "vllm.v1.worker.gpu.model_states.default"


def register() -> None:
    import importlib

    import vllm
    from vllm import ModelRegistry

    try:
        importlib.import_module(MODEL_STATE_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"serving.vllm_plugin was built on vLLM 0.34.0's Model Runner V2, but vLLM "
            f"{vllm.__version__} has no {MODEL_STATE_MODULE}: {exc}"
        ) from exc
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE, "serving.vllm_plugin.model:ChainedQwen3ForCausalLM"
        )
