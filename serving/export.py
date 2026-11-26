from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig, Qwen3ForCausalLM

from chained_transformer.modes import Mode
from serving.mixed import stack_any

ARCHITECTURE = "ChainedQwen3ForCausalLM"
TOKEN_TEXTS_FILE = "token_texts.json"
MLP_IMPLS = ("select", "loop", "triton")


@dataclass(frozen=True)
class TrackingSettings:
    num_modes: int
    prompt_mode: Mode
    token_texts_file: str | None
    mlp_impl: str
    record: bool


def tracking_settings(config: PretrainedConfig) -> TrackingSettings:
    surgery = getattr(config, "mode_surgery", None) or {}
    tracking = getattr(config, "mode_tracking", None) or {}
    variant = surgery.get("variant", "full")
    if variant != "full":
        raise ValueError(
            f"vLLM serves full per-mode MLPs; export the {variant!r} checkpoint with "
            "serving.export.export_for_vllm first"
        )
    impl = tracking.get("mlp_impl", "select")
    if impl not in MLP_IMPLS:
        raise ValueError(f"mode_tracking.mlp_impl must be one of {MLP_IMPLS}, got {impl!r}")
    return TrackingSettings(
        num_modes=int(surgery.get("num_modes", len(Mode))),
        prompt_mode=Mode(int(tracking.get("prompt_mode", Mode.THINK))),
        token_texts_file=tracking.get("token_texts_file"),
        mlp_impl=impl,
        record=bool(tracking.get("record", False)),
    )


def export_state_dict(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    state = {k: v for k, v in model.state_dict().items() if ".mlp." not in k}
    for i, layer in enumerate(model.model.layers):
        w = stack_any(layer.mlp)
        inter = w.intermediate
        for m in range(w.num_modes):
            prefix = f"model.layers.{i}.mlp.modes.{m}."
            state[prefix + "gate_proj.weight"] = w.gate_up[m, :inter]
            state[prefix + "up_proj.weight"] = w.gate_up[m, inter:]
            state[prefix + "down_proj.weight"] = w.down[m]
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    return {k: v.detach().to("cpu").contiguous().clone() for k, v in state.items()}


def export_config(
    model: Qwen3ForCausalLM,
    num_modes: int,
    prompt_mode: Mode,
    token_texts_file: str | None,
    mlp_impl: str,
) -> dict[str, Any]:
    cfg = model.config.to_dict()
    cfg["architectures"] = [ARCHITECTURE]
    surgery = dict(cfg.get("mode_surgery") or {})
    surgery["variant"] = "full"
    surgery["num_modes"] = num_modes
    cfg["mode_surgery"] = surgery
    tracking: dict[str, Any] = {"prompt_mode": int(prompt_mode), "mlp_impl": mlp_impl}
    if token_texts_file is not None:
        tracking["token_texts_file"] = token_texts_file
    cfg["mode_tracking"] = tracking
    return cfg


def export_for_vllm(
    model: Qwen3ForCausalLM,
    out_dir: Path,
    prompt_mode: Mode = Mode.THINK,
    token_texts: Sequence[str] | None = None,
    mlp_impl: str = "select",
    dtype: torch.dtype | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = export_state_dict(model)
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    num_modes = stack_any(model.model.layers[0].mlp).num_modes
    texts_file = TOKEN_TEXTS_FILE if token_texts is not None else None
    cfg = export_config(model, num_modes, prompt_mode, texts_file, mlp_impl)
    if dtype is not None:
        cfg["torch_dtype"] = str(dtype).removeprefix("torch.")
        cfg["dtype"] = cfg["torch_dtype"]
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if token_texts is not None:
        (out_dir / TOKEN_TEXTS_FILE).write_text(json.dumps(list(token_texts)) + "\n")
    return out_dir


def load_token_texts(model_dir: str | Path, settings: TrackingSettings) -> list[str] | None:
    if settings.token_texts_file is None:
        return None
    path = Path(model_dir) / settings.token_texts_file
    texts = json.loads(path.read_text())
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError(f"{path} must hold a JSON list of strings")
    return texts


def export_single_mode(
    model: Qwen3ForCausalLM, mode: int, out_dir: Path, dtype: torch.dtype | None = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in export_state_dict(model).items() if ".mlp." not in k}
    for i, layer in enumerate(model.model.layers):
        w = stack_any(layer.mlp)
        inter = w.intermediate
        prefix = f"model.layers.{i}.mlp."
        state[prefix + "gate_proj.weight"] = w.gate_up[mode, :inter].contiguous().clone()
        state[prefix + "up_proj.weight"] = w.gate_up[mode, inter:].contiguous().clone()
        state[prefix + "down_proj.weight"] = w.down[mode].contiguous().clone()
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    cfg = model.config.to_dict()
    cfg["architectures"] = ["Qwen3ForCausalLM"]
    cfg.pop("mode_surgery", None)
    cfg.pop("mode_tracking", None)
    if dtype is not None:
        cfg["torch_dtype"] = str(dtype).removeprefix("torch.")
        cfg["dtype"] = cfg["torch_dtype"]
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out_dir
