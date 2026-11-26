from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from chained_transformer.modes import Mode
from chained_transformer.surgery import LoRAModeMLP, Routing, SurgeryConfig, apply_surgery
from serving.export import (
    ARCHITECTURE,
    export_for_vllm,
    export_single_mode,
    load_token_texts,
    tracking_settings,
)
from serving.mixed import stack_lora_mode_mlp
from serving.tiny import tiny_chained_model, tiny_config, tiny_token_texts


def reload(out: Path) -> tuple[Qwen3ForCausalLM, Routing]:
    cfg = json.loads((out / "config.json").read_text())
    model = Qwen3ForCausalLM(Qwen3Config(**cfg)).float().eval()
    routing = apply_surgery(
        model, SurgeryConfig(variant="full", num_modes=cfg["mode_surgery"]["num_modes"])
    )
    missing, unexpected = model.load_state_dict(load_file(str(out / "model.safetensors")), False)
    assert unexpected == []
    assert set(missing) <= {"lm_head.weight"}
    model.tie_weights()
    return model, routing


def test_full_export_round_trips(tmp_path: Path) -> None:
    model, routing = tiny_chained_model()
    out = export_for_vllm(model, tmp_path, Mode.ANSWER, tiny_token_texts())
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == [ARCHITECTURE]
    settings = tracking_settings(Qwen3Config(**cfg))
    assert settings.prompt_mode == Mode.ANSWER
    assert settings.num_modes == 3
    assert load_token_texts(out, settings) == tiny_token_texts()
    again, again_routing = reload(out)
    ids = torch.randint(0, 64, (2, 9))
    modes = torch.randint(0, 3, (2, 9))
    routing.set(modes)
    with torch.no_grad():
        want = model(ids).logits
    again_routing.set(modes)
    with torch.no_grad():
        got = again(ids).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_lora_export_merges_each_mode(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).float().eval()
    apply_surgery(model, SurgeryConfig(variant="lora", lora_rank=4))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_b" in name:
                p.normal_(0, 0.05)
    out = export_for_vllm(model, tmp_path)
    state = load_file(str(out / "model.safetensors"))
    mlp = model.model.layers[1].mlp
    assert isinstance(mlp, LoRAModeMLP)
    w = stack_lora_mode_mlp(mlp)
    inter = w.intermediate
    for m in range(3):
        prefix = f"model.layers.1.mlp.modes.{m}."
        torch.testing.assert_close(state[prefix + "gate_proj.weight"], w.gate_up[m, :inter])
        torch.testing.assert_close(state[prefix + "up_proj.weight"], w.gate_up[m, inter:])
        torch.testing.assert_close(state[prefix + "down_proj.weight"], w.down[m])
    assert not any("lora" in k for k in state)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["mode_surgery"]["variant"] == "full"


def test_settings_reject_unexported_variants() -> None:
    cfg = tiny_config()
    cfg.update({"mode_surgery": {"variant": "lora", "num_modes": 3}})
    with pytest.raises(ValueError, match="export"):
        tracking_settings(cfg)
    cfg.update({"mode_surgery": {"variant": "full"}, "mode_tracking": {"mlp_impl": "fast"}})
    with pytest.raises(ValueError, match="mlp_impl"):
        tracking_settings(cfg)


def test_single_mode_export_is_a_plain_qwen3(tmp_path: Path) -> None:
    model, routing = tiny_chained_model()
    out = export_single_mode(model, 2, tmp_path)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3ForCausalLM"]
    assert "mode_surgery" not in cfg
    plain = Qwen3ForCausalLM(Qwen3Config(**cfg)).float().eval()
    missing, unexpected = plain.load_state_dict(load_file(str(out / "model.safetensors")), False)
    assert unexpected == [] and set(missing) <= {"lm_head.weight"}
    plain.tie_weights()
    ids = torch.randint(0, 64, (1, 11))
    routing.set(torch.full((1, 11), 2))
    with torch.no_grad():
        torch.testing.assert_close(plain(ids).logits, model(ids).logits, rtol=0, atol=0)
