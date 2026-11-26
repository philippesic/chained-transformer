from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen3ForCausalLM

from chained_transformer.checkpoint import (
    ChainedQwen3ForCausalLM,
    chained_config,
    convert,
    is_chained,
    load_checkpoint,
    mode_slice,
    plan,
    save_checkpoint,
    shape_config,
)
from chained_transformer.generate import TokenTexts, mode_tracking
from chained_transformer.modes import MODE_NAMES
from chained_transformer.surgery import LoRALinear, MoEMLP, SurgeryConfig, Variant

VARIANTS: list[Variant] = ["full", "lora", "moe"]
SPECIAL = {0: "<|im_start|>", 1: "assistant\n", 2: "</think>", 3: "```\n", 4: "\n", 5: "<|im_end|>"}


def _ids(seed: int, shape: tuple[int, int], high: int) -> torch.Tensor:
    return torch.randint(0, high, shape, generator=torch.Generator().manual_seed(seed))


def _convert(tiny_model: Qwen3ForCausalLM, variant: Variant) -> ChainedQwen3ForCausalLM:
    surgery = SurgeryConfig(variant=variant, lora_rank=4)
    return convert(copy.deepcopy(tiny_model), surgery, "tiny").eval()


@pytest.mark.parametrize("variant", VARIANTS)
def test_conversion_reproduces_base_logits(tiny_model: Qwen3ForCausalLM, variant: Variant) -> None:
    model = _convert(tiny_model, variant)
    ids = _ids(1, (2, 12), 64)
    with torch.no_grad():
        want = tiny_model(input_ids=ids).logits
        for mode in range(len(MODE_NAMES)):
            model.routing.set(torch.full((2, 12), mode))
            got = model(input_ids=ids).logits
            if variant == "moe":
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
            else:
                assert torch.equal(got, want)
        model.routing.set(_ids(2, (2, 12), 3))
        torch.testing.assert_close(model(input_ids=ids).logits, want, rtol=1e-5, atol=1e-5)


def test_conversion_keeps_ties_buffers_and_fresh_adapters(tiny_model: Qwen3ForCausalLM) -> None:
    lora = _convert(tiny_model, "lora")
    assert lora.lm_head.weight is lora.model.embed_tokens.weight
    assert not any(t.is_meta for t in [*lora.parameters(), *lora.buffers()])
    adapters = [m for m in lora.modules() if isinstance(m, LoRALinear)]
    assert adapters
    assert all(not b.detach().any() for m in adapters for b in m.lora_b)
    assert all(a.detach().any() for m in adapters for a in m.lora_a)
    moe = _convert(tiny_model, "moe")
    routers = [m.router.weight.detach() for m in moe.modules() if isinstance(m, MoEMLP)]
    assert routers and all(r.any() for r in routers)


@pytest.mark.parametrize("variant", VARIANTS)
def test_round_trip_is_bitwise(
    tmp_path: Path, tiny_model: Qwen3ForCausalLM, variant: Variant
) -> None:
    model = _convert(tiny_model, variant)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    save_checkpoint(model, tmp_path, extra={"step": 3})
    assert is_chained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "chained_qwen3"
    assert config["mode_surgery"]["variant"] == variant
    assert config["mode_names"] == list(MODE_NAMES)
    assert config["mode_switch"]["think_close"] == "</think>"
    assert json.loads((tmp_path / "training_state.json").read_text()) == {"step": 3}
    built = load_checkpoint(tmp_path, dtype=torch.float32)
    loaded = built.model
    assert isinstance(loaded, ChainedQwen3ForCausalLM) and built.routing is loaded.routing
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    saved, reread = model.state_dict(), loaded.state_dict()
    assert saved.keys() == reread.keys()
    assert all(torch.equal(saved[k], reread[k]) for k in saved)
    ids = _ids(3, (1, 10), 64)
    modes = _ids(4, (1, 10), 3)
    with torch.no_grad():
        model.routing.set(modes)
        loaded.eval().routing.set(modes)
        assert torch.equal(model(input_ids=ids).logits, loaded(input_ids=ids).logits)


def test_auto_classes_load_chained_and_plain(tmp_path: Path, tiny_model: Qwen3ForCausalLM) -> None:
    save_checkpoint(_convert(tiny_model, "full"), tmp_path / "chained")
    assert isinstance(
        AutoModelForCausalLM.from_pretrained(tmp_path / "chained"), ChainedQwen3ForCausalLM
    )
    tiny_model.save_pretrained(tmp_path / "plain")
    plain = load_checkpoint(tmp_path / "plain", dtype=torch.float32)
    assert plain.routing is None and not is_chained(tmp_path / "plain")


def test_loaded_checkpoint_generates_with_mode_tracking(
    tmp_path: Path, tiny_model: Qwen3ForCausalLM
) -> None:
    with torch.no_grad():
        tiny_model.lm_head.weight[2].mul_(3.0)
        tiny_model.lm_head.weight[3].mul_(3.0)
    save_checkpoint(_convert(tiny_model, "full"), tmp_path)
    built = load_checkpoint(tmp_path, dtype=torch.float32)
    assert built.routing is not None
    texts = TokenTexts([SPECIAL.get(i, f"w{i} ") for i in range(64)])
    prompt = torch.tensor([[0, 1, 7, 8, 9]])
    cfg = GenerationConfig(max_new_tokens=20, do_sample=False, eos_token_id=None, pad_token_id=63)
    with torch.no_grad(), mode_tracking(built.model.eval(), built.routing, texts) as hook:
        got = built.model.generate(prompt, generation_config=cfg)
    want = tiny_model.generate(prompt, generation_config=cfg)
    assert torch.equal(got, want)
    assert hook.trace[0] == texts.modes(got[0].tolist())[:-1]


def test_mode_slice_holds_only_that_mode(tiny_model: Qwen3ForCausalLM) -> None:
    full = mode_slice(_convert(tiny_model, "full"), 1)
    assert full and all(".mlp.modes.1." in n for n in full)
    lora = mode_slice(_convert(tiny_model, "lora"), 2)
    assert lora and all(n.endswith((".lora_a.2", ".lora_b.2")) for n in lora)


def test_wrong_mode_count_is_rejected(tiny_model: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError, match="num_modes"):
        chained_config(tiny_model.config, SurgeryConfig(num_modes=4))


def test_recorded_shapes_reproduce_published_parameter_counts(tmp_path: Path) -> None:
    report = plan("Qwen/Qwen3-0.6B", tmp_path)
    assert report["base"]["total"] == 596_049_920
    assert report["variants"]["full"]["total"] == 1_124_532_224
    assert report["variants"]["lora"]["total"] == 662_110_208
    assert report["variants"]["moe"]["total"] == 1_124_618_240
    written = json.loads((tmp_path / "full" / "config.json").read_text())
    assert written["model_type"] == "chained_qwen3" and written["hidden_size"] == 1024
    big = shape_config("Qwen/Qwen3-1.7B")
    assert big.hidden_size == 2048 and big.intermediate_size == 6144
