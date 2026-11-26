from __future__ import annotations

import copy

import pytest
import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from chained_transformer.checkpoint import convert
from chained_transformer.surgery import MoEMLP, SurgeryConfig, apply_surgery

NEW_TOKENS = 1024
PAD = 63
PROMPTS = [[0, 1, 7, 8, 9, 10, 11, 12], [0, 1, 20, 21], [0, 1, 30, 31, 32, 33]]
Run = tuple[torch.Tensor, torch.Tensor]


def _model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=NEW_TOKENS + 64,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, PROMPTS))
    ids = torch.tensor([[PAD] * (width - len(p)) + p for p in PROMPTS])
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in PROMPTS])
    return ids, mask


def _greedy(model: Qwen3ForCausalLM, use_cache: bool) -> Run:
    ids, mask = _batch()
    cfg = GenerationConfig(
        max_new_tokens=NEW_TOKENS,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=PAD,
        use_cache=use_cache,
        output_logits=True,
        return_dict_in_generate=True,
    )
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
    assert out.sequences.shape == (len(PROMPTS), ids.shape[1] + NEW_TOKENS)
    return out.sequences, torch.stack(out.logits)


def _assert_same(got: Run, want: Run) -> None:
    assert got[1].shape == want[1].shape
    differ = (got[1] != want[1]).flatten(1).any(dim=1).nonzero()
    assert not differ.numel(), f"logits first differ at new token {int(differ[0])}"
    assert torch.equal(got[0], want[0])


def _first_experts(model: Qwen3ForCausalLM, ids: torch.Tensor) -> set[int]:
    moe = model.model.layers[0].mlp
    assert isinstance(moe, MoEMLP) and not moe.training
    seen: list[torch.Tensor] = []
    handle = moe.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    _, _, chosen = moe.route(seen[0].reshape(-1, seen[0].shape[-1]))
    return set(chosen[:, 0].tolist())


@pytest.mark.parametrize("use_cache", [True, False])
def test_converted_moe_matches_base_over_long_greedy_runs_in_fp32(use_cache: bool) -> None:
    base = _model()
    moe = convert(copy.deepcopy(base), SurgeryConfig(variant="moe"), "tiny").eval()
    want = _greedy(base, use_cache)
    got = _greedy(moe, use_cache)
    _assert_same(got, want)
    assert len(_first_experts(moe, want[0])) > 1


def test_moe_matches_base_over_a_long_greedy_run_in_bf16() -> None:
    base = _model().bfloat16()
    moe = copy.deepcopy(base)
    apply_surgery(moe, SurgeryConfig(variant="moe"))
    _assert_same(_greedy(moe, use_cache=True), _greedy(base, use_cache=True))
