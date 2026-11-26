from __future__ import annotations

import copy

import pytest
import torch
from transformers import GenerationConfig, Qwen3ForCausalLM

from chained_transformer.generate import mode_tracking
from chained_transformer.modes import Mode
from chained_transformer.surgery import (
    LoRAModeMLP,
    ModeMLP,
    Routing,
    SurgeryConfig,
    apply_surgery,
)
from serving.mixed import (
    IMPLS,
    StackedWeights,
    align_by_mode,
    batched_greedy_decode,
    mode_mlp_loop,
    padded_length,
    stack_lora_mode_mlp,
    stack_mode_mlp,
    use_grouped_mlps,
)
from serving.tiny import (
    TINY_PAD,
    TINY_PROMPTS,
    collapse_modes,
    tiny_chained_model,
    tiny_texts,
)


def random_weights(seed: int, modes: int, hidden: int, inter: int) -> StackedWeights:
    g = torch.Generator().manual_seed(seed)
    return StackedWeights(
        torch.randn(modes, 2 * inter, hidden, generator=g) * 0.2,
        torch.randn(modes, hidden, inter, generator=g) * 0.2,
    )


def distinct_modes(model: Qwen3ForCausalLM) -> Routing:
    with torch.no_grad():
        model.lm_head.weight[2].mul_(3.0)
        model.lm_head.weight[3].mul_(3.0)
    routing = apply_surgery(model, SurgeryConfig(variant="full"))
    with torch.no_grad():
        for m, scale in ((1, 0.4), (2, -0.4)):
            for layer in model.model.layers:
                layer.mlp.modes[m].gate_proj.weight.add_(scale)
    return routing


@pytest.mark.parametrize("impl", sorted(IMPLS))
@pytest.mark.parametrize("pattern", ["mixed", "uniform", "one_empty"])
def test_impls_match_loop(impl: str, pattern: str) -> None:
    w = random_weights(0, 3, 32, 48)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(37, 32, generator=g)
    ids = {
        "mixed": torch.randint(0, 3, (37,), generator=g),
        "uniform": torch.full((37,), 2),
        "one_empty": torch.randint(0, 2, (37,), generator=g) * 2,
    }[pattern]
    want = mode_mlp_loop(x, ids, w)
    torch.testing.assert_close(IMPLS[impl](x, ids, w), want, rtol=1e-5, atol=1e-5)


def test_stacked_full_modes_match_module(tiny_model: Qwen3ForCausalLM) -> None:
    routing = distinct_modes(tiny_model)
    mlp = tiny_model.model.layers[0].mlp
    assert isinstance(mlp, ModeMLP)
    x = torch.randn(1, 20, 32)
    ids = torch.randint(0, 3, (1, 20))
    routing.set(ids)
    with torch.no_grad():
        want = mlp(x)[0]
        got = mode_mlp_loop(x[0], ids[0], stack_mode_mlp(mlp))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_merged_lora_modes_match_module(tiny_model: Qwen3ForCausalLM) -> None:
    routing = apply_surgery(tiny_model, SurgeryConfig(variant="lora", lora_rank=4))
    mlp = tiny_model.model.layers[0].mlp
    assert isinstance(mlp, LoRAModeMLP)
    with torch.no_grad():
        for p in mlp.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    x = torch.randn(1, 20, 32)
    ids = torch.randint(0, 3, (1, 20))
    routing.set(ids)
    with torch.no_grad():
        want = mlp(x)[0]
        got = mode_mlp_loop(x[0], ids[0], stack_lora_mode_mlp(mlp))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block_m", [1, 4, 16])
def test_alignment_covers_every_token_once(block_m: int) -> None:
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 3, (29,), generator=g)
    ids[ids == 1] = 2
    al = align_by_mode(ids, 3, block_m)
    assert al.sorted_ids.numel() == padded_length(29, 3, block_m)
    real = al.sorted_ids[al.sorted_ids < 29]
    assert sorted(real.tolist()) == list(range(29))
    blocks = al.sorted_ids.view(-1, block_m)
    for b, mode in enumerate(al.block_modes.tolist()):
        tokens = blocks[b][blocks[b] < 29]
        if mode < 0:
            assert tokens.numel() == 0
        else:
            assert tokens.numel() > 0
            assert set(ids[tokens].tolist()) == {mode}
    assert 1 not in al.block_modes.tolist()


def per_sequence(model: Qwen3ForCausalLM, routing: Routing, n: int) -> tuple[list, list]:
    cfg = GenerationConfig(
        max_new_tokens=n, do_sample=False, eos_token_id=None, pad_token_id=TINY_PAD
    )
    tokens, traces = [], []
    for prompt in TINY_PROMPTS:
        with torch.no_grad(), mode_tracking(model, routing, tiny_texts()) as hook:
            out = model.generate(torch.tensor([prompt]), generation_config=cfg)
        tokens.append(out[0, len(prompt) :].tolist())
        traces.append(hook.trace[0])
    return tokens, traces


@pytest.mark.parametrize("impl", ["grouped", "aligned", "loop"])
def test_batched_mixed_mode_decode_matches_per_sequence(impl: str) -> None:
    model, routing = tiny_chained_model()
    want_tokens, want_modes = per_sequence(model, routing, 24)
    grouped = copy.deepcopy(model)
    grouped_routing = Routing()
    use_grouped_mlps(grouped, grouped_routing, impl)
    got = batched_greedy_decode(grouped, grouped_routing, tiny_texts(), TINY_PROMPTS, 24, TINY_PAD)
    assert got.token_ids == want_tokens
    assert got.modes == want_modes
    assert got.mixed_steps > 0
    assert {Mode.THINK, Mode.ANSWER, Mode.CODE} <= {m for row in got.modes for m in row}
    switched = [len(set(m[len(p) - 1 :])) > 1 for m, p in zip(got.modes, TINY_PROMPTS, strict=True)]
    assert sum(switched) >= 2


def test_tiny_model_outputs_depend_on_routing() -> None:
    model, routing = tiny_chained_model()
    distinct, _ = per_sequence(model, routing, 24)
    collapse_modes(model)
    same, _ = per_sequence(model, routing, 24)
    assert sum(a != b for a, b in zip(distinct, same, strict=True)) >= 3
