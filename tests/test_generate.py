from __future__ import annotations

import copy

import torch
from transformers import GenerationConfig, Qwen3ForCausalLM

from chained_transformer.generate import TokenTexts, generate_separate_models, mode_tracking
from chained_transformer.modes import Mode
from chained_transformer.surgery import SurgeryConfig, apply_surgery

SPECIAL = {0: "<|im_start|>", 1: "assistant\n", 2: "</think>", 3: "```\n", 4: "\n", 5: "<|im_end|>"}


def tiny_texts() -> TokenTexts:
    return TokenTexts([SPECIAL.get(i, f"w{i} ") for i in range(64)])


def greedy(model: Qwen3ForCausalLM, prompt: torch.Tensor, n: int) -> torch.Tensor:
    cfg = GenerationConfig(max_new_tokens=n, do_sample=False, eos_token_id=None, pad_token_id=63)
    return model.generate(prompt, generation_config=cfg)


def bias_toward_switch_tokens(model: Qwen3ForCausalLM) -> None:
    with torch.no_grad():
        model.lm_head.weight[2].mul_(3.0)
        model.lm_head.weight[3].mul_(3.0)


PROMPT = torch.tensor([[0, 1, 7, 8, 9]])


def test_hook_generation_with_identical_modes_matches_base(tiny_model: Qwen3ForCausalLM) -> None:
    bias_toward_switch_tokens(tiny_model)
    routed = copy.deepcopy(tiny_model)
    routing = apply_surgery(routed, SurgeryConfig(variant="full"))
    with torch.no_grad(), mode_tracking(routed, routing, tiny_texts()) as hook:
        got = greedy(routed, PROMPT, 30)
    want = greedy(tiny_model, PROMPT, 30)
    assert torch.equal(got, want)
    assert len(set(hook.trace[0])) > 1


def test_hook_modes_match_tracker_and_full_recompute(tiny_model: Qwen3ForCausalLM) -> None:
    bias_toward_switch_tokens(tiny_model)
    routing = apply_surgery(tiny_model, SurgeryConfig(variant="full"))
    with torch.no_grad():
        for m, scale in ((1, 0.4), (2, -0.4)):
            for layer in tiny_model.model.layers:
                layer.mlp.modes[m].gate_proj.weight.add_(scale)
    texts = tiny_texts()
    with torch.no_grad(), mode_tracking(tiny_model, routing, texts) as hook:
        out = greedy(tiny_model, PROMPT, 30)
    seq = out[0].tolist()
    modes = texts.modes(seq)
    assert hook.trace[0] == modes[:-1]
    assert {Mode.THINK, Mode.ANSWER} <= set(modes)
    routing.set(torch.tensor([modes[:-1]]))
    with torch.no_grad():
        logits = tiny_model(input_ids=out[:, :-1]).logits
    assert logits[0, PROMPT.shape[1] - 1 :].argmax(-1).tolist() == seq[PROMPT.shape[1] :]


def test_batched_rows_track_modes_independently(tiny_model: Qwen3ForCausalLM) -> None:
    bias_toward_switch_tokens(tiny_model)
    routing = apply_surgery(tiny_model, SurgeryConfig(variant="full"))
    prompts = torch.tensor([[0, 1, 7, 8, 9], [0, 1, 2, 4, 3]])
    texts = tiny_texts()
    with torch.no_grad(), mode_tracking(tiny_model, routing, texts) as hook:
        out = greedy(tiny_model, prompts, 12)
    for row in range(2):
        assert hook.trace[row] == texts.modes(out[row].tolist())[:-1]
    assert routing.mode_ids is None


def test_separate_models_with_identical_weights_match_one_model(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    bias_toward_switch_tokens(tiny_model)
    models = [copy.deepcopy(tiny_model) for _ in Mode]
    res = generate_separate_models(models, tiny_texts(), PROMPT[0].tolist(), 30, eos_ids=set())
    with torch.no_grad():
        want = greedy(tiny_model, PROMPT, 30)[0, PROMPT.shape[1] :].tolist()
    assert res.token_ids == want
    assert res.switches > 0 and res.reprefilled_tokens > 0
