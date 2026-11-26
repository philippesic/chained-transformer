from __future__ import annotations

from pathlib import Path

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from chained_transformer.checkpoint import save_trained
from chained_transformer.generate import TokenTexts, mode_tracking
from chained_transformer.surgery import ModeMLP, Routing, SurgeryConfig, apply_surgery

TINY_SPECIAL = {
    0: "<|im_start|>",
    1: "assistant\n",
    2: "</think>",
    3: "```\n",
    4: "\n",
    5: "<|im_end|>",
}
TINY_VOCAB = 64
TINY_PAD = 63
TINY_LORA_RANK = 8
TINY_PROMPTS: list[list[int]] = [
    [0, 1, 7, 8, 9],
    [0, 1, 7, 2, 8],
    [0, 1, 2, 4, 3, 10, 11],
    [0, 1, 12],
]


def tiny_token_texts() -> list[str]:
    return [TINY_SPECIAL.get(i, f"w{i} ") for i in range(TINY_VOCAB)]


def tiny_texts() -> TokenTexts:
    return TokenTexts(tiny_token_texts())


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=TINY_VOCAB,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )


def tiny_base(seed: int = 0, std: float = 0.2, switch_scale: float = 2.0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(tiny_config()).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            else:
                p.normal_(0, std)
        model.lm_head.weight[2].mul_(switch_scale)
        model.lm_head.weight[3].mul_(switch_scale)
    return model


def tiny_chained_model(
    seed: int = 0, std: float = 0.2, mlp_std: float = 0.3, switch_scale: float = 2.0
) -> tuple[Qwen3ForCausalLM, Routing]:
    model = tiny_base(seed, std, switch_scale)
    routing = apply_surgery(model, SurgeryConfig(variant="full"))
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            assert isinstance(mlp, ModeMLP)
            for mode in mlp.modes:
                for p in mode.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * mlp_std)
    return model, routing


def tiny_lora_model(
    seed: int = 0, std: float = 0.2, lora_std: float = 0.3, rank: int = TINY_LORA_RANK
) -> tuple[Qwen3ForCausalLM, Routing]:
    model = tiny_base(seed, std)
    routing = apply_surgery(model, SurgeryConfig(variant="lora", lora_rank=rank))
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_" in name:
                p.copy_(torch.randn(p.shape, generator=g) * lora_std)
    return model, routing


def tiny_pilot_checkpoint(variant: str, path: Path) -> Path:
    if variant == "full":
        model, _ = tiny_chained_model()
        surgery = SurgeryConfig(variant="full")
    elif variant == "lora":
        model, _ = tiny_lora_model()
        surgery = SurgeryConfig(variant="lora", lora_rank=TINY_LORA_RANK)
    else:
        raise ValueError(f"no tiny pilot checkpoint for variant {variant!r}")
    save_trained(model, surgery, path, base_model="tiny")
    return path


def collapse_modes(model: Qwen3ForCausalLM) -> None:
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            assert isinstance(mlp, ModeMLP)
            for mode in mlp.modes[1:]:
                mode.load_state_dict(mlp.modes[0].state_dict())


def tiny_greedy(
    model: Qwen3ForCausalLM, routing: Routing, prompts: list[list[int]], new_tokens: int
) -> tuple[list[list[int]], list[list[int]]]:
    cfg = GenerationConfig(
        max_new_tokens=new_tokens, do_sample=False, eos_token_id=None, pad_token_id=TINY_PAD
    )
    tokens, traces = [], []
    for prompt in prompts:
        with torch.no_grad(), mode_tracking(model, routing, tiny_texts()) as hook:
            out = model.generate(torch.tensor([prompt]), generation_config=cfg)
        tokens.append(out[0, len(prompt) :].tolist())
        traces.append(hook.trace[0])
    return tokens, traces
