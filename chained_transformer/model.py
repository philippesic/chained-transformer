from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.surgery import Routing, SurgeryConfig, Variant, apply_surgery

BASE_MODEL = "Qwen/Qwen3-0.6B"

Arm = Literal["sft", "full", "lora", "moe", "separate"]
ARMS: tuple[Arm, ...] = ("sft", "full", "lora", "moe", "separate")


@dataclass
class Built:
    model: Qwen3ForCausalLM
    routing: Routing | None


def load_tokenizer(name: str = BASE_MODEL) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name)


def load_base(
    name: str = BASE_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    attn_implementation: str = "sdpa",
) -> Qwen3ForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation=attn_implementation
    )
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError(f"{name} loaded as {type(model).__name__}, expected Qwen3ForCausalLM")
    nn.Module.to(model, torch.device(device))
    return model


VARIANT_BY_ARM: dict[Arm, Variant] = {"full": "full", "lora": "lora", "moe": "moe"}


def surgery_for_arm(arm: Arm, base: SurgeryConfig | None = None) -> SurgeryConfig | None:
    variant = VARIANT_BY_ARM.get(arm)
    if variant is None:
        return None
    return replace(base or SurgeryConfig(), variant=variant)


def build(model: Qwen3ForCausalLM, arm: Arm, cfg: SurgeryConfig | None = None) -> Built:
    surgery = surgery_for_arm(arm, cfg)
    if surgery is None:
        return Built(model, None)
    return Built(model, apply_surgery(model, surgery))


def param_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
