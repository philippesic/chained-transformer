from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

NAME = "ct_fused_sdpa"
ATTENTION_CHOICES = ("auto", "sdpa", "eager")
FUSED_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


def needs_fp32(t: torch.Tensor) -> bool:
    if not t.is_cuda or t.dtype != torch.bfloat16:
        return False
    return torch.cuda.get_device_capability(t.device) < (8, 0)


def fused_sdpa(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    dtype = query.dtype
    if needs_fp32(query):
        query, key, value = query.float(), key.float(), value.float()
        if attention_mask is not None and attention_mask.is_floating_point():
            attention_mask = attention_mask.float()
    if is_causal is None:
        is_causal = attention_mask is None and query.shape[2] > 1
        is_causal = is_causal and bool(getattr(module, "is_causal", True))
    with sdpa_kernel(FUSED_BACKENDS):
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
            is_causal=is_causal,
        )
    return out.to(dtype).transpose(1, 2).contiguous(), None


def register() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, fused_sdpa)
    try:
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    except ImportError:
        return
    AttentionMaskInterface.register(NAME, sdpa_mask)


def use_attention(model: nn.Module, choice: str) -> None:
    if choice not in ATTENTION_CHOICES:
        raise ValueError(f"attention must be one of {ATTENTION_CHOICES}, got {choice!r}")
    name = {"auto": NAME, "sdpa": "sdpa", "eager": "eager"}[choice]
    if name == NAME:
        register()
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter(name)
    else:
        model.config._attn_implementation = name


def attention_backend(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "_attn_implementation", "unknown"))
