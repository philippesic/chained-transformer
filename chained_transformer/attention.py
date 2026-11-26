from __future__ import annotations

import warnings
from typing import Any, cast

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


FALLBACKS: list[str] = []


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
    q_len = query.shape[2]
    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", True))
    is_causal = q_len > 1 and attention_mask is None and is_causal
    if is_causal and key.shape[2] > q_len:
        key, value = key[:, :, :q_len], value[:, :, :q_len]
    dropout_p = dropout if module.training else 0.0

    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scaling,
            is_causal=is_causal,
        )

    if not query.is_cuda:
        out = attend(query, key, value)
    else:
        try:
            with sdpa_kernel(FUSED_BACKENDS):
                out = attend(query, key, value)
        except RuntimeError as err:
            reason = f"{tuple(query.shape)} {query.dtype} mask={attention_mask is not None}: {err}"
            if not FALLBACKS:
                warnings.warn(f"fused attention fell back to math: {reason}", stacklevel=2)
            FALLBACKS.append(reason)
            out = attend(query, key, value)
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
        cast(Any, model).config._attn_implementation = name


def attention_backend(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "_attn_implementation", "unknown"))
