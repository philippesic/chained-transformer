from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from chained_transformer.modes import MODE_NAMES

Variant = Literal["full", "lora", "moe"]


@dataclass(frozen=True)
class SurgeryConfig:
    variant: Variant = "full"
    num_modes: int = len(MODE_NAMES)
    lora_rank: int = 64
    lora_alpha: float = 128.0
    moe_experts: int = len(MODE_NAMES)
    moe_top_k: int = 2
    moe_router_std: float = 0.02

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Routing:
    def __init__(self) -> None:
        self.mode_ids: torch.Tensor | None = None
        self.uniform_mode: int | None = None

    def set(self, mode_ids: torch.Tensor) -> None:
        unique = torch.unique(mode_ids).tolist()
        self.mode_ids = mode_ids
        self.uniform_mode = int(unique[0]) if len(unique) == 1 else None

    def set_uniform(self, mode: int, shape: tuple[int, int], device: torch.device) -> None:
        self.mode_ids = torch.full(shape, mode, dtype=torch.long, device=device)
        self.uniform_mode = mode

    def clear(self) -> None:
        self.mode_ids = None
        self.uniform_mode = None

    def flat_ids(self, num_tokens: int) -> torch.Tensor:
        if self.mode_ids is None:
            raise RuntimeError("mode ids not set: call Routing.set before the forward pass")
        ids = self.mode_ids.reshape(-1)
        if ids.numel() != num_tokens:
            raise RuntimeError(
                f"mode ids cover {ids.numel()} tokens but the MLP received {num_tokens}; "
                "set mode ids with the same [batch, seq] shape as input_ids"
            )
        return ids


def _route(
    x: torch.Tensor,
    routing: Routing,
    num_modes: int,
    fn: Callable[[int, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if routing.uniform_mode is not None:
        routing.flat_ids(x.shape[0] * x.shape[1])
        return fn(routing.uniform_mode, x)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    ids = routing.flat_ids(flat.shape[0])
    out = flat.new_zeros(flat.shape[0], shape[-1])
    for m in range(num_modes):
        idx = (ids == m).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        out = out.index_add(0, idx, fn(m, flat.index_select(0, idx)))
    return out.reshape(shape)


class ModeMLP(nn.Module):
    def __init__(self, base: nn.Module, num_modes: int, routing: Routing) -> None:
        super().__init__()
        self.modes = nn.ModuleList(copy.deepcopy(base) for _ in range(num_modes))
        self.routing = routing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _route(x, self.routing, len(self.modes), lambda m, h: self.modes[m](h))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, num_modes: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        dtype, device = base.weight.dtype, base.weight.device
        self.lora_a = nn.ParameterList(
            nn.Parameter(torch.empty(rank, base.in_features, dtype=dtype, device=device))
            for _ in range(num_modes)
        )
        self.lora_b = nn.ParameterList(
            nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype, device=device))
            for _ in range(num_modes)
        )
        for a in self.lora_a:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, mode: int, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(x, self.lora_a[mode]), self.lora_b[mode])
        return self.base(x) + delta * self.scale


class LoRAModeMLP(nn.Module):
    def __init__(
        self, base: Qwen3MLP, num_modes: int, rank: int, alpha: float, routing: Routing
    ) -> None:
        super().__init__()
        self.gate_proj = LoRALinear(base.gate_proj, num_modes, rank, alpha)
        self.up_proj = LoRALinear(base.up_proj, num_modes, rank, alpha)
        self.down_proj = LoRALinear(base.down_proj, num_modes, rank, alpha)
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = base.act_fn
        self.num_modes = num_modes
        self.routing = routing

    def _mode_forward(self, m: int, h: torch.Tensor) -> torch.Tensor:
        inner = self.act_fn(self.gate_proj(m, h)) * self.up_proj(m, h)
        return self.down_proj(m, inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _route(x, self.routing, self.num_modes, self._mode_forward)


class MoEMLP(nn.Module):
    def __init__(self, base: Qwen3MLP, num_experts: int, top_k: int, router_std: float) -> None:
        super().__init__()
        weight = next(base.parameters())
        self.experts = nn.ModuleList(copy.deepcopy(base) for _ in range(num_experts))
        self.router = nn.Linear(
            base.hidden_size, num_experts, bias=False, dtype=weight.dtype, device=weight.device
        )
        nn.init.normal_(self.router.weight, std=router_std)
        self.top_k = top_k
        self.aux_loss: torch.Tensor | None = None

    def route(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(flat.float(), self.router.weight.float())
        probs = F.softmax(logits, dim=-1)
        weights, chosen = probs.topk(self.top_k, dim=-1)
        return probs, weights / weights.sum(dim=-1, keepdim=True), chosen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs, weights, chosen = self.route(flat)
        num_experts = len(self.experts)
        load = F.one_hot(chosen, num_experts).float().sum(dim=1).mean(dim=0) / self.top_k
        self.aux_loss = num_experts * (load * probs.mean(dim=0)).sum()
        out = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (chosen == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y = expert(flat.index_select(0, token_idx)).float() * weights[token_idx, slot, None]
            out = out.index_add(0, token_idx, y)
        return out.to(flat.dtype).reshape(shape)


def decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    layers = model.get_submodule("model.layers")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("expected model.model.layers to be a ModuleList (Qwen3-style decoder)")
    return layers


def apply_surgery(model: PreTrainedModel, cfg: SurgeryConfig) -> Routing:
    routing = Routing()
    for layer in decoder_layers(model):
        base = layer.get_submodule("mlp")
        if not isinstance(base, Qwen3MLP):
            raise TypeError(f"expected Qwen3MLP layers, found {type(base).__name__}")
        if cfg.variant == "full":
            layer.mlp = ModeMLP(base, cfg.num_modes, routing)
        elif cfg.variant == "lora":
            layer.mlp = LoRAModeMLP(base, cfg.num_modes, cfg.lora_rank, cfg.lora_alpha, routing)
        else:
            layer.mlp = MoEMLP(base, cfg.moe_experts, cfg.moe_top_k, cfg.moe_router_std)
    model.config.update({"mode_surgery": cfg.to_dict()})
    return routing


def moe_aux_loss(model: nn.Module) -> torch.Tensor | None:
    losses = [
        m.aux_loss for m in model.modules() if isinstance(m, MoEMLP) and m.aux_loss is not None
    ]
    if not losses:
        return None
    return torch.stack(losses).mean()


def mode_parameters(model: nn.Module, mode: int) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, ModeMLP):
            params.extend(module.modes[mode].parameters())
        elif isinstance(module, LoRALinear):
            params.extend([module.lora_a[mode], module.lora_b[mode]])
    return params


def freeze_all_but_mode(model: nn.Module, mode: int) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = mode_parameters(model, mode)
    for p in trainable:
        p.requires_grad_(True)
    return sum(p.numel() for p in trainable)
