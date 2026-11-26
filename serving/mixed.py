from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache, PreTrainedModel

from chained_transformer.generate import TokenTexts
from chained_transformer.modes import Mode, ModeTracker
from chained_transformer.surgery import (
    LoRALinear,
    LoRAModeMLP,
    ModeMLP,
    Routing,
    decoder_layers,
)

Impl = Literal["loop", "gather", "grouped"]


@dataclass(frozen=True)
class StackedWeights:
    gate_up: torch.Tensor
    down: torch.Tensor

    @property
    def num_modes(self) -> int:
        return int(self.down.shape[0])

    @property
    def intermediate(self) -> int:
        return int(self.down.shape[2])

    def to(self, dtype: torch.dtype) -> StackedWeights:
        return StackedWeights(self.gate_up.to(dtype), self.down.to(dtype))


def stack_mode_mlp(mlp: ModeMLP) -> StackedWeights:
    gate_up = torch.stack(
        [
            torch.cat([m.get_parameter("gate_proj.weight"), m.get_parameter("up_proj.weight")])
            for m in mlp.modes
        ]
    )
    down = torch.stack([m.get_parameter("down_proj.weight") for m in mlp.modes])
    return StackedWeights(gate_up.detach().clone(), down.detach().clone())


def merged_lora_weight(linear: LoRALinear, mode: int) -> torch.Tensor:
    base = linear.base.weight
    delta = linear.lora_b[mode].float() @ linear.lora_a[mode].float()
    return (base.float() + linear.scale * delta).to(base.dtype)


def stack_lora_mode_mlp(mlp: LoRAModeMLP) -> StackedWeights:
    gate_up = torch.stack(
        [
            torch.cat([merged_lora_weight(mlp.gate_proj, m), merged_lora_weight(mlp.up_proj, m)], 0)
            for m in range(mlp.num_modes)
        ]
    )
    down = torch.stack([merged_lora_weight(mlp.down_proj, m) for m in range(mlp.num_modes)])
    return StackedWeights(gate_up.detach().clone(), down.detach().clone())


def stack_any(mlp: nn.Module) -> StackedWeights:
    if isinstance(mlp, ModeMLP):
        return stack_mode_mlp(mlp)
    if isinstance(mlp, LoRAModeMLP):
        return stack_lora_mode_mlp(mlp)
    raise TypeError(f"expected ModeMLP or LoRAModeMLP, found {type(mlp).__name__}")


def _mlp(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


def mode_mlp_loop(x: torch.Tensor, mode_ids: torch.Tensor, w: StackedWeights) -> torch.Tensor:
    out = torch.zeros_like(x)
    for m in range(w.num_modes):
        idx = (mode_ids == m).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        out.index_copy_(0, idx, _mlp(x.index_select(0, idx), w.gate_up[m], w.down[m]))
    return out


def mode_mlp_gather(x: torch.Tensor, mode_ids: torch.Tensor, w: StackedWeights) -> torch.Tensor:
    gate_up = w.gate_up[mode_ids]
    down = w.down[mode_ids]
    gate, up = torch.bmm(gate_up, x.unsqueeze(-1)).squeeze(-1).chunk(2, dim=-1)
    return torch.bmm(down, (F.silu(gate) * up).unsqueeze(-1)).squeeze(-1)


def mode_counts(mode_ids: torch.Tensor, num_modes: int) -> torch.Tensor:
    counts = torch.zeros(num_modes, dtype=torch.long, device=mode_ids.device)
    return counts.scatter_add_(0, mode_ids.long(), torch.ones_like(mode_ids, dtype=torch.long))


def mode_mlp_grouped(x: torch.Tensor, mode_ids: torch.Tensor, w: StackedWeights) -> torch.Tensor:
    order = torch.argsort(mode_ids, stable=True)
    xs = x.index_select(0, order)
    counts = mode_counts(mode_ids, w.num_modes).tolist()
    parts = []
    start = 0
    for m, count in enumerate(counts):
        if count:
            parts.append(_mlp(xs[start : start + count], w.gate_up[m], w.down[m]))
        start += count
    out = torch.empty_like(x)
    out.index_copy_(0, order, torch.cat(parts) if parts else xs)
    return out


IMPLS: dict[str, Callable[[torch.Tensor, torch.Tensor, StackedWeights], torch.Tensor]] = {
    "loop": mode_mlp_loop,
    "gather": mode_mlp_gather,
    "grouped": mode_mlp_grouped,
}


@dataclass(frozen=True)
class ModeAlignment:
    sorted_ids: torch.Tensor
    block_modes: torch.Tensor
    block_m: int
    num_tokens: int


def padded_length(num_tokens: int, num_modes: int, block_m: int) -> int:
    worst = num_tokens + num_modes * (block_m - 1)
    return -(-worst // block_m) * block_m


def align_by_mode(mode_ids: torch.Tensor, num_modes: int, block_m: int) -> ModeAlignment:
    device = mode_ids.device
    ids = mode_ids.long()
    num_tokens = int(ids.shape[0])
    total = padded_length(num_tokens, num_modes, block_m)
    counts = mode_counts(ids, num_modes)
    padded = (counts + block_m - 1) // block_m * block_m
    ends = torch.cumsum(padded, 0)
    starts = ends - padded
    first = torch.cumsum(counts, 0) - counts
    order = torch.argsort(ids, stable=True)
    sorted_modes = ids.index_select(0, order)
    rank = torch.arange(num_tokens, device=device) - first.index_select(0, sorted_modes)
    dest = starts.index_select(0, sorted_modes) + rank
    sorted_ids = torch.full((total,), num_tokens, dtype=torch.long, device=device)
    sorted_ids.scatter_(0, dest, order)
    block_starts = torch.arange(0, total, block_m, device=device)
    block_modes = torch.bucketize(block_starts, ends, right=True)
    block_modes = torch.where(block_modes < num_modes, block_modes, -1)
    return ModeAlignment(sorted_ids, block_modes, block_m, num_tokens)


def mode_mlp_aligned(
    x: torch.Tensor, mode_ids: torch.Tensor, w: StackedWeights, block_m: int = 16
) -> torch.Tensor:
    al = align_by_mode(mode_ids, w.num_modes, block_m)
    padded_x = torch.cat([x, x.new_zeros(1, x.shape[1])])
    rows = padded_x.index_select(0, al.sorted_ids).view(-1, block_m, x.shape[1])
    modes = al.block_modes.clamp(min=0)
    gate, up = torch.bmm(rows, w.gate_up.index_select(0, modes).transpose(1, 2)).chunk(2, -1)
    h = torch.bmm(F.silu(gate) * up, w.down.index_select(0, modes).transpose(1, 2))
    out = x.new_zeros(x.shape[0] + 1, x.shape[1])
    out.index_copy_(0, al.sorted_ids, h.reshape(-1, x.shape[1]))
    return out[: x.shape[0]]


IMPLS["aligned"] = mode_mlp_aligned


class GroupedModeMLP(nn.Module):
    def __init__(self, weights: StackedWeights, routing: Routing, impl: str = "grouped") -> None:
        super().__init__()
        self.gate_up = nn.Parameter(weights.gate_up, requires_grad=False)
        self.down = nn.Parameter(weights.down, requires_grad=False)
        self.routing = routing
        self.impl = IMPLS[impl]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        ids = self.routing.flat_ids(flat.shape[0]).to(flat.device)
        out = self.impl(flat, ids, StackedWeights(self.gate_up, self.down))
        return out.reshape(shape)


def use_grouped_mlps(model: PreTrainedModel, routing: Routing, impl: str = "grouped") -> None:
    for layer in decoder_layers(model):
        layer.mlp = GroupedModeMLP(stack_any(layer.get_submodule("mlp")), routing, impl)


@dataclass
class BatchedDecode:
    token_ids: list[list[int]]
    modes: list[list[int]]
    step_modes: list[list[int]] = field(default_factory=list)

    @property
    def mixed_steps(self) -> int:
        return sum(len(set(step)) > 1 for step in self.step_modes)


@torch.no_grad()
def batched_greedy_decode(
    model: nn.Module,
    routing: Routing,
    texts: TokenTexts,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    pad_id: int,
    prompt_mode: Mode = Mode.THINK,
) -> BatchedDecode:
    device = next(model.parameters()).device
    width = max(len(p) for p in prompts)
    rows = [[pad_id] * (width - len(p)) + list(p) for p in prompts]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    mask = torch.tensor(
        [[0] * (width - len(p)) + [1] * len(p) for p in prompts], dtype=torch.long, device=device
    )
    trackers = [ModeTracker(prompt_mode=prompt_mode) for _ in prompts]
    modes = [[int(t.feed(texts[i])) for i in p] for t, p in zip(trackers, prompts, strict=True)]
    step = [[0] * (width - len(p)) + m for p, m in zip(prompts, modes, strict=True)]
    routing.set(torch.tensor(step, dtype=torch.long, device=device))
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    cache = DynamicCache()
    logits = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=positions,
        cache=cache,
        use_cache=True,
    ).logits
    out: list[list[int]] = [[] for _ in prompts]
    step_modes: list[list[int]] = []
    last_pos = positions[:, -1:]
    for i in range(max_new_tokens):
        next_ids = logits[:, -1].argmax(-1).tolist()
        for row, token in enumerate(next_ids):
            out[row].append(token)
        if i == max_new_tokens - 1:
            break
        new_modes = [int(t.feed(texts[tok])) for t, tok in zip(trackers, next_ids, strict=True)]
        for history, m in zip(modes, new_modes, strict=True):
            history.append(m)
        step_modes.append(new_modes)
        routing.set(torch.tensor([[m] for m in new_modes], dtype=torch.long, device=device))
        mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=1)
        last_pos = last_pos + 1
        logits = model(
            input_ids=torch.tensor([[t] for t in next_ids], dtype=torch.long, device=device),
            attention_mask=mask,
            position_ids=last_pos,
            cache=cache,
            use_cache=True,
        ).logits
    routing.clear()
    return BatchedDecode(out, modes, step_modes)
