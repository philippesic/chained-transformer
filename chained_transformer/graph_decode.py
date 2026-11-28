from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RMSNorm

from chained_transformer.cache import RoomyLayer, roomy_cache
from chained_transformer.generate import ModeHook
from chained_transformer.surgery import LoRAModeMLP, ModeMLP, MoEMLP, Routing

PREFILL_CHUNK = 512
BUCKET = 512
DECODE_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def rms(norm: Qwen3RMSNorm, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * h.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def dense_mlp(mlp: nn.Module, h: torch.Tensor, modes: torch.Tensor | None) -> torch.Tensor:
    if isinstance(mlp, ModeMLP):
        assert modes is not None
        outs = torch.stack([m(h) for m in mlp.modes])
        return outs[modes, torch.arange(h.shape[0], device=h.device)]
    if isinstance(mlp, LoRAModeMLP):
        assert modes is not None
        outs = torch.stack([mlp._mode_forward(m, h) for m in range(mlp.num_modes)])
        return outs[modes, torch.arange(h.shape[0], device=h.device)]
    if isinstance(mlp, MoEMLP):
        flat = h.reshape(-1, h.shape[-1])
        _, weights, chosen = mlp.route(flat)
        outs = torch.stack([e(h).reshape(-1, h.shape[-1]) for e in mlp.experts])
        rows = torch.arange(chosen.shape[0], device=h.device)
        first = outs[chosen[:, 0], rows].float()
        out = first
        for k in range(1, mlp.top_k):
            out = out + weights[:, k, None] * (outs[chosen[:, k], rows].float() - first)
        return out.to(h.dtype).reshape(h.shape)
    return mlp(h)


@dataclass
class DecodeState:
    token: torch.Tensor
    position: torch.Tensor
    column: torch.Tensor
    modes: torch.Tensor
    valid: torch.Tensor
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


def decode_step(model: Qwen3ForCausalLM, s: DecodeState, limit: int) -> torch.Tensor:
    core = model.model
    cfg = model.config
    x = core.embed_tokens(s.token)
    batch = x.shape[0]
    inv_freq = core.rotary_emb.inv_freq.to(torch.float32)
    freqs = s.position[:, None].to(torch.float32) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    scaling = core.rotary_emb.attention_scaling
    cos = (emb.cos() * scaling).to(x.dtype)[:, None, :]
    sin = (emb.sin() * scaling).to(x.dtype)[:, None, :]
    mask = s.valid[:, None, None, :limit]
    for i, layer in enumerate(core.layers[: cfg.num_hidden_layers]):
        attn = layer.self_attn
        dim = attn.head_dim
        h = rms(layer.input_layernorm, x)
        q = rms(attn.q_norm, attn.q_proj(h).view(batch, -1, dim))
        k = rms(attn.k_norm, attn.k_proj(h).view(batch, -1, dim))
        v = attn.v_proj(h).view(batch, -1, dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        keys, values = s.keys[i], s.values[i]
        keys.index_copy_(2, s.column, k[:, :, None, :])
        values.index_copy_(2, s.column, v[:, :, None, :])
        with sdpa_kernel(DECODE_BACKENDS):
            out = F.scaled_dot_product_attention(
                q[:, :, None, :],
                keys[:, :, :limit],
                values[:, :, :limit],
                attn_mask=mask,
                scale=attn.scaling,
                enable_gqa=True,
            )
        x = x + attn.o_proj(out.reshape(batch, -1))
        h = rms(layer.post_attention_layernorm, x)
        x = x + dense_mlp(layer.mlp, h, s.modes)
    x = rms(core.norm, x)
    return model.lm_head(x).argmax(dim=-1)


class GraphSteps:
    def __init__(self, model: Qwen3ForCausalLM, state: DecodeState, capture: bool) -> None:
        self.model = model
        self.state = state
        self.capture = capture
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.pool = torch.cuda.graph_pool_handle() if capture else None

    def _record(self, limit: int) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        state = self.state
        saved = [t.clone() for t in [state.token, state.valid]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            decode_step(self.model, state, limit)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = decode_step(self.model, state, limit)
        for dst, src in zip([state.token, state.valid], saved, strict=True):
            dst.copy_(src)
        return graph, out

    def __call__(self, column: int) -> torch.Tensor:
        limit = min(self.state.valid.shape[1], (column // BUCKET + 1) * BUCKET)
        if not self.capture:
            return decode_step(self.model, self.state, limit)
        if limit not in self.graphs:
            self.graphs[limit] = self._record(limit)
        graph, out = self.graphs[limit]
        graph.replay()
        return out


EMPTY = cast(torch.FloatTensor, torch.empty(0))


@torch.no_grad()
def greedy_generate(
    model: Qwen3ForCausalLM,
    ids: torch.Tensor,
    mask: torch.Tensor,
    room: int,
    pad: int,
    stop: Callable[[torch.LongTensor, torch.FloatTensor], torch.BoolTensor],
    routing: Routing | None = None,
    hook: ModeHook | None = None,
    capture: bool | None = None,
) -> torch.Tensor:
    device = ids.device
    batch, width = ids.shape
    total = width + room
    capture = device.type == "cuda" if capture is None else capture
    cache = roomy_cache(total)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    for start in range(0, width, PREFILL_CHUNK):
        end = min(width, start + PREFILL_CHUNK)
        logits = model(
            input_ids=ids[:, start:end],
            attention_mask=mask[:, :end],
            position_ids=positions[:, start:end],
            cache=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits
    layers = [layer for layer in cache.layers if isinstance(layer, RoomyLayer)]
    stores = [layer._store for layer in layers]
    if any(s is None or s[0].shape[-2] < total for s in stores):
        raise RuntimeError("prefill cache is smaller than the decode window")
    keys = [s[0] for s in stores if s is not None]
    values = [s[1] for s in stores if s is not None]
    for t in keys + values:
        t[..., width:, :].zero_()
    valid = torch.zeros(batch, total, dtype=torch.bool, device=device)
    valid[:, :width] = mask.bool()
    seq = torch.full((batch, total), pad, dtype=torch.long, device=device)
    seq[:, :width] = ids
    state = DecodeState(
        token=logits[:, -1].argmax(dim=-1),
        position=positions[:, -1] + 1,
        column=torch.tensor([width], dtype=torch.long, device=device),
        modes=torch.zeros(batch, dtype=torch.long, device=device),
        valid=valid,
        keys=keys,
        values=values,
    )
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    steps = GraphSteps(model, state, capture)
    for s in range(room):
        token = torch.where(finished, torch.full_like(state.token, pad), state.token)
        seq[:, width + s] = token
        done = stop(cast(torch.LongTensor, seq[:, : width + s + 1]), EMPTY)
        if routing is not None and hook is not None:
            modes = [
                int(t.feed(hook.texts[tok]))
                for t, tok in zip(hook.trackers, token.tolist(), strict=True)
            ]
            for history, m in zip(hook.trace, modes, strict=True):
                history.append(m)
            state.modes.copy_(torch.tensor(modes, dtype=torch.long))
        if bool(done.all()) or s == room - 1:
            return seq[:, : width + s + 1]
        finished |= done
        state.token.copy_(token)
        state.valid[:, width + s] = True
        state.token.copy_(steps(width + s))
        state.position += 1
        state.column += 1
    return seq
