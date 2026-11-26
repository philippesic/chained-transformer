from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

BLOCK = 256
MIN_QUANT_NUMEL = 4096
DEFAULT_CHUNK = 1 << 22
_BF16_LOW_BITS = 0xFFFF
_BF16_HIGH_MASK = -65536


def dynamic_map(signed: bool, max_exponent_bits: int = 7, total_bits: int = 8) -> Tensor:
    data: list[float] = []
    non_sign_bits = total_bits - 1
    for i in range(max_exponent_bits):
        exponent = 10.0 ** (-(max_exponent_bits - 1) + i)
        shift = i + non_sign_bits - max_exponent_bits + (0 if signed else 1)
        items = 2**shift + 1
        bounds = torch.linspace(0.1, 1.0, items, dtype=torch.float64)
        means = ((bounds[:-1] + bounds[1:]) / 2.0).tolist()
        data += [exponent * m for m in means]
        if signed:
            data += [-exponent * m for m in means]
    data += [0.0, 1.0]
    if len(data) != 2**total_bits:
        raise ValueError(f"dynamic map has {len(data)} entries, expected {2**total_bits}")
    return torch.tensor(sorted(data), dtype=torch.float32)


class BlockQuantizer:
    def __init__(self, signed: bool, device: torch.device) -> None:
        self.signed = signed
        self.qmap = dynamic_map(signed).to(device)
        self.midpoints = (self.qmap[1:] + self.qmap[:-1]) / 2

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        blocks = x.view(-1, BLOCK)
        scale = blocks.abs().amax(dim=1).clamp_(min=1e-12)
        normed = blocks / scale[:, None]
        codes = torch.bucketize(normed, self.midpoints).to(torch.uint8)
        return codes.view(-1), scale

    def dequantize(self, codes: Tensor, scale: Tensor) -> Tensor:
        values = self.qmap[codes.long()].view(-1, BLOCK)
        return (values * scale[:, None]).view(-1)


def fp32_to_bf16_stochastic(x: Tensor, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.randint(
        0, 1 << 16, x.shape, device=x.device, dtype=torch.int32, generator=generator
    )
    bits = x.view(torch.int32)
    toward_zero = bits & _BF16_HIGH_MASK
    up = noise < (bits & _BF16_LOW_BITS)
    rounded = torch.where(up, toward_zero + (1 << 16), toward_zero)
    return rounded.view(torch.float32).to(torch.bfloat16)


def _quantizable(p: Tensor) -> bool:
    return p.numel() >= MIN_QUANT_NUMEL and p.numel() % BLOCK == 0


class ChunkedAdamW8bit(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bf16_stochastic_round: bool = True,
        chunk_numel: int = DEFAULT_CHUNK,
        seed: int = 0,
    ) -> None:
        if chunk_numel % BLOCK != 0:
            raise ValueError(f"chunk_numel must be a multiple of {BLOCK}, got {chunk_numel}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "bf16_stochastic_round": bf16_stochastic_round,
        }
        super().__init__(params, defaults)
        self.chunk_numel = chunk_numel
        self.seed = seed
        self._quantizers: dict[tuple[bool, torch.device], BlockQuantizer] = {}
        self._generators: dict[torch.device, torch.Generator] = {}

    def _quantizer(self, signed: bool, device: torch.device) -> BlockQuantizer:
        key = (signed, device)
        if key not in self._quantizers:
            self._quantizers[key] = BlockQuantizer(signed, device)
        return self._quantizers[key]

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[device]

    def _init_state(self, p: Tensor) -> dict[str, Any]:
        state: dict[str, Any] = {"step": 0}
        if _quantizable(p):
            blocks = p.numel() // BLOCK
            state["m_codes"] = torch.full(
                (p.numel(),), self._zero_code(True, p.device), dtype=torch.uint8, device=p.device
            )
            state["v_codes"] = torch.full(
                (p.numel(),), self._zero_code(False, p.device), dtype=torch.uint8, device=p.device
            )
            state["m_scale"] = torch.zeros(blocks, dtype=torch.float32, device=p.device)
            state["v_scale"] = torch.zeros(blocks, dtype=torch.float32, device=p.device)
        else:
            state["m"] = torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
            state["v"] = torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
        return state

    def _zero_code(self, signed: bool, device: torch.device) -> int:
        qmap = self._quantizer(signed, device).qmap
        return int((qmap == 0).nonzero()[0])

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            sr = group["bf16_stochastic_round"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ChunkedAdamW8bit does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state.update(self._init_state(p))
                state["step"] += 1
                step = state["step"]
                hyper = (lr, beta1, beta2, eps, wd, 1 - beta1**step, 1 - beta2**step, sr)
                self._update(p, state, hyper)
        return loss

    def _update(self, p: Tensor, state: dict[str, Any], hyper: tuple[Any, ...]) -> None:
        lr, beta1, beta2, eps, wd, bc1, bc2, sr = hyper
        flat_p = p.view(-1)
        flat_g = p.grad.view(-1) if p.grad is not None else None
        if flat_g is None:
            return
        quantized = "m_codes" in state
        step_size = lr / bc1
        bc2_sqrt = math.sqrt(bc2)
        for start in range(0, flat_p.numel(), self.chunk_numel):
            end = min(start + self.chunk_numel, flat_p.numel())
            g = flat_g[start:end].float()
            if quantized:
                mq = self._quantizer(True, p.device)
                vq = self._quantizer(False, p.device)
                bs, be = start // BLOCK, end // BLOCK
                m = mq.dequantize(state["m_codes"][start:end], state["m_scale"][bs:be])
                v = vq.dequantize(state["v_codes"][start:end], state["v_scale"][bs:be])
            else:
                m = state["m"][start:end]
                v = state["v"][start:end]
            m.lerp_(g, 1 - beta1)
            v.lerp_(g.square_(), 1 - beta2)
            denom = v.sqrt().div_(bc2_sqrt).add_(eps)
            pf = flat_p[start:end].float()
            if wd:
                pf.mul_(1 - lr * wd)
            pf.addcdiv_(m, denom, value=-step_size)
            if quantized:
                codes, scale = mq.quantize(m)
                state["m_codes"][start:end] = codes
                state["m_scale"][bs:be] = scale
                codes, scale = vq.quantize(v)
                state["v_codes"][start:end] = codes
                state["v_scale"][bs:be] = scale
            if p.dtype == torch.bfloat16 and sr:
                flat_p[start:end] = fp32_to_bf16_stochastic(pf, self._generator(p.device))
            else:
                flat_p[start:end] = pf.to(p.dtype)
