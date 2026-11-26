from __future__ import annotations

import torch
import triton
import triton.language as tl

from serving.mixed import ModeAlignment, StackedWeights, align_by_mode

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def dot_precision(x: torch.Tensor) -> str:
    return "ieee" if x.dtype == torch.float32 else "tf32"


def _check_block(name: str, value: int) -> None:
    if value < 16 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two of at least 16 for tl.dot, got {value}")


def check_grouped_inputs(
    x: torch.Tensor,
    mode_ids: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    k_dim: int,
    blocks: tuple[int, int, int],
) -> None:
    if x.dim() != 2:
        raise ValueError(f"x must be [tokens, features], got shape {tuple(x.shape)}")
    if mode_ids.dim() != 1 or mode_ids.shape[0] != x.shape[0]:
        raise ValueError(
            f"mode_ids must be [tokens] with tokens={x.shape[0]}, got {tuple(mode_ids.shape)}"
        )
    if mode_ids.dtype.is_floating_point or mode_ids.dtype == torch.bool:
        raise ValueError(f"mode_ids must be an integer tensor, got {mode_ids.dtype}")
    if x.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"x must be one of {SUPPORTED_DTYPES}, got {x.dtype}")
    if x.shape[1] != k_dim:
        raise ValueError(f"x has {x.shape[1]} features but the weights expect {k_dim}")
    for t in (mode_ids, *weights):
        if t.device != x.device:
            raise ValueError(f"all inputs must be on {x.device}, found a tensor on {t.device}")
    for w in weights:
        if w.dtype != x.dtype:
            raise ValueError(
                f"weights are {w.dtype} but x is {x.dtype}; tl.dot needs both the same dtype"
            )
    if x.device.type == "cuda" and x.dtype == torch.bfloat16:
        major, minor = torch.cuda.get_device_capability(x.device)
        if major < 8:
            raise ValueError(
                f"bfloat16 tl.dot needs compute capability 8.0 or newer, this GPU is "
                f"{major}.{minor}; serve in float16"
            )
    for name, value in zip(("block_m", "block_n", "block_k"), blocks, strict=True):
        _check_block(name, value)


@triton.jit(do_not_specialize=["num_tokens"])
def _gate_up_silu_kernel(
    x_ptr,
    w_ptr,
    h_ptr,
    sorted_ids_ptr,
    block_modes_ptr,
    num_tokens,
    hidden,
    inter,
    stride_xt,
    stride_xh,
    stride_wm,
    stride_wn,
    stride_wk,
    stride_ht,
    stride_hi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    mode = tl.load(block_modes_ptr + pid_m).to(tl.int64)
    if mode < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    tok_mask = tok < num_tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < inter
    w_base = w_ptr + mode * stride_wm
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, hidden, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden
        a = tl.load(
            x_ptr + tok[:, None] * stride_xt + offs_k[None, :] * stride_xh,
            mask=tok_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        wk_mask = k_mask[:, None] & n_mask[None, :]
        wg = tl.load(
            w_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=wk_mask,
            other=0.0,
        )
        wu = tl.load(
            w_base + (offs_n[None, :] + inter) * stride_wn + offs_k[:, None] * stride_wk,
            mask=wk_mask,
            other=0.0,
        )
        acc_g += tl.dot(a, wg, input_precision=PRECISION)
        acc_u += tl.dot(a, wu, input_precision=PRECISION)
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(
        h_ptr + offs_m[:, None] * stride_ht + offs_n[None, :] * stride_hi,
        h.to(h_ptr.dtype.element_ty),
        mask=n_mask[None, :],
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _down_kernel(
    h_ptr,
    w_ptr,
    y_ptr,
    sorted_ids_ptr,
    block_modes_ptr,
    num_tokens,
    inter,
    hidden,
    stride_ht,
    stride_hi,
    stride_wm,
    stride_wn,
    stride_wk,
    stride_yt,
    stride_yh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    mode = tl.load(block_modes_ptr + pid_m).to(tl.int64)
    if mode < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    tok_mask = tok < num_tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < hidden
    w_base = w_ptr + mode * stride_wm
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, inter, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < inter
        a = tl.load(
            h_ptr + offs_m[:, None] * stride_ht + offs_k[None, :] * stride_hi,
            mask=k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, w, input_precision=PRECISION)
    tl.store(
        y_ptr + tok[:, None] * stride_yt + offs_n[None, :] * stride_yh,
        acc.to(y_ptr.dtype.element_ty),
        mask=tok_mask[:, None] & n_mask[None, :],
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _grouped_mm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    sorted_ids_ptr,
    block_modes_ptr,
    num_tokens,
    k_dim,
    n_dim,
    stride_xt,
    stride_xk,
    stride_wm,
    stride_wn,
    stride_wk,
    stride_yt,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    mode = tl.load(block_modes_ptr + pid_m).to(tl.int64)
    if mode < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    tok_mask = tok < num_tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < n_dim
    w_base = w_ptr + mode * stride_wm
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_dim
        a = tl.load(
            x_ptr + tok[:, None] * stride_xt + offs_k[None, :] * stride_xk,
            mask=tok_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, w, input_precision=PRECISION)
    tl.store(
        y_ptr + tok[:, None] * stride_yt + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=tok_mask[:, None] & n_mask[None, :],
    )


def grouped_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    mode_ids: torch.Tensor,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    alignment: ModeAlignment | None = None,
) -> torch.Tensor:
    if w.dim() != 3:
        raise ValueError(f"w must be [modes, out, in], got shape {tuple(w.shape)}")
    check_grouped_inputs(x, mode_ids, (w,), int(w.shape[2]), (block_m, block_n, block_k))
    al = alignment or align_by_mode(mode_ids, int(w.shape[0]), block_m)
    n_dim, k_dim = int(w.shape[1]), int(w.shape[2])
    y = torch.empty(x.shape[0], n_dim, dtype=x.dtype, device=x.device)
    grid = (al.sorted_ids.numel() // al.block_m, triton.cdiv(n_dim, block_n))
    _grouped_mm_kernel[grid](
        x,
        w,
        y,
        al.sorted_ids,
        al.block_modes,
        x.shape[0],
        k_dim,
        n_dim,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        y.stride(0),
        y.stride(1),
        BLOCK_M=al.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        PRECISION=dot_precision(x),
    )
    return y


def grouped_mode_mlp(
    x: torch.Tensor,
    mode_ids: torch.Tensor,
    w: StackedWeights,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    if w.gate_up.dim() != 3 or w.down.dim() != 3:
        raise ValueError(
            f"stacked weights must be 3-D, got gate_up {tuple(w.gate_up.shape)} "
            f"and down {tuple(w.down.shape)}"
        )
    modes, inter, hidden = w.num_modes, w.intermediate, int(w.down.shape[1])
    if tuple(w.gate_up.shape) != (modes, 2 * inter, hidden):
        raise ValueError(
            f"gate_up must be {(modes, 2 * inter, hidden)} to match down "
            f"{tuple(w.down.shape)}, got {tuple(w.gate_up.shape)}"
        )
    check_grouped_inputs(x, mode_ids, (w.gate_up, w.down), hidden, (block_m, block_n, block_k))
    al = align_by_mode(mode_ids, w.num_modes, block_m)
    num_blocks = al.sorted_ids.numel() // block_m
    h = torch.empty(al.sorted_ids.numel(), inter, dtype=x.dtype, device=x.device)
    _gate_up_silu_kernel[(num_blocks, triton.cdiv(inter, block_n))](
        x,
        w.gate_up,
        h,
        al.sorted_ids,
        al.block_modes,
        x.shape[0],
        hidden,
        inter,
        x.stride(0),
        x.stride(1),
        w.gate_up.stride(0),
        w.gate_up.stride(1),
        w.gate_up.stride(2),
        h.stride(0),
        h.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        PRECISION=dot_precision(x),
    )
    y = torch.empty_like(x)
    _down_kernel[(num_blocks, triton.cdiv(hidden, block_n))](
        h,
        w.down,
        y,
        al.sorted_ids,
        al.block_modes,
        x.shape[0],
        inter,
        hidden,
        h.stride(0),
        h.stride(1),
        w.down.stride(0),
        w.down.stride(1),
        w.down.stride(2),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        PRECISION=dot_precision(x),
    )
    return y
