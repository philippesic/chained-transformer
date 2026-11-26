from __future__ import annotations

import torch
import triton
import triton.language as tl

from serving.mixed import ModeAlignment, StackedWeights, align_by_mode


@triton.jit
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
        acc_g += tl.dot(a, wg)
        acc_u += tl.dot(a, wu)
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(
        h_ptr + offs_m[:, None] * stride_ht + offs_n[None, :] * stride_hi,
        h.to(h_ptr.dtype.element_ty),
        mask=n_mask[None, :],
    )


@triton.jit
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
        acc += tl.dot(a, w)
    tl.store(
        y_ptr + tok[:, None] * stride_yt + offs_n[None, :] * stride_yh,
        acc.to(y_ptr.dtype.element_ty),
        mask=tok_mask[:, None] & n_mask[None, :],
    )


@triton.jit
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
        acc += tl.dot(a, w)
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
    al = align_by_mode(mode_ids, w.num_modes, block_m)
    num_blocks = al.sorted_ids.numel() // block_m
    hidden, inter = int(x.shape[1]), w.intermediate
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
    )
    return y
