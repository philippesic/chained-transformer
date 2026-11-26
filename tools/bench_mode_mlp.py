from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from serving.mixed import StackedWeights, mode_mlp_grouped, mode_mlp_loop

SHAPES = {"qwen3-0.6b": (1024, 3072), "qwen3-1.7b": (2048, 6144), "tiny": (128, 256)}
CONFIGS = [(16, 64, 64), (16, 128, 64), (32, 64, 64), (64, 64, 64), (64, 128, 32)]


def make_ids(mix: str, n: int, modes: int, device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(n)
    if mix == "uniform":
        ids = torch.zeros(n, dtype=torch.long)
    elif mix == "two":
        ids = torch.randint(0, 2, (n,), generator=g)
    else:
        ids = torch.randint(0, modes, (n,), generator=g)
    return ids.to(device)


def timed(fn: Callable[[], torch.Tensor], device: str, iters: int) -> float:
    for _ in range(3):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) * 1000 / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters


def graph_check(w: StackedWeights, hidden: int, n: int, modes: int) -> dict[str, Any]:
    from serving.triton_grouped import grouped_mode_mlp

    x = torch.randn(n, hidden, device="cuda", dtype=w.down.dtype) * 0.5
    ids = make_ids("three", n, modes, "cuda")
    grouped_mode_mlp(x, ids, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = grouped_mode_mlp(x, ids, w)
    errors = []
    for seed in range(3):
        g = torch.Generator().manual_seed(100 + seed)
        ids.copy_(torch.randint(0, modes, (n,), generator=g).cuda())
        x.copy_(torch.randn(n, hidden, generator=g).cuda().to(x.dtype) * 0.5)
        graph.replay()
        torch.cuda.synchronize()
        want = mode_mlp_loop(x.float(), ids, StackedWeights(w.gate_up.float(), w.down.float()))
        errors.append(float((out.float() - want).abs().max()))
    return {"captured": True, "max_abs_err_after_replays": max(errors)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    dtype = getattr(torch, args.dtype)
    hidden, inter = SHAPES[args.shape]
    g = torch.Generator().manual_seed(0)
    w = StackedWeights(
        (torch.randn(args.modes, 2 * inter, hidden, generator=g) * 0.02).to(device, dtype),
        (torch.randn(args.modes, hidden, inter, generator=g) * 0.02).to(device, dtype),
    )
    impls: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "dense_one_mode": lambda x, ids: F.linear(
            F.silu(F.linear(x, w.gate_up[0][:inter])) * F.linear(x, w.gate_up[0][inter:]),
            w.down[0],
        ),
        "loop": lambda x, ids: mode_mlp_loop(x, ids, w),
        "grouped_torch": lambda x, ids: mode_mlp_grouped(x, ids, w),
    }
    if device == "cuda" and not args.skip_triton:
        from serving.triton_grouped import grouped_mode_mlp

        for bm, bn, bk in CONFIGS:
            impls[f"triton_{bm}x{bn}x{bk}"] = functools.partial(
                grouped_mode_mlp, w=w, block_m=bm, block_n=bn, block_k=bk
            )
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

            w13 = w.gate_up.contiguous()
            w2 = w.down.contiguous()

            def vllm_moe(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
                topk_ids = ids.to(torch.int32).unsqueeze(1)
                weights = torch.ones(ids.shape[0], 1, dtype=torch.float32, device=ids.device)
                return fused_experts(x, w13, w2, weights, topk_ids)

            impls["vllm_fused_experts_top1"] = vllm_moe
        except Exception as e:
            print(f"vLLM fused_experts unavailable: {e!r}")
    rows = []
    for n in args.tokens:
        for mix in args.mixes:
            x = torch.randn(n, hidden, device=device, dtype=dtype) * 0.5
            ids = make_ids(mix, n, args.modes, device)
            ref = mode_mlp_loop(x.float(), ids, StackedWeights(w.gate_up.float(), w.down.float()))
            for name, fn in impls.items():
                try:
                    us = timed(functools.partial(fn, x, ids), device, args.iters)
                    err = float((fn(x, ids).float() - ref).abs().max())
                except Exception as e:
                    rows.append({"tokens": n, "mix": mix, "impl": name, "error": repr(e)[:200]})
                    continue
                if name == "dense_one_mode":
                    err = float("nan")
                rows.append({"tokens": n, "mix": mix, "impl": name, "us": us, "max_abs_err": err})
                print(f"{n:5d} {mix:8s} {name:28s} {us:10.1f} us  err {err:.2e}", flush=True)
    report: dict[str, Any] = {
        "shape": args.shape,
        "hidden": hidden,
        "intermediate": inter,
        "modes": args.modes,
        "dtype": args.dtype,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "rows": rows,
    }
    if device == "cuda" and not args.skip_triton:
        try:
            report["cuda_graph"] = graph_check(w, hidden, 64, args.modes)
        except Exception as e:
            report["cuda_graph"] = {"captured": False, "error": repr(e)[:300]}
        print("cuda graph:", report["cuda_graph"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Time one layer's mode MLP on mixed batches")
    parser.add_argument("--shape", choices=sorted(SHAPES), default="qwen3-0.6b")
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256, 512])
    parser.add_argument("--mixes", nargs="+", default=["uniform", "two", "three"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
