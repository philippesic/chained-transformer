from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from serving.mixed import StackedWeights
from serving.paging import ModePager

SHAPES = {
    "qwen3-0.6b": (28, 1024, 3072),
    "qwen3-1.7b": (28, 2048, 6144),
    "tiny": (2, 128, 256),
}


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run(args: argparse.Namespace) -> dict[str, Any]:
    layers_n, hidden, inter = SHAPES[args.shape]
    dtype = getattr(torch, args.dtype)
    layers = [
        StackedWeights(
            torch.randn(args.modes, 2 * inter, hidden).to(dtype) * 0.02,
            torch.randn(args.modes, hidden, inter).to(dtype) * 0.02,
        )
        for _ in range(layers_n)
    ]
    pager = ModePager(layers, capacity=1, device=args.device)
    gib = pager.mode_bytes() / 2**30
    loads = []
    for i in range(args.iters):
        sync(args.device)
        t0 = time.perf_counter()
        pager.ensure([i % args.modes])
        sync(args.device)
        loads.append(time.perf_counter() - t0)
    loads = loads[1:] if len(loads) > 1 else loads
    x = torch.randn(args.tokens, hidden, device=args.device, dtype=dtype)
    w = pager.weights(0)
    sync(args.device)
    t0 = time.perf_counter()
    for _ in range(layers_n):
        torch.nn.functional.linear(x, w.gate_up[0])
    sync(args.device)
    step = time.perf_counter() - t0
    hidden_ms = []
    if args.device == "cuda":
        for i in range(args.iters):
            pager.ensure([i % args.modes])
            sync(args.device)
            t0 = time.perf_counter()
            pager.prefetch((i + 1) % args.modes)
            for _ in range(args.overlap_steps):
                for _ in range(layers_n):
                    torch.nn.functional.linear(x, w.gate_up[0])
            pager.ensure([(i + 1) % args.modes])
            sync(args.device)
            hidden_ms.append((time.perf_counter() - t0) * 1000)
    mean = sum(loads) / len(loads)
    report = {
        "shape": args.shape,
        "dtype": args.dtype,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "mode_gib": round(gib, 3),
        "load_ms_mean": round(mean * 1000, 2),
        "load_gib_per_s": round(gib / mean, 2),
        "gate_up_pass_ms": round(step * 1000, 3),
        "prefetch_then_ensure_ms": [round(v, 2) for v in hidden_ms],
        "overlap_steps": args.overlap_steps,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Time paging one mode's MLPs host to device")
    parser.add_argument("--shape", choices=sorted(SHAPES), default="qwen3-0.6b")
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--overlap-steps", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
