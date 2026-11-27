from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache

from chained_transformer.attention import use_attention
from chained_transformer.model import BASE_MODEL, load_base

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PREFILL_CHUNK = 512


def parse_shape(text: str) -> tuple[int, int]:
    rows, _, width = text.partition("x")
    return int(rows), int(width)


def padded_batch(
    rows: int, width: int, vocab: int, pad: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    margin = min(1000, vocab // 4)
    ids = torch.randint(margin, vocab - margin, (rows, width), generator=gen)
    mask = torch.ones_like(ids)
    for row in range(1, rows):
        blank = int(torch.randint(0, width // 2 + 1, (1,), generator=gen))
        ids[row, :blank] = pad
        mask[row, :blank] = 0
    return ids, mask


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_shape(
    model: Any, ids: torch.Tensor, mask: torch.Tensor, steps: int, forced: torch.Tensor | None
) -> dict[str, Any]:
    device = ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cache = DynamicCache()
    start = time.perf_counter()
    for begin in range(0, ids.shape[1], PREFILL_CHUNK):
        end = min(begin + PREFILL_CHUNK, ids.shape[1])
        logits = model(
            input_ids=ids[:, begin:end],
            attention_mask=mask[:, :end],
            cache=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits
    sync(device)
    prefill_s = time.perf_counter() - start
    step_s: list[float] = []
    tokens: list[torch.Tensor] = []
    kept: list[torch.Tensor] = []
    for step in range(steps):
        last = logits[:, -1].float()
        kept.append(last.cpu())
        nxt = last.argmax(-1, keepdim=True) if forced is None else forced[:, step : step + 1]
        tokens.append(nxt.cpu())
        mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
        sync(device)
        start = time.perf_counter()
        logits = model(
            input_ids=nxt, attention_mask=mask, cache=cache, use_cache=True, logits_to_keep=1
        ).logits
        sync(device)
        step_s.append(time.perf_counter() - start)
    warm = step_s[2:] or step_s
    out: dict[str, Any] = {
        "prefill_s": round(prefill_s, 3),
        "prefill_tokens_per_s": round(ids.numel() / prefill_s, 1),
        "step_ms": round(1000 * statistics.median(warm), 2),
        "tokens": torch.cat(tokens, dim=1),
        "logits": torch.stack(kept, dim=1),
    }
    if device.type == "cuda":
        out["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    return out


def fit_line(points: list[tuple[int, float]]) -> dict[str, float] | None:
    if len({x for x, _ in points}) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum(
        (x - mean_x) ** 2 for x, _ in points
    )
    return {
        "fixed_ms": round(mean_y - slope * mean_x, 2),
        "ms_per_1000_cache_tokens": round(1000 * slope, 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Decode step time against cache size, for each attention choice"
    )
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=list(DTYPES))
    parser.add_argument("--attention", nargs="+", default=["auto", "lean"])
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=[(7, 1024), (7, 4096), (7, 8192), (16, 2048)],
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.allow_busy_gpu:
        from tools.pilot import other_gpu_processes

        busy = other_gpu_processes()
        if busy:
            raise SystemExit("another process is using the GPU:\n" + "\n".join(busy))
    dtype = (
        DTYPES[args.dtype]
        if args.dtype
        else (torch.bfloat16 if device.type == "cuda" else torch.float32)
    )
    model = load_base(args.base, dtype=dtype, device=device)
    model.eval()
    pad = int(model.config.bos_token_id or 0)
    rows_out: list[dict[str, Any]] = []
    reference: dict[tuple[int, int], dict[str, Any]] = {}
    for choice in args.attention:
        use_attention(model, choice)
        for rows, width in args.shapes:
            ids, mask = padded_batch(rows, width, model.config.vocab_size, pad, args.seed)
            first = reference.get((rows, width))
            forced = first["tokens"].to(device) if first else None
            result = run_shape(model, ids.to(device), mask.to(device), args.steps, forced)
            record = {
                "attention": choice,
                "rows": rows,
                "width": width,
                "cache_tokens": rows * width,
                **{k: v for k, v in result.items() if k not in ("tokens", "logits")},
            }
            if first is None:
                reference[(rows, width)] = result
            else:
                diff = (result["logits"] - first["logits"]).abs().max()
                same = result["logits"].argmax(-1) == first["logits"].argmax(-1)
                record["max_abs_logit_diff_vs_first"] = round(float(diff), 5)
                record["same_argmax_rate"] = round(float(same.float().mean()), 5)
            rows_out.append(record)
            print(json.dumps(record), flush=True)
    fits = {
        choice: fit_line(
            [(r["cache_tokens"], r["step_ms"]) for r in rows_out if r["attention"] == choice]
        )
        for choice in args.attention
    }
    report = {
        "base": args.base,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype).removeprefix("torch."),
        "steps": args.steps,
        "step_ms_model": "fixed_ms + ms_per_1000_cache_tokens * rows * width / 1000",
        "fits": fits,
        "shapes": rows_out,
    }
    print(json.dumps({"fits": fits}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
