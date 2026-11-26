from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch import nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.attention import ATTENTION_CHOICES, attention_backend, use_attention
from chained_transformer.model import ARMS, BASE_MODEL, Arm, build, load_base, param_counts
from chained_transformer.modes import MODE_NAMES, Mode
from chained_transformer.optim import DEFAULT_CHUNK, ChunkedAdamW8bit
from chained_transformer.surgery import LoRALinear, SurgeryConfig, freeze_all_but_mode, moe_aux_loss

IGNORE = -100
OPTIMIZERS = ("chunked", "torchao")
ALLOC_CONF = "expandable_segments:True"
PhaseHook = Callable[[str], None]


@dataclass
class Batch:
    input_ids: torch.Tensor
    mode_ids: torch.Tensor
    targets: torch.Tensor
    target_labels: torch.Tensor


def make_batch(
    row: dict[str, Any], seq_len: int, device: torch.device, only_label: int | None
) -> Batch:
    ids = torch.tensor(row["input_ids"][: seq_len + 1], dtype=torch.long)
    modes = torch.tensor(row["mode_ids"][: seq_len + 1], dtype=torch.long)
    mask = torch.tensor(row["loss_mask"][: seq_len + 1], dtype=torch.bool)
    targets = ids[1:].clone()
    labels = modes[:-1]
    keep = mask[1:]
    if only_label is not None:
        keep &= labels == only_label
    targets[~keep] = IGNORE
    return Batch(
        ids[:-1].unsqueeze(0).to(device),
        modes[:-1].unsqueeze(0).to(device),
        targets.unsqueeze(0).to(device),
        labels.unsqueeze(0).to(device),
    )


def _chunk_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    return F.cross_entropy(logits, targets, ignore_index=IGNORE, reduction="none")


def token_losses(model: Qwen3ForCausalLM, batch: Batch, chunk: int) -> torch.Tensor:
    hidden = model.model(input_ids=batch.input_ids, use_cache=False).last_hidden_state
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = batch.targets.reshape(-1)
    weight = model.lm_head.weight
    parts = []
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start : start + chunk]
        t = targets[start : start + chunk]
        parts.append(checkpoint(_chunk_loss, h, weight, t, use_reentrant=False))
    return torch.cat(parts)


def configure_trainable(
    model: nn.Module, arm: Arm, only_mode: int | None, freeze_embeddings: bool = False
) -> None:
    if only_mode is not None:
        freeze_all_but_mode(model, only_mode)
        return
    for p in model.parameters():
        p.requires_grad_(True)
    if arm == "lora":
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.base.weight.requires_grad_(False)
    if freeze_embeddings:
        embeddings = model.get_input_embeddings()
        head = model.get_output_embeddings()
        for module in (embeddings, head):
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad_(False)


def make_optimizer(
    params: list[nn.Parameter], name: str, lr: float, chunk_numel: int, seed: int
) -> Optimizer:
    if name == "torchao":
        from torchao.optim import AdamW8bit

        optimizer: Optimizer = AdamW8bit(
            params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True
        )
        return optimizer
    return ChunkedAdamW8bit(
        params,
        lr=lr,
        weight_decay=0.0,
        bf16_stochastic_round=True,
        chunk_numel=chunk_numel,
        seed=seed,
    )


def set_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        if isinstance(group["lr"], torch.Tensor):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr


def configure_allocator() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", ALLOC_CONF)


def lr_at(step: int, steps: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_gib(device: torch.device, reserved: bool) -> float:
    if device.type != "cuda":
        return 0.0
    peak = (
        torch.cuda.max_memory_reserved(device)
        if reserved
        else torch.cuda.max_memory_allocated(device)
    )
    return round(peak / 2**30, 3)


def parse_mode(name: str | None) -> int | None:
    return None if name is None else int(Mode[name.upper()])


def main(argv: list[str] | None = None, on_phase: PhaseHook | None = None) -> None:
    configure_allocator()
    phase = on_phase or (lambda name: None)
    parser = argparse.ArgumentParser(description="Train one experiment arm")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-chunk", type=int, default=256)
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--mode", choices=MODE_NAMES, help="separate arm: the mode it trains")
    parser.add_argument("--only-mode", choices=MODE_NAMES, help="train one mode, freeze the rest")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--aux-coef", type=float, default=0.01)
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="chunked")
    parser.add_argument("--optim-chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--attention", choices=ATTENTION_CHOICES, default="auto")
    args = parser.parse_args(argv)
    if (args.arm == "separate") != (args.mode is not None):
        parser.error("--mode is required for --arm separate and only valid there")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_from_disk(str(args.data))
    if args.complete_only:
        rows = rows.filter(lambda r: r["complete"])
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)

    model = load_base(args.base, device=device)
    use_attention(model, args.attention)
    model.config.use_cache = False
    cfg = SurgeryConfig(
        lora_rank=args.lora_rank, lora_alpha=2.0 * args.lora_rank, moe_top_k=args.moe_top_k
    )
    built = build(model, args.arm, cfg)
    configure_trainable(model, args.arm, parse_mode(args.only_mode), args.freeze_embeddings)
    if not args.no_grad_ckpt:
        model.activation_checkpointing_enable({"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, args.optimizer, args.lr, args.optim_chunk, args.seed)
    only_label = parse_mode(args.mode)

    counts = param_counts(model)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "params": counts, "rows": len(rows)}
    (args.out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    log = (args.out / "train.jsonl").open("w")
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    tokens_per_step: list[int] = []
    cursor = 0
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.warmup, args.lr)
        set_lr(optimizer, lr)
        sync(device)
        start = time.perf_counter()
        sums = torch.zeros(len(MODE_NAMES), device=device)
        counts_by_label = torch.zeros(len(MODE_NAMES), device=device)
        aux_total = 0.0
        tokens = 0
        truncated = 0
        for _ in range(args.grad_accum):
            row = rows[order[cursor % len(order)]]
            cursor += 1
            batch = make_batch(row, args.seq_len, device, only_label)
            truncated += len(row["input_ids"]) > args.seq_len + 1
            if built.routing is not None:
                built.routing.set(batch.mode_ids)
            phase("forward")
            losses = token_losses(model, batch, args.loss_chunk)
            valid = batch.targets.reshape(-1) != IGNORE
            loss = losses[valid].sum() / valid.sum().clamp(min=1)
            aux = moe_aux_loss(model)
            if aux is not None:
                loss = loss + args.aux_coef * aux
                aux_total += float(aux.detach())
            phase("backward")
            (loss / args.grad_accum).backward()
            labels = batch.target_labels.reshape(-1)[valid]
            sums.index_add_(0, labels, losses[valid].detach())
            counts_by_label.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))
            tokens += batch.input_ids.numel()
        phase("optimizer")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        phase("idle")
        sync(device)
        elapsed = time.perf_counter() - start
        step_times.append(elapsed)
        tokens_per_step.append(tokens)
        by_label = {
            MODE_NAMES[i]: round(float(sums[i] / counts_by_label[i]), 4)
            for i in range(len(MODE_NAMES))
            if counts_by_label[i] > 0
        }
        record = {
            "step": step,
            "loss": round(float(sums.sum() / counts_by_label.sum().clamp(min=1)), 4),
            "loss_by_label": by_label,
            "aux": round(aux_total / args.grad_accum, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": lr,
            "tokens": tokens,
            "truncated_rows": truncated,
            "step_s": round(elapsed, 3),
            "peak_alloc_gib": peak_gib(device, reserved=False),
            "peak_reserved_gib": peak_gib(device, reserved=True),
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(
            f"step={step} loss={record['loss']} step_s={elapsed:.2f} "
            f"peak={record['peak_alloc_gib']}GiB",
            flush=True,
        )
    log.close()
    steady = step_times[2:] or step_times
    summary = {
        "arm": args.arm,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "params": counts,
        "peak_alloc_gib": peak_gib(device, reserved=False),
        "peak_reserved_gib": peak_gib(device, reserved=True),
        "median_step_s": round(statistics.median(steady), 3),
        "tokens_per_s": round(sum(tokens_per_step[2:] or tokens_per_step) / sum(steady), 1),
        "device": torch.cuda.get_device_name(device) if on_gpu else "cpu",
        "optimizer": args.optimizer,
        "attention": attention_backend(model),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("training complete", json.dumps(summary))


if __name__ == "__main__":
    main()
