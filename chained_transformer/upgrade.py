from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.checkpoint import load_checkpoint, save_checkpoint
from chained_transformer.model import Built
from chained_transformer.modes import MODE_NAMES, Mode
from chained_transformer.surgery import Routing, freeze_all_but_mode, mode_parameters
from chained_transformer.train import IGNORE, make_batch, token_losses

PACK_FORMAT = 1
LossOn = Literal["mode", "all"]
Row = dict[str, Any]


def mode_tensors(model: nn.Module, mode: int) -> dict[str, torch.Tensor]:
    wanted = {id(p) for p in mode_parameters(model, mode)}
    if not wanted:
        raise ValueError("model has no per-mode parameters (use a full or lora mode checkpoint)")
    return {n: p.detach() for n, p in model.named_parameters() if id(p) in wanted}


def _variant(model: nn.Module) -> str | None:
    surgery = getattr(getattr(model, "config", None), "mode_surgery", None)
    return surgery.get("variant") if isinstance(surgery, dict) else None


def trunk_fingerprint(model: nn.Module) -> str:
    per_mode = {id(p) for m in range(len(MODE_NAMES)) for p in mode_parameters(model, m)}
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if id(p) in per_mode:
            continue
        digest.update(name.encode())
        digest.update(p.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_mode_pack(
    model: nn.Module, mode: int, path: Path, meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    tensors = {n: t.contiguous().cpu() for n, t in mode_tensors(model, mode).items()}
    header = {
        "format": PACK_FORMAT,
        "mode": MODE_NAMES[mode],
        "variant": _variant(model),
        "trunk_sha256": trunk_fingerprint(model),
        "tensors": len(tensors),
        "params": sum(t.numel() for t in tensors.values()),
        **(meta or {}),
    }
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "mode.safetensors"))
    (path / "mode.json").write_text(json.dumps(header, indent=2) + "\n")
    return header


def load_mode_pack(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    header: dict[str, Any] = json.loads((path / "mode.json").read_text())
    if header.get("format") != PACK_FORMAT:
        raise ValueError(f"unsupported mode pack format {header.get('format')}")
    return load_file(str(path / "mode.safetensors")), header


def swap_mode(
    model: nn.Module,
    mode: int,
    tensors: dict[str, torch.Tensor],
    header: dict[str, Any] | None = None,
    check_trunk: bool = True,
) -> int:
    if header is not None:
        if header.get("variant") != _variant(model):
            raise ValueError(f"pack variant {header.get('variant')} != model {_variant(model)}")
        if header.get("mode") != MODE_NAMES[mode]:
            raise ValueError(f"pack holds mode {header.get('mode')}, not {MODE_NAMES[mode]}")
        if check_trunk and header.get("trunk_sha256") != trunk_fingerprint(model):
            raise ValueError("pack trunk differs from the model; check_trunk=False forces it")
    current = mode_tensors(model, mode)
    if set(current) != set(tensors):
        missing = sorted(set(current) - set(tensors))[:3]
        extra = sorted(set(tensors) - set(current))[:3]
        raise ValueError(f"mode tensors do not match: missing {missing}, unexpected {extra}")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in tensors.items():
            target = params[name]
            if target.shape != value.shape:
                raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(target.shape)}")
            target.copy_(value.to(dtype=target.dtype, device=target.device))
    return sum(t.numel() for t in tensors.values())


@dataclass
class RetrainConfig:
    mode: int = int(Mode.CODE)
    steps: int = 100
    lr: float = 1e-4
    batch_rows: int = 8
    seq_len: int = 4096
    loss_on: LossOn = "mode"
    seed: int = 0
    loss_chunk: int = 1024
    weight_decay: float = 0.0


@dataclass
class RetrainLog:
    trainable: int
    losses: list[float] = field(default_factory=list)


def retrain_mode(
    model: Qwen3ForCausalLM,
    routing: Routing,
    rows: Sequence[Row],
    cfg: RetrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> RetrainLog:
    trainable = freeze_all_but_mode(model, cfg.mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer or torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    device = next(model.parameters()).device
    only = cfg.mode if cfg.loss_on == "mode" else None
    order = list(range(len(rows)))
    rng = random.Random(cfg.seed)
    log = RetrainLog(trainable)
    model.train()
    cursor = len(order)
    for _ in range(cfg.steps):
        total = torch.zeros((), device=device)
        count = 0
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.batch_rows):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch = make_batch(rows[order[cursor]], cfg.seq_len, device, only)
            cursor += 1
            valid = batch.targets.reshape(-1) != IGNORE
            if not bool(valid.any()):
                continue
            routing.set(batch.mode_ids)
            losses = token_losses(model, batch, cfg.loss_chunk)
            loss = losses[valid].sum()
            loss.backward()
            total += loss.detach()
            count += int(valid.sum())
        if count == 0:
            raise ValueError("no trainable tokens in the sampled rows for this mode")
        for p in params:
            if p.grad is not None:
                p.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.losses.append(round(float(total) / count, 5))
        if on_step is not None:
            on_step(len(log.losses) - 1, log.losses[-1])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return log


def _first_code_position(mode_ids: torch.Tensor) -> int:
    hits = (mode_ids == int(Mode.CODE)).nonzero()
    return int(hits[0, 0]) if hits.numel() else mode_ids.numel()


@torch.no_grad()
def _hidden(
    model: Qwen3ForCausalLM, routing: Routing | None, ids: torch.Tensor, mode_ids: torch.Tensor
) -> torch.Tensor:
    if routing is not None:
        routing.set(mode_ids.unsqueeze(0))
    return model.model(input_ids=ids.unsqueeze(0), use_cache=False).last_hidden_state[0]


def _logprobs(model: Qwen3ForCausalLM, hidden: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(model.lm_head(hidden).float(), dim=-1)


@dataclass
class Tally:
    nll: float = 0.0
    correct: int = 0
    tokens: int = 0
    kl: float = 0.0
    max_abs_logprob_diff: float = 0.0

    def add(self, nll: torch.Tensor, correct: torch.Tensor) -> None:
        self.nll += float(nll.sum())
        self.correct += int(correct.sum())
        self.tokens += int(nll.numel())

    def as_dict(self) -> dict[str, float | int | None]:
        n = self.tokens
        return {
            "tokens": n,
            "nll": round(self.nll / n, 5) if n else None,
            "accuracy": round(self.correct / n, 5) if n else None,
            "kl_from_reference": round(self.kl / n, 6) if n else None,
            "max_abs_logprob_diff": self.max_abs_logprob_diff,
        }


GROUPS = ("before_code", "after_code")


@torch.no_grad()
def measure(
    model: Qwen3ForCausalLM,
    routing: Routing | None,
    rows: Sequence[Row],
    seq_len: int = 4096,
    reference: tuple[Qwen3ForCausalLM, Routing | None] | None = None,
    chunk: int = 1024,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    tallies = {(m, g): Tally() for m in MODE_NAMES for g in GROUPS}
    for row in rows:
        batch = make_batch(row, seq_len, device, None)
        ids, modes = batch.input_ids[0], batch.mode_ids[0]
        targets, labels = batch.targets[0], batch.target_labels[0]
        hidden = _hidden(model, routing, ids, modes)
        ref_hidden = None
        if reference is not None:
            to = next(reference[0].parameters()).device
            ref_hidden = _hidden(reference[0], reference[1], ids.to(to), modes.to(to))
        first_code = _first_code_position(modes)
        positions = torch.arange(ids.numel(), device=device)
        valid = targets != IGNORE
        for start in range(0, ids.numel(), chunk):
            span = slice(start, start + chunk)
            logp = _logprobs(model, hidden[span])
            ref = None
            if ref_hidden is not None and reference is not None:
                ref = _logprobs(reference[0], ref_hidden[span]).to(device)
            for m, name in enumerate(MODE_NAMES):
                for g in GROUPS:
                    pos = positions[span]
                    side = pos < first_code if g == "before_code" else pos >= first_code
                    sel = valid[span] & (labels[span] == m) & side
                    if not bool(sel.any()):
                        continue
                    tally = tallies[(name, g)]
                    t = targets[span][sel]
                    lp = logp[sel]
                    tally.add(-lp.gather(1, t.unsqueeze(1)).squeeze(1), lp.argmax(-1) == t)
                    if ref is not None:
                        r = ref[sel]
                        tally.kl += float((r.exp() * (r - lp)).sum())
                        tally.max_abs_logprob_diff = max(
                            tally.max_abs_logprob_diff, float((r - lp).abs().max())
                        )
    by_label: dict[str, Any] = {}
    for name in MODE_NAMES:
        merged = Tally()
        for g in GROUPS:
            part = tallies[(name, g)]
            merged.nll += part.nll
            merged.correct += part.correct
            merged.tokens += part.tokens
            merged.kl += part.kl
            merged.max_abs_logprob_diff = max(
                merged.max_abs_logprob_diff, part.max_abs_logprob_diff
            )
        by_label[name] = {
            "all": merged.as_dict(),
            **{g: tallies[(name, g)].as_dict() for g in GROUPS},
        }
    return by_label


def _rows(path: Path, limit: int | None, complete_only: bool, seed: int) -> list[Row]:
    data = load_from_disk(str(path))
    if complete_only:
        data = data.filter(lambda r: r["complete"])
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    return [data[i] for i in order[: limit or len(order)]]


def _load(path: Path, args: argparse.Namespace) -> Built:
    return load_checkpoint(path, getattr(torch, args.dtype), args.device)


def _tokenizer(path: Path) -> PreTrainedTokenizerBase | None:
    if not (path / "tokenizer_config.json").exists():
        return None
    return AutoTokenizer.from_pretrained(str(path))


def cmd_retrain(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    built = _load(args.checkpoint, args)
    if built.routing is None:
        raise SystemExit(f"{args.checkpoint} is not a mode checkpoint")
    mode = int(Mode[args.mode.upper()])
    rows = _rows(args.data, args.limit, args.complete_only, args.seed)
    cfg = RetrainConfig(
        mode=mode,
        steps=args.steps,
        lr=args.lr,
        batch_rows=args.batch_rows,
        seq_len=args.seq_len,
        loss_on=args.loss_on,
        seed=args.seed,
        loss_chunk=args.loss_chunk,
    )
    model = built.model
    model.config.use_cache = False
    if not args.no_grad_ckpt:
        model.activation_checkpointing_enable({"use_reentrant": False})
    optimizer = None
    if args.device.startswith("cuda") and args.dtype == "bfloat16":
        from torchao.optim import AdamW8bit

        optimizer = AdamW8bit(
            mode_parameters(model, mode), lr=args.lr, weight_decay=0.0, bf16_stochastic_round=True
        )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as log_file:

        def on_step(step: int, loss: float) -> None:
            log_file.write(json.dumps({"step": step, "loss": loss}) + "\n")
            log_file.flush()
            print(f"step={step} loss={loss}", flush=True)

        log = retrain_mode(model, built.routing, rows, cfg, optimizer, on_step)
    meta = {
        "source_checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "rows": len(rows),
        "retrain": {k: v for k, v in vars(cfg).items()},
        "trainable": log.trainable,
        "final_loss": log.losses[-1] if log.losses else None,
    }
    header = save_mode_pack(model, mode, args.out, meta)
    print(json.dumps(header, indent=2))


def cmd_swap(args: argparse.Namespace) -> None:
    built = _load(args.checkpoint, args)
    tensors, header = load_mode_pack(args.pack)
    mode = int(Mode[str(header["mode"]).upper()])
    copied = swap_mode(built.model, mode, tensors, header, check_trunk=not args.force)
    extra = {"swapped_mode": header["mode"], "pack": str(args.pack), "params": copied}
    save_checkpoint(built.model, args.out, _tokenizer(args.checkpoint), extra)
    print(json.dumps(extra))


def cmd_measure(args: argparse.Namespace) -> None:
    built = _load(args.checkpoint, args)
    reference = None
    if args.reference is not None:
        ref = _load(args.reference, args)
        reference = (ref.model, ref.routing)
    rows = _rows(args.data, args.limit, args.complete_only, args.seed)
    report = {
        "checkpoint": str(args.checkpoint),
        "reference": None if args.reference is None else str(args.reference),
        "data": str(args.data),
        "rows": len(rows),
        "by_label": measure(built.model, built.routing, rows, args.seq_len, reference),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrain, swap and measure one mode's MLPs")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")

    def data(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--complete-only", action="store_true")
        p.add_argument("--seq-len", type=int, default=4096)
        p.add_argument("--seed", type=int, default=0)

    retrain = sub.add_parser("retrain", help="train one mode with everything else frozen")
    common(retrain)
    data(retrain)
    retrain.add_argument("--mode", choices=MODE_NAMES, default="code")
    retrain.add_argument("--steps", type=int, default=100)
    retrain.add_argument("--lr", type=float, default=1e-4)
    retrain.add_argument("--batch-rows", type=int, default=8)
    retrain.add_argument("--loss-on", choices=["mode", "all"], default="mode")
    retrain.add_argument("--loss-chunk", type=int, default=1024)
    retrain.add_argument("--no-grad-ckpt", action="store_true")
    retrain.add_argument("--out", type=Path, required=True)
    retrain.set_defaults(func=cmd_retrain)

    swap = sub.add_parser("swap", help="write a checkpoint with one mode replaced by a pack")
    common(swap)
    swap.add_argument("--pack", type=Path, required=True)
    swap.add_argument("--force", action="store_true", help="skip the trunk fingerprint check")
    swap.add_argument("--out", type=Path, required=True)
    swap.set_defaults(func=cmd_swap)

    meas = sub.add_parser("measure", help="per-label loss and drift against a reference")
    common(meas)
    data(meas)
    meas.add_argument("--reference", type=Path)
    meas.add_argument("--out", type=Path)
    meas.set_defaults(func=cmd_measure)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
