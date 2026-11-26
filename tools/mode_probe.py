from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from chained_transformer.modes import MODE_NAMES, segment_labels


def pick_rows(ds: Any, max_tokens: int, rows: int, seed: int) -> list[int]:
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    chosen = []
    for i in order:
        row = ds[i]
        if row.get("complete", True) and len(row["input_ids"]) <= max_tokens:
            chosen.append(i)
        if len(chosen) == rows:
            break
    return chosen


@torch.no_grad()
def collect(
    model: Any,
    ds: Any,
    indices: list[int],
    layers: list[int],
    per_label: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    feats: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    labels: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    for row_id, i in enumerate(indices):
        row = ds[i]
        ids = torch.tensor([row["input_ids"]], device=device)
        position = torch.tensor(row["mode_ids"])
        segment = torch.tensor(segment_labels(row["mode_ids"]))
        mask = torch.tensor(row["loss_mask"], dtype=torch.bool)
        keep = []
        for label in range(len(MODE_NAMES)):
            where = (mask & (position == label)).nonzero().squeeze(1)
            if where.numel() == 0:
                continue
            take = where[torch.randperm(where.numel(), generator=gen)[:per_label]]
            keep.append(take)
        if not keep:
            continue
        pos = torch.cat(keep)
        out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for layer in layers:
            feats[layer].append(out.hidden_states[layer][0, pos.to(device)].float().cpu())
        labels.append(torch.stack([position[pos], segment[pos]], dim=1))
        groups.append(torch.full((pos.numel(),), row_id))
    return (
        {k: torch.cat(v) for k, v in feats.items()},
        torch.cat(labels),
        torch.cat(groups),
    )


def balanced_accuracy(pred: torch.Tensor, gold: torch.Tensor, classes: int) -> float:
    scores = []
    for c in range(classes):
        sel = gold == c
        if sel.any():
            scores.append(float((pred[sel] == c).float().mean()))
    return sum(scores) / len(scores) if scores else 0.0


def train_probe(
    x: torch.Tensor, y: torch.Tensor, classes: int, steps: int, weight_decay: float
) -> torch.nn.Linear:
    mean, std = x.mean(0), x.std(0).clamp(min=1e-6)
    probe = torch.nn.Linear(x.shape[1], classes)
    xs = (x - mean) / std
    counts = torch.bincount(y, minlength=classes).float().clamp(min=1)
    weight = counts.sum() / (classes * counts)
    opt = torch.optim.LBFGS(probe.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(probe(xs), y, weight=weight)
        loss = loss + weight_decay * probe.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        probe.weight.div_(std)
        probe.bias.sub_((probe.weight * mean).sum(1))
    return probe


def evaluate(
    feats: dict[int, torch.Tensor],
    labels: torch.Tensor,
    groups: torch.Tensor,
    target: int,
    steps: int,
    weight_decay: float,
) -> dict[str, Any]:
    classes = len(MODE_NAMES)
    y = labels[:, target]
    test = groups % 5 == 0
    out: dict[str, Any] = {}
    for layer, x in feats.items():
        probe = train_probe(x[~test], y[~test], classes, steps, weight_decay)
        with torch.no_grad():
            pred = probe(x[test]).argmax(1)
        out[str(layer)] = round(balanced_accuracy(pred, y[test], classes), 4)
    majority = torch.bincount(y[~test], minlength=classes).argmax()
    out["majority_baseline"] = round(
        balanced_accuracy(torch.full_like(y[test], int(majority)), y[test], classes), 4
    )
    out["test_tokens"] = int(test.sum())
    out["test_by_label"] = {MODE_NAMES[c]: int((y[test] == c).sum()) for c in range(classes)}
    return out


def main(argv: list[str] | None = None) -> None:
    from datasets import load_from_disk

    from chained_transformer.model import BASE_MODEL, load_base

    parser = argparse.ArgumentParser(description="Linear probes: which mode is this token in")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--per-label", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/probes/mode-probe.json"))
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_base(args.base, dtype=getattr(torch, args.dtype), device=device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = args.layers or sorted({0, 1, 2, 4, 8, 12, 16, 20, 24, n_layers})
    ds = load_from_disk(str(args.data))
    indices = pick_rows(ds, args.max_tokens, args.rows, args.seed)
    feats, labels, groups = collect(model, ds, indices, layers, args.per_label, device, args.seed)
    report = {
        "base": args.base,
        "rows": len(indices),
        "tokens": int(labels.shape[0]),
        "layers": layers,
        "position_mode": evaluate(feats, labels, groups, 0, args.steps, args.weight_decay),
        "segment_label": evaluate(feats, labels, groups, 1, args.steps, args.weight_decay),
        "note": "balanced accuracy on held-out rows (every 5th row); layer 0 is the embedding",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
