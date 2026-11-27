from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk

from chained_transformer.attention import use_attention
from chained_transformer.model import load_base
from chained_transformer.modes import MODE_NAMES
from chained_transformer.train import IGNORE, make_batch, token_losses

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def run_args(run_dir: Path) -> dict[str, str]:
    args: dict[str, str] = json.loads((run_dir / "config.json").read_text())["args"]
    return args


def row_order(n_rows: int, seed: int) -> list[int]:
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    return order


def rows_of_step(step: int, grad_accum: int, order: list[int]) -> list[int]:
    return [order[(step * grad_accum + i) % len(order)] for i in range(grad_accum)]


def parse_steps(items: list[str]) -> list[int]:
    steps: list[int] = []
    for item in items:
        first, sep, last = item.partition("-")
        steps += list(range(int(first), int(last) + 1)) if sep else [int(first)]
    return sorted(set(steps))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@torch.no_grad()
def step_loss(
    model: Any, rows: Any, row_ids: list[int], seq_len: int, loss_chunk: int, device: torch.device
) -> dict[str, Any]:
    sums = torch.zeros(len(MODE_NAMES), dtype=torch.float64)
    counts = torch.zeros(len(MODE_NAMES), dtype=torch.float64)
    domains: list[str] = []
    for row_id in row_ids:
        row = rows[row_id]
        batch = make_batch(row, seq_len, device, None)
        losses = token_losses(model, batch, loss_chunk).double().cpu()
        valid = (batch.targets.reshape(-1) != IGNORE).cpu()
        labels = batch.target_labels.reshape(-1).cpu()[valid]
        sums.index_add_(0, labels, losses[valid])
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        domains.append(str(row["domain"]))
    return {
        "loss": float(sums.sum() / counts.sum().clamp(min=1)),
        "loss_by_label": {
            MODE_NAMES[i]: float(sums[i] / counts[i]) for i in range(len(MODE_NAMES)) if counts[i]
        },
        "tokens_by_label": {MODE_NAMES[i]: int(counts[i]) for i in range(len(MODE_NAMES))},
        "domains": domains,
    }


def summarize(records: list[dict[str, Any]], calibration_below: int) -> dict[str, Any]:
    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
        return {
            "n": len(values),
            "mean": round(statistics.fmean(values), 5),
            "se": None if se is None else round(se, 5),
        }

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = {"steps": [r["step"] for r in rows], "all": stats([r["diff"] for r in rows])}
        for name in MODE_NAMES:
            out[name] = stats(
                [r["diff_by_label"][name] for r in rows if name in r["diff_by_label"]]
            )
        return out

    early = [r for r in records if r["step"] < calibration_below]
    late = [r for r in records if r["step"] >= calibration_below]
    return {
        "diff": "trained minus base, nats per token, on the same rows",
        "calibration": block(early),
        "trained": block(late),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Base-model loss on the rows a training run saw, against its logged loss"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", required=True, help="steps or ranges such as 0-3")
    parser.add_argument("--data", type=Path, help="default: the run's own --data")
    parser.add_argument("--calibration-below", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    train_args = run_args(args.run_dir)
    if train_args["arm"] != "sft" or train_args["mode"] != "None":
        raise SystemExit("only a plain sft run logs a loss comparable with the unrouted base")
    seq_len, grad_accum = int(train_args["seq_len"]), int(train_args["grad_accum"])
    rows = load_from_disk(str(args.data or train_args["data"]))
    if train_args["complete_only"] == "True":
        rows = rows.filter(lambda complete: complete, input_columns="complete")
    order = row_order(len(rows), int(train_args["seed"]))
    logged = {r["step"]: r for r in read_jsonl(args.run_dir / "train.jsonl")}
    wanted = [s for s in parse_steps(args.steps) if s in logged]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["step"]: r for r in read_jsonl(args.out)}
    device = torch.device(args.device)
    model = load_base(train_args["base"], dtype=DTYPES[args.dtype], device=device)
    use_attention(model, train_args["attention"])
    model.config.use_cache = False
    model.eval()
    with args.out.open("a") as sink:
        for step in wanted:
            if step in done:
                continue
            start = time.perf_counter()
            base = step_loss(
                model,
                rows,
                rows_of_step(step, grad_accum, order),
                seq_len,
                int(train_args["loss_chunk"]),
                device,
            )
            trained = logged[step]
            record = {
                "step": step,
                "passes_before": step * grad_accum // len(order),
                "lr": trained["lr"],
                "base": round(base["loss"], 5),
                "trained": trained["loss"],
                "diff": round(trained["loss"] - base["loss"], 5),
                "base_by_label": {k: round(v, 5) for k, v in base["loss_by_label"].items()},
                "diff_by_label": {
                    k: round(trained["loss_by_label"][k] - v, 5)
                    for k, v in base["loss_by_label"].items()
                    if k in trained["loss_by_label"]
                },
                "tokens_by_label": base["tokens_by_label"],
                "domains": base["domains"],
                "seconds": round(time.perf_counter() - start, 1),
            }
            done[step] = record
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(
                json.dumps({k: record[k] for k in ("step", "base", "trained", "diff", "seconds")})
            )
    records = [done[s] for s in sorted(done)]
    summary = summarize(records, args.calibration_below)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
