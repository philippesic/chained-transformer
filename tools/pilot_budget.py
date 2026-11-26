from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chained_transformer.memory import QWEN3_0_6B, ModelShape
from tools.pilot import (
    ARMS,
    BENCHMARKS,
    PilotConfig,
    arms_of,
    first_sample,
    load_plan,
    other_gpu_processes,
)

PHASES = ("forward", "backward", "optimizer")
PROBLEMS = {"math500": 500, "humanevalplus": 164, "livecodebench": 880}
RUNS_PER_SEED = {"sft": 1, "full": 1, "lora": 1, "moe": 1, "separate": 3}
PROMPT_TOKENS = 600
OVERHEAD_GIB = 1.5
GiB = 2**30


@dataclass(frozen=True)
class Gpu:
    name: str
    tflops: float
    mfu: tuple[float, float]
    decode_launch_s: tuple[float, float]
    bandwidth_gbs: float
    memory_gib: float


GPUS = {
    "2080ti": Gpu("RTX 2080 Ti", 13.4, (0.3, 0.6), (0.025, 0.045), 616.0, 11.0),
    "5080": Gpu("RTX 5080", 112.0, (0.25, 0.45), (0.02, 0.035), 960.0, 16.0),
}
ASSUMPTIONS = [
    "RTX 2080 Ti: bf16 matmuls have no tensor cores on Turing, so the peak is taken as the "
    "13.4 TFLOPS fp32 rate, at 30 to 60% utilisation.",
    "RTX 5080: about 112 TFLOPS dense bf16 with fp32 accumulation, at 25 to 45% utilisation, "
    "with the whole card free (BitDraft shares it).",
    "Training FLOPs per token: forward, backward with weight gradients only for trainable "
    "weights, one more forward for activation and loss checkpointing, and causal attention.",
    "Every row is assumed to fill --seq-len unless the plan's data is readable, in which case "
    "the mean row length is used.",
    "Decoding: Hugging Face generate is launch-bound, about 20 to 45 ms per step for 28 layers, "
    "plus weight and KV-cache reads at the card's bandwidth.",
    "Without refill a batch runs as long as its longest response: 1.5 times the mean at best, "
    "the token cap at worst. With refill (the default) finished rows are replaced once half "
    "the batch is done, so at least 75% (best) or 50% (worst) of the batch is decoding, and "
    "never less than without refill; each benchmark ends with one longest-response tail, and "
    "every problem is prefilled once (best) or twice (worst) at the training utilisation.",
    "Responses average the --mean-new-tokens value or the token cap, whichever is lower.",
    "separate decodes at batch 1; the other arms use the auto batch from free memory.",
    "Scoring is left out: HumanEval+ took about 22 s per eval run on this CPU (measured) and "
    "MATH-500 is string matching; LiveCodeBench scoring is not measured (its data is on the Hub).",
]


def layer_params(shape: ModelShape) -> tuple[int, int, int]:
    attn = 2 * shape.hidden * shape.q_width + 2 * shape.hidden * shape.kv_width
    mlp = 3 * shape.hidden * shape.intermediate
    return attn, mlp, shape.vocab * shape.hidden


def train_flops_per_token(
    shape: ModelShape, arm: str, seq_len: float, top_k: int = 2, rank: int = 64
) -> float:
    attn, mlp, head = layer_params(shape)
    adapter = 3 * rank * (shape.hidden + shape.intermediate) if arm == "lora" else 0
    touched = shape.layers * (attn + mlp * (top_k if arm == "moe" else 1) + adapter) + head
    trained = touched - (shape.layers * mlp if arm == "lora" else 0)
    forward = 2 * touched
    backward = 2 * touched + 2 * trained
    recompute = 2 * touched
    attention = 4 * shape.layers * 4 * shape.q_width * (seq_len + 1) / 2
    return float(forward + backward + recompute + attention)


def kv_bytes_per_token(shape: ModelShape) -> int:
    return 2 * shape.layers * shape.kv_width * 2


def weight_bytes(shape: ModelShape, arm: str, rank: int = 64) -> int:
    attn, mlp, head = layer_params(shape)
    extra = {
        "full": 2 * mlp,
        "moe": 2 * mlp,
        "lora": 3 * 3 * rank * (shape.hidden + shape.intermediate),
    }
    return 2 * (shape.layers * (attn + mlp + extra.get(arm, 0)) + head)


def auto_batch(shape: ModelShape, gpu: Gpu, arm: str, cap: int, wanted: int = 16) -> int:
    if arm == "separate":
        return 1
    free = (gpu.memory_gib - OVERHEAD_GIB) * GiB - weight_bytes(shape, arm)
    per_sequence = kv_bytes_per_token(shape) * (PROMPT_TOKENS + cap)
    return max(1, min(wanted, int(free * 0.8) // per_sequence))


def mean_row_tokens(cfg: PilotConfig, sample: int = 2000) -> float | None:
    path = Path(cfg.data)
    if not (path / "dataset_info.json").exists():
        return None
    from datasets import load_from_disk

    data = load_from_disk(str(path))
    if cfg.complete_only and "complete" in data.column_names:
        data = data.filter(lambda r: r["complete"])
    rows = data.select(range(min(sample, len(data))))
    lengths = [min(len(ids) - 1, cfg.seq_len) for ids in rows["input_ids"]]
    return statistics.fmean(lengths) if lengths else None


def load_measured(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    if isinstance(raw, dict):
        for arm, r in raw["arms"].items():
            out[arm] = {
                "seq_len": float(raw["seq_len"]),
                "micro_s": r["phases"]["forward"] + r["phases"]["backward"],
                "optimizer_s": r["phases"]["optimizer"],
            }
        return out
    for entry in raw:
        summary = entry.get("train_summary") or {}
        if "median_step_s" in summary and entry["arm"] not in out:
            out[entry["arm"]] = {
                "seq_len": float(entry["seq_len"]),
                "micro_s": float(summary["median_step_s"]),
                "optimizer_s": 0.0,
            }
    return out


def train_hours(
    cfg: PilotConfig,
    shape: ModelShape,
    gpu: Gpu,
    arm: str,
    row_tokens: float,
    measured: dict[str, float] | None,
) -> tuple[float, float]:
    runs = RUNS_PER_SEED[arm] * len(cfg.seeds)
    micro_steps = cfg.steps * cfg.grad_accum
    flops = train_flops_per_token(shape, arm, row_tokens, cfg.moe_top_k, cfg.lora_rank)
    if measured is not None:
        then = train_flops_per_token(shape, arm, measured["seq_len"], cfg.moe_top_k, cfg.lora_rank)
        scale = flops * row_tokens / (then * measured["seq_len"])
        seconds = cfg.steps * measured["optimizer_s"] + micro_steps * measured["micro_s"] * scale
        return runs * seconds / 3600, runs * seconds / 3600
    tokens = micro_steps * row_tokens
    lo, hi = (tokens * flops / (gpu.tflops * 1e12 * m) for m in reversed(gpu.mfu))
    return runs * lo / 3600, runs * hi / 3600


def eval_problems(cfg: PilotConfig) -> int:
    sizes = {b: PROBLEMS.get(b, 0) for b in cfg.benchmarks}
    if cfg.eval_set == "verdict" and cfg.eval_sample:
        sizes = {b: min(n, cfg.eval_sample.get(b, n)) for b, n in sizes.items()}
    if cfg.eval_limit:
        sizes = {b: min(n, cfg.eval_limit) for b, n in sizes.items()}
    return sum(sizes.values())


def eval_hours(
    cfg: PilotConfig, shape: ModelShape, gpu: Gpu, arm: str, mean_tokens: float
) -> tuple[float, float, int]:
    cap = cfg.eval_max_new_tokens
    batch = auto_batch(shape, gpu, arm, cap)
    if isinstance(cfg.eval_batch_size, int):
        batch = 1 if arm == "separate" else cfg.eval_batch_size
    problems = eval_problems(cfg)
    mean = min(mean_tokens, cap)
    refill = batch > 1 and "--no-refill" not in cfg.eval_extra
    benches = len(cfg.benchmarks)
    prefill_flops = 2 * weight_bytes(shape, arm) / 2 * problems * (PROMPT_TOKENS + mean / 2)
    out = []
    for launch, longest, floor, prefills, mfu in (
        (gpu.decode_launch_s[0], min(cap, 1.5 * mean), 0.75, 1, gpu.mfu[1]),
        (gpu.decode_launch_s[1], cap, 0.5, 2, gpu.mfu[0]),
    ):
        longest = mean if batch == 1 else longest
        reads = weight_bytes(shape, arm) + batch * kv_bytes_per_token(shape) * (
            PROMPT_TOKENS + longest / 2
        )
        step = launch + reads / (gpu.bandwidth_gbs * 1e9)
        if refill:
            busy = max(mean / longest, floor)
            steps = problems * mean / (busy * batch) + benches * longest
        else:
            steps, prefills = math.ceil(problems / batch) * longest, 1
        prefill = prefills * prefill_flops / (gpu.tflops * 1e12 * mfu)
        out.append(len(cfg.seeds) * (steps * step + prefill) / 3600)
    return out[0], out[1], batch


def span(lo: float, hi: float) -> str:
    return f"{lo:.1f}" if abs(hi - lo) < 0.05 else f"{lo:.1f} to {hi:.1f}"


def days(hours: list[float]) -> str:
    return f"{span(*hours)} hours ({span(hours[0] / 24, hours[1] / 24)} days)"


def plan_hours(
    cfg: PilotConfig,
    shape: ModelShape,
    gpu: Gpu,
    measured: dict[str, dict[str, float]],
    tokens: float,
    mean_tokens: float,
) -> tuple[dict[str, Any], list[list[str]], list[float]]:
    per_arm: dict[str, Any] = {}
    rows = []
    total = [0.0, 0.0]
    for arm in arms_of(cfg):
        source = measured.get(arm)
        t_lo, t_hi = train_hours(cfg, shape, gpu, arm, tokens, source)
        e_lo, e_hi, batch = eval_hours(cfg, shape, gpu, arm, mean_tokens)
        total[0] += t_lo + e_lo
        total[1] += t_hi + e_hi
        per_arm[arm] = {
            "train_hours": [round(t_lo, 2), round(t_hi, 2)],
            "train_source": "measured" if source else "flops",
            "eval_hours": [round(e_lo, 2), round(e_hi, 2)],
            "eval_batch": batch,
        }
        rows.append(
            [
                arm,
                str(RUNS_PER_SEED[arm] * len(cfg.seeds)),
                span(t_lo, t_hi) + (" (measured)" if source else ""),
                str(batch),
                span(e_lo, e_hi),
                span(t_lo + e_lo, t_hi + e_hi),
            ]
        )
    return per_arm, rows, [round(total[0], 1), round(total[1], 1)]


def estimate(
    cfg: PilotConfig,
    gpus: list[str],
    measured: dict[str, dict[str, dict[str, float]]],
    mean_tokens: float,
    row_tokens: float | None,
    shape: ModelShape = QWEN3_0_6B,
) -> tuple[dict[str, Any], str]:
    rows_known = row_tokens is not None
    tokens = row_tokens if row_tokens is not None else float(cfg.seq_len)
    verdict_mode = cfg.eval_set == "verdict"
    grown = replace(cfg, eval_sample=None)
    widened = replace(cfg, eval_set="full", eval_sample=None)
    result: dict[str, Any] = {
        "estimate": True,
        "assumptions": ASSUMPTIONS,
        "row_tokens": tokens,
        "mean_new_tokens": mean_tokens,
        "eval_set": cfg.eval_set,
        "problems_per_eval": eval_problems(cfg),
        "gpus": {},
    }
    refill = "off" if "--no-refill" in cfg.eval_extra else "on"
    lines = [
        "# Pilot GPU time: an estimate, not a measurement",
        "",
        f"Plan: {len(cfg.seeds)} seeds, {cfg.steps} steps x {cfg.grad_accum} rows of "
        f"{tokens:.0f} tokens ({'mean of the data' if rows_known else 'every row full'}), "
        f"benchmarks {', '.join(cfg.benchmarks)}"
        + (f" (first {cfg.eval_limit} problems each)" if cfg.eval_limit else "")
        + f", {eval_problems(cfg)} problems per eval run ({cfg.eval_set} set), up to "
        f"{cfg.eval_max_new_tokens} new tokens with an assumed mean of {mean_tokens:.0f}, "
        f"refill {refill}.",
        "",
    ]
    for key in gpus:
        gpu = GPUS[key]
        got = measured.get(key, {})
        per_arm, rows, planned = plan_hours(cfg, shape, gpu, got, tokens, mean_tokens)
        everything = plan_hours(widened, shape, gpu, got, tokens, mean_tokens)[2]
        info: dict[str, Any] = {"name": gpu.name, "arms": per_arm, "total_hours": everything}
        if verdict_mode:
            full_sample = plan_hours(grown, shape, gpu, got, tokens, mean_tokens)[2]
            info |= {"verdict_hours": planned, "verdict_full_suite_hours": full_sample}
            summary = [
                f"Kill-rule verdict as planned (sft, full, lora and moe, "
                f"{eval_problems(cfg)} problems per eval): about {days(planned)}.",
                "",
                "If the arms disagree on 15% or more of the problems, the sample grows to the "
                f"whole suite: about {days(full_sample)}.",
                "",
                f"Widened to everything, separate included: about {days(everything)} in total.",
            ]
        else:
            verdict = [0.0, 0.0]
            for arm, v in per_arm.items():
                if arm != "separate":
                    verdict[0] += v["train_hours"][0] + v["eval_hours"][0]
                    verdict[1] += v["train_hours"][1] + v["eval_hours"][1]
            info["verdict_hours"] = [round(verdict[0], 1), round(verdict[1], 1)]
            summary = [
                f"Everything: about {days(planned)}.",
                "",
                "The kill-rule verdict needs every arm but separate, which runs last: about "
                f"{span(*info['verdict_hours'])} hours.",
            ]
        result["gpus"][key] = info
        lines += [
            f"## {gpu.name}",
            "",
            *summary,
            "",
            "| arm | train runs | train h | eval batch | eval h | total h |",
            "|---|---|---|---|---|---|",
            *("| " + " | ".join(r) + " |" for r in rows),
            "",
        ]
    lines += ["## Assumptions", "", *(f"- {a}" for a in ASSUMPTIONS), ""]
    return result, "\n".join(lines)


def default_config() -> PilotConfig:
    return PilotConfig(
        name="default", data="", eval_set="verdict", eval_sample=first_sample(list(BENCHMARKS))
    )


class PhaseTimer:
    def __init__(self, device: str) -> None:
        import torch

        self.torch = torch
        self.cuda = device.startswith("cuda")
        self.current: str | None = None
        self.last = 0.0
        self.now: dict[str, float] = {}
        self.steps: list[dict[str, float]] = []

    def __call__(self, name: str) -> None:
        if self.cuda:
            self.torch.cuda.synchronize()
        t = time.perf_counter()
        if self.current in PHASES:
            self.now[self.current] = self.now.get(self.current, 0.0) + t - self.last
        if name == "idle":
            self.steps.append(self.now)
            self.now = {}
        self.current, self.last = name, t

    def medians(self, skip: int) -> dict[str, float]:
        steady = self.steps[skip:] or self.steps
        return {p: round(statistics.median(s.get(p, 0.0) for s in steady), 4) for p in PHASES}


def cmd_measure_one(args: argparse.Namespace) -> None:
    from chained_transformer.train import main as train_main

    timer = PhaseTimer(args.device)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        argv = [
            "--arm",
            args.arm,
            "--data",
            str(args.data),
            "--out",
            str(out),
            "--steps",
            str(args.steps),
            "--seq-len",
            str(args.seq_len),
            "--device",
            args.device,
            "--base",
            str(args.base),
            "--warmup",
            "1",
        ]
        if args.arm == "separate":
            argv += ["--mode", "think"]
        train_main(argv, on_phase=timer)
        summary = json.loads((out / "summary.json").read_text())
    result = {
        "arm": args.arm,
        "median_step_s": summary["median_step_s"],
        "tokens_per_s": summary["tokens_per_s"],
        "peak_reserved_gib": summary["peak_reserved_gib"],
        "device": summary["device"],
        "phases": timer.medians(skip=2),
    }
    args.result.write_text(json.dumps(result) + "\n")


def cmd_measure(args: argparse.Namespace) -> None:
    from datasets import Dataset

    from tools.memory_check import save_random_model, synthetic_rows

    if args.device.startswith("cuda") and not args.allow_busy_gpu:
        busy = other_gpu_processes()
        if busy:
            print("another process is using the GPU; not measuring:", *busy, sep="\n  ")
            raise SystemExit(3)
    shape = QWEN3_0_6B if args.layers is None else replace(QWEN3_0_6B, layers=args.layers)
    report: dict[str, Any] = {
        "device": args.device,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "layers": shape.layers,
        "base": str(args.base) if args.base else "random weights, Qwen3-0.6B shape",
        "arms": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = args.base
        if base is None:
            base = root / "model"
            save_random_model(shape, base)
        data = root / "data"
        rows = synthetic_rows(args.seq_len, 8, shape.vocab, 0)
        Dataset.from_list(rows).save_to_disk(str(data))
        env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
        for arm in args.arms:
            path = root / f"{arm}.json"
            cmd = [
                sys.executable,
                "-m",
                "tools.pilot_budget",
                "measure-one",
                "--arm",
                arm,
                "--base",
                str(base),
                "--data",
                str(data),
                "--seq-len",
                str(args.seq_len),
                "--steps",
                str(args.steps),
                "--device",
                args.device,
                "--result",
                str(path),
            ]
            start = time.perf_counter()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0 or not path.exists():
                tail = (proc.stderr or proc.stdout)[-2000:]
                report["arms"][arm] = {"error": tail}
                print(f"{arm:<9} FAILED\n{tail}", flush=True)
                continue
            result = json.loads(path.read_text())
            result["wall_s"] = round(time.perf_counter() - start, 1)
            report["arms"][arm] = result
            ph = result["phases"]
            print(
                f"{arm:<9} step {result['median_step_s']:.3f}s  forward {ph['forward']:.3f}s  "
                f"backward {ph['backward']:.3f}s  optimizer {ph['optimizer']:.3f}s",
                flush=True,
            )
    flops = {a: train_flops_per_token(shape, a, args.seq_len) for a in ARMS}
    ok = {a: r for a, r in report["arms"].items() if "error" not in r}
    if "sft" in ok:
        sft = ok["sft"]["phases"]["forward"] + ok["sft"]["phases"]["backward"]
        report["relative_to_sft"] = {
            a: {
                "measured": round((r["phases"]["forward"] + r["phases"]["backward"]) / sft, 3),
                "flops": round(flops[a] / flops["sft"], 3),
            }
            for a, r in ok.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if "relative_to_sft" in report:
        print(json.dumps(report["relative_to_sft"], indent=2))


def cmd_estimate(args: argparse.Namespace) -> None:
    cfg = default_config() if args.root is None else load_plan(args.root)[0]
    measured = {}
    for item in args.measured:
        key, _, path = item.partition("=")
        if key not in GPUS or not path:
            raise SystemExit(f"--measured wants GPU=PATH with GPU in {', '.join(GPUS)}")
        measured[key] = load_measured(Path(path))
    rows = mean_row_tokens(cfg) if args.row_tokens is None else args.row_tokens
    if args.decode_ms is not None:
        lo, hi = sorted(args.decode_ms)
        for key in args.gpus:
            GPUS[key] = replace(GPUS[key], decode_launch_s=(lo / 1000, hi / 1000))
    data, text = estimate(cfg, args.gpus, measured, args.mean_new_tokens, rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(text, end="")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step cost per arm and a pilot GPU-time estimate")
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="time a few training steps per arm")
    measure.add_argument("--device", default="cpu")
    measure.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    measure.add_argument("--seq-len", type=int, default=256)
    measure.add_argument("--steps", type=int, default=4)
    measure.add_argument("--layers", type=int, default=None, help="random model with N layers")
    measure.add_argument("--base", type=Path, default=None, help="real model dir or hub name")
    measure.add_argument("--out", type=Path, default=Path("artifacts/pilot-budget/measure.json"))
    measure.add_argument("--allow-busy-gpu", action="store_true")
    measure.set_defaults(func=cmd_measure)

    one = sub.add_parser("measure-one", help=argparse.SUPPRESS)
    one.add_argument("--arm", choices=ARMS, required=True)
    one.add_argument("--base", required=True)
    one.add_argument("--data", type=Path, required=True)
    one.add_argument("--seq-len", type=int, required=True)
    one.add_argument("--steps", type=int, required=True)
    one.add_argument("--device", required=True)
    one.add_argument("--result", type=Path, required=True)
    one.set_defaults(func=cmd_measure_one)

    est = sub.add_parser("estimate", help="hours for a pilot plan on each GPU")
    est.add_argument("--root", type=Path, help="artifacts/pilot/NAME; default plan if omitted")
    est.add_argument("--gpus", nargs="+", choices=list(GPUS), default=list(GPUS))
    est.add_argument(
        "--measured", nargs="*", default=[], help="GPU=PATH from measure or tools.memory_check"
    )
    est.add_argument("--mean-new-tokens", type=float, default=4000.0)
    est.add_argument("--row-tokens", type=float, help="mean training row length")
    est.add_argument(
        "--decode-ms",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="ms per decode step, e.g. 1000 x batch_size / tokens_per_s from a GPU eval report",
    )
    est.add_argument("--out", type=Path)
    est.set_defaults(func=cmd_estimate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
