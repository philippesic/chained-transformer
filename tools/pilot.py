from __future__ import annotations

import argparse
import inspect
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

ARMS = ("sft", "full", "lora", "moe", "separate")
SEPARATE_MODES = ("think", "code", "answer")
BENCHMARKS = ("math500", "humanevalplus", "livecodebench")
LR = {"sft": 1e-5, "full": 1e-5, "lora": 1e-4, "moe": 1e-5, "separate": 1e-5}
KILL_MARGIN_POINTS = 2.0
LORA_TIE_POINTS = 1.0
DONE = "done"


@dataclass
class PilotConfig:
    name: str
    data: str
    base: str = "Qwen/Qwen3-0.6B"
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    steps: int = 1000
    seq_len: int = 4096
    grad_accum: int = 8
    warmup: int = 50
    lr: dict[str, float] = field(default_factory=lambda: dict(LR))
    complete_only: bool = True
    lora_rank: int = 64
    moe_top_k: int = 2
    train_extra: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=lambda: list(BENCHMARKS))
    eval_limit: int | None = None
    eval_max_new_tokens: int = 8192
    eval_batch_size: int | str = "auto"
    eval_extra: list[str] = field(default_factory=list)
    device: str = "cuda"


@dataclass
class TrainRun:
    id: str
    arm: str
    seed: int
    mode: str | None
    argv: list[str]


@dataclass
class EvalRun:
    id: str
    arm: str
    seed: int
    needs: list[str]
    argv: list[str]


def train_runs(cfg: PilotConfig, root: Path) -> list[TrainRun]:
    runs: list[TrainRun] = []
    for seed in cfg.seeds:
        for arm in ARMS:
            for mode in SEPARATE_MODES if arm == "separate" else (None,):
                run_id = f"{arm}-s{seed}" + (f"-{mode}" if mode else "")
                flags = {
                    "--arm": arm,
                    "--data": cfg.data,
                    "--out": root / run_id,
                    "--base": cfg.base,
                    "--device": cfg.device,
                    "--steps": cfg.steps,
                    "--seq-len": cfg.seq_len,
                    "--grad-accum": cfg.grad_accum,
                    "--warmup": cfg.warmup,
                    "--lr": cfg.lr[arm],
                    "--seed": seed,
                    "--lora-rank": cfg.lora_rank,
                    "--moe-top-k": cfg.moe_top_k,
                    **({"--mode": mode} if mode else {}),
                }
                argv = [x for k, v in flags.items() for x in (k, str(v))]
                argv += ["--complete-only"] if cfg.complete_only else []
                argv += cfg.train_extra
                runs.append(TrainRun(run_id, arm, seed, mode, argv))
    return runs


def eval_runs(cfg: PilotConfig, root: Path) -> list[EvalRun]:
    runs: list[EvalRun] = []
    for seed in cfg.seeds:
        for arm in ARMS:
            run_id = f"{arm}-s{seed}"
            if arm == "separate":
                needs = [f"{run_id}-{m}" for m in SEPARATE_MODES]
                who = ["--separate", *(str(root / n / "checkpoint") for n in needs)]
            else:
                needs = [run_id]
                who = ["--model", str(root / run_id / "checkpoint")]
            flags = {
                "--max-new-tokens": cfg.eval_max_new_tokens,
                "--batch-size": cfg.eval_batch_size,
                "--device": cfg.device,
                "--out": root / run_id / "eval",
                **({"--limit": cfg.eval_limit} if cfg.eval_limit else {}),
            }
            argv = [*who, "--benchmarks", *cfg.benchmarks, "--greedy"]
            argv += [x for k, v in flags.items() for x in (k, str(v))]
            argv += cfg.eval_extra
            runs.append(EvalRun(run_id, arm, seed, needs, argv))
    return runs


def eval_batch(value: str) -> int | str:
    return value if value == "auto" else int(value)


def load_plan(root: Path) -> tuple[PilotConfig, list[TrainRun], list[EvalRun]]:
    plan = json.loads((root / "plan.json").read_text())
    cfg = PilotConfig(**plan["config"])
    return cfg, train_runs(cfg, root), eval_runs(cfg, root)


def status_path(root: Path, run_id: str, kind: str) -> Path:
    return root / run_id / f"{kind}-status.json"


def read_status(root: Path, run_id: str, kind: str) -> dict[str, Any]:
    path = status_path(root, run_id, kind)
    return json.loads(path.read_text()) if path.exists() else {"state": "pending"}


def write_status(root: Path, run_id: str, kind: str, **fields: Any) -> None:
    path = status_path(root, run_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps({**fields, "updated": now}, indent=2) + "\n")


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def own_command(root: Path, kind: str, run_id: str) -> list[str]:
    return [sys.executable, "-m", "tools.pilot", kind, "--root", str(root), "--run", run_id]


def cmd_plan(args: argparse.Namespace) -> None:
    root: Path = args.root / args.name
    cfg = PilotConfig(
        name=args.name,
        data=str(args.data),
        base=args.base,
        seeds=args.seeds,
        steps=args.steps,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        warmup=args.warmup,
        complete_only=not args.all_rows,
        train_extra=shlex.split(args.train_extra),
        benchmarks=args.benchmarks,
        eval_limit=args.eval_limit,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_batch_size=args.eval_batch_size,
        eval_extra=shlex.split(args.eval_extra),
        device=args.device,
    )
    plan_file = root / "plan.json"
    if plan_file.exists():
        old = json.loads(plan_file.read_text())["config"]
        if old != asdict(cfg) and not args.replace:
            raise SystemExit(f"{plan_file} exists with a different config; pick a new --name")
    trains, evals = train_runs(cfg, root), eval_runs(cfg, root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "config": asdict(cfg),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "tokens_per_run_at_most": cfg.steps * cfg.grad_accum * cfg.seq_len,
        "train": [asdict(r) for r in trains],
        "eval": [asdict(r) for r in evals],
    }
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    lines = ["#!/bin/sh", "set -e", f"# pilot {cfg.name}: run with python -m tools.pilot run"]
    lines += [shlex.join(own_command(root, k, i)) for k, i in ordered_steps(trains, evals)]
    (root / "commands.sh").write_text("\n".join(lines) + "\n")
    print(f"plan: {plan_file}")
    print(f"{len(trains)} training runs, {len(evals)} eval runs, nothing started")
    print(f"at most {plan['tokens_per_run_at_most']:,} tokens per training run")


def surgery_for(cfg: PilotConfig, arm: str) -> Any:
    from chained_transformer.model import Arm, surgery_for_arm
    from chained_transformer.surgery import SurgeryConfig

    base = SurgeryConfig(
        lora_rank=cfg.lora_rank, lora_alpha=2.0 * cfg.lora_rank, moe_top_k=cfg.moe_top_k
    )
    return surgery_for_arm(cast("Arm", arm), base)


def cmd_train_one(args: argparse.Namespace) -> None:
    from torch import nn
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from chained_transformer import train
    from chained_transformer.checkpoint import save_trained

    cfg, trains, _ = load_plan(args.root)
    run = next((r for r in trains if r.id == args.run), None)
    if run is None:
        raise SystemExit(f"no training run {args.run} in the plan")
    if "on_model" not in inspect.signature(train.main).parameters:
        raise SystemExit(
            "chained_transformer.train.main has no on_model hook, so the trained model cannot be "
            "saved; merge cloud/harness first"
        )
    captured: list[nn.Module] = []
    write_status(args.root, run.id, "train", state="training", argv=run.argv)
    train_main: Callable[..., None] = train.main
    train_main(run.argv, on_model=captured.append)
    model = captured[0]
    model.eval()
    extra = {"arm": run.arm, "seed": run.seed, "mode": run.mode, "train_argv": run.argv}
    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    save_trained(
        cast("Qwen3ForCausalLM", model),
        surgery_for(cfg, run.arm),
        args.root / run.id / "checkpoint",
        cfg.base,
        tokenizer,
        extra,
    )
    write_status(args.root, run.id, "train", state=DONE, argv=run.argv)


def cmd_eval_one(args: argparse.Namespace) -> None:
    from evals.run import main as eval_main

    _, trains, evals = load_plan(args.root)
    run = next((r for r in evals if r.id == args.run), None)
    if run is None:
        raise SystemExit(f"no eval run {args.run} in the plan")
    missing = [n for n in run.needs if read_status(args.root, n, "train")["state"] != DONE]
    if missing:
        raise SystemExit(f"{run.id} needs trained checkpoints first: {missing}")
    write_status(args.root, run.id, "eval", state="evaluating", argv=run.argv)
    eval_main(run.argv)
    write_status(args.root, run.id, "eval", state=DONE, argv=run.argv)


def other_gpu_processes() -> list[str] | None:
    query = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory"]
    try:
        out = subprocess.run([*query, "--format=csv,noheader"], capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def ordered_steps(trains: list[TrainRun], evals: list[EvalRun]) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    trained: set[str] = set()
    waiting = list(evals)
    for r in trains:
        steps.append(("train-one", r.id))
        trained.add(r.id)
        ready = [e for e in waiting if set(e.needs) <= trained]
        steps += [("eval-one", e.id) for e in ready]
        waiting = [e for e in waiting if e not in ready]
    return steps + [("eval-one", e.id) for e in waiting]


def pending_steps(
    root: Path, trains: list[TrainRun], evals: list[EvalRun], only: list[str]
) -> list[tuple[str, str]]:
    def wanted(run_id: str) -> bool:
        return not only or any(run_id == o or run_id.startswith(f"{o}-") for o in only)

    kinds = {"train-one": "train", "eval-one": "eval"}
    return [
        (k, i)
        for k, i in ordered_steps(trains, evals)
        if wanted(i) and read_status(root, i, kinds[k])["state"] != DONE
    ]


def cmd_run(args: argparse.Namespace) -> None:
    cfg, trains, evals = load_plan(args.root)
    todo = pending_steps(args.root, trains, evals, args.only)
    if args.dry_run or not todo:
        for kind, run_id in todo:
            print(shlex.join(own_command(args.root, kind, run_id)))
        print(f"{len(todo)} steps pending" + (" (dry run, nothing started)" if todo else ""))
        return
    for kind, run_id in todo:
        if cfg.device.startswith("cuda") and not args.allow_busy_gpu:
            busy = other_gpu_processes()
            if busy:
                print("another process is using the GPU; stopping without touching it:")
                print("\n".join(busy))
                raise SystemExit(3)
        log_path = args.root / run_id / f"{kind.split('-')[0]}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{kind} {run_id} -> {log_path}", flush=True)
        with log_path.open("a") as log:
            code = subprocess.run(
                own_command(args.root, kind, run_id), stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if code != 0:
            status_kind = "train" if kind == "train-one" else "eval"
            write_status(args.root, run_id, status_kind, state="failed", returncode=code)
            print(f"{kind} {run_id} failed with exit code {code}; see {log_path}")
            if not args.keep_going:
                raise SystemExit(code)


def last_json_line(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def cmd_status(args: argparse.Namespace) -> None:
    cfg, trains, evals = load_plan(args.root)
    print(f"{'run':<22} {'train':<10} {'step':>9} {'loss':>8} {'peak GiB':>9}")
    for r in trains:
        state = read_status(args.root, r.id, "train")["state"]
        last = last_json_line(args.root / r.id / "train.jsonl") or {}
        step = f"{last.get('step', -1) + 1}/{cfg.steps}"
        print(
            f"{r.id:<22} {state:<10} {step:>9} {last.get('loss', ''):>8} "
            f"{last.get('peak_alloc_gib', ''):>9}"
        )
    print()
    for e in evals:
        print(f"{e.id:<22} eval {read_status(args.root, e.id, 'eval')['state']}")


def eval_scores(root: Path, run_id: str) -> dict[str, float] | None:
    report = root / run_id / "eval" / "report.json"
    if not report.exists():
        return None
    benches = json.loads(report.read_text()).get("benchmarks", {})
    scores = {
        name: 100.0 * float(b["scores"]["accuracy"])
        for name, b in benches.items()
        if b.get("scores", {}).get("accuracy") is not None
    }
    return scores or None


def verdict(table: dict[str, dict[int, dict[str, float]]], seeds: list[int]) -> dict[str, Any]:
    def primary(arm: str, seed: int) -> float | None:
        scores = table.get(arm, {}).get(seed)
        return sum(scores.values()) / len(scores) if scores else None

    margins: dict[str, dict[str, float | None]] = {}
    beats: list[bool] = []
    ties: list[bool] = []
    for seed in seeds:
        full, sft, moe, lora = (primary(a, seed) for a in ("full", "sft", "moe", "lora"))
        if full is None or sft is None or moe is None:
            margins[str(seed)] = {"full_minus_sft": None, "full_minus_moe": None}
            continue
        beats.append(min(full - sft, full - moe) >= KILL_MARGIN_POINTS)
        m: dict[str, float | None] = {
            "full_minus_sft": round(full - sft, 2),
            "full_minus_moe": round(full - moe, 2),
        }
        if lora is not None:
            m["lora_minus_full"] = round(lora - full, 2)
            ties.append(abs(lora - full) <= LORA_TIE_POINTS)
        margins[str(seed)] = m
    complete = len(beats) == len(seeds)
    if not complete:
        kill = "incomplete: some arms have no eval report yet"
    elif all(beats):
        kill = f"full beats sft and moe by at least {KILL_MARGIN_POINTS} points on every seed"
    elif not any(beats):
        kill = f"kill rule met: full is not {KILL_MARGIN_POINTS} points ahead on any seed"
    else:
        kill = "mixed: full clears the margin on some seeds only"
    lora_note = None
    if complete and len(ties) == len(seeds):
        lora_note = (
            f"lora is within {LORA_TIE_POINTS} point of full on every seed"
            if all(ties)
            else "lora does not tie full on every seed"
        )
    return {
        "primary": "mean accuracy over the benchmarks run, in points",
        "margins": margins,
        "kill_rule": kill,
        "lora": lora_note,
    }


def cmd_compare(args: argparse.Namespace) -> None:
    cfg, _, evals = load_plan(args.root)
    table: dict[str, dict[int, dict[str, float]]] = {}
    for e in evals:
        scores = eval_scores(args.root, e.id)
        if scores is not None:
            table.setdefault(e.arm, {})[e.seed] = scores
    result = {
        "pilot": cfg.name,
        "scores_points": {a: {str(s): v for s, v in by.items()} for a, by in table.items()},
        **verdict(table, cfg.seeds),
    }
    text = json.dumps(result, indent=2) + "\n"
    (args.root / "compare.json").write_text(text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan and run the five-arm pilot")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="write plan.json and commands.sh; starts nothing")
    plan.add_argument("--name", required=True)
    plan.add_argument("--data", type=Path, required=True)
    plan.add_argument("--root", type=Path, default=Path("artifacts/pilot"))
    plan.add_argument("--base", default="Qwen/Qwen3-0.6B")
    plan.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    plan.add_argument("--steps", type=int, default=1000)
    plan.add_argument("--seq-len", type=int, default=4096)
    plan.add_argument("--grad-accum", type=int, default=8)
    plan.add_argument("--warmup", type=int, default=50)
    plan.add_argument("--all-rows", action="store_true", help="train on incomplete rows too")
    plan.add_argument("--train-extra", default="", help='train.py flags: --train-extra="..."')
    plan.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    plan.add_argument("--eval-limit", type=int)
    plan.add_argument("--eval-max-new-tokens", type=int, default=8192)
    plan.add_argument(
        "--eval-batch-size", type=eval_batch, default="auto", help="N, or auto to fill the GPU"
    )
    plan.add_argument("--eval-extra", default="", help='evals.run flags: --eval-extra="..."')
    plan.add_argument("--device", default="cuda")
    plan.add_argument("--replace", action="store_true", help="overwrite a plan with this name")
    plan.set_defaults(func=cmd_plan)

    for name, func in (("train-one", cmd_train_one), ("eval-one", cmd_eval_one)):
        one = sub.add_parser(name)
        one.add_argument("--root", type=Path, required=True)
        one.add_argument("--run", required=True)
        one.set_defaults(func=func)

    run = sub.add_parser("run", help="train and evaluate every pending run, one at a time")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--only", nargs="*", default=[], help="run ids or prefixes, e.g. full-s0")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-going", action="store_true")
    run.add_argument("--allow-busy-gpu", action="store_true")
    run.set_defaults(func=cmd_run)

    for name, func in (("status", cmd_status), ("compare", cmd_compare)):
        view = sub.add_parser(name)
        view.add_argument("--root", type=Path, required=True)
        view.set_defaults(func=func)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
