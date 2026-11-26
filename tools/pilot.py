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
VERDICT_ARMS = ("sft", "full", "lora", "moe")
EVAL_SETS = ("verdict", "full")
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
    eval_max_new_tokens: int = 4096
    eval_set: str = "full"
    eval_sample: dict[str, int] | None = None
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


def arms_of(cfg: PilotConfig) -> tuple[str, ...]:
    return VERDICT_ARMS if cfg.eval_set == "verdict" else ARMS


def train_runs(cfg: PilotConfig, root: Path) -> list[TrainRun]:
    runs: list[TrainRun] = []
    for seed in cfg.seeds:
        for arm in arms_of(cfg):
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
    sample = cfg.eval_sample if cfg.eval_set == "verdict" else None
    for seed in cfg.seeds:
        for arm in arms_of(cfg):
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
            argv = [*who, "--benchmarks", *cfg.benchmarks, "--greedy", "--resume"]
            argv += [x for k, v in flags.items() for x in (k, str(v))]
            if sample:
                argv += ["--sample", *(f"{b}={n}" for b, n in sample.items())]
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


def write_plan(root: Path, cfg: PilotConfig, **extra: Any) -> tuple[list[TrainRun], list[EvalRun]]:
    trains, evals = train_runs(cfg, root), eval_runs(cfg, root)
    plan_file = root / "plan.json"
    old = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan = {
        **{k: v for k, v in old.items() if k in ("created", "git_head", "sizing")},
        **extra,
        "config": asdict(cfg),
        "updated": now,
        "tokens_per_run_at_most": cfg.steps * cfg.grad_accum * cfg.seq_len,
        "train": [asdict(r) for r in trains],
        "eval": [asdict(r) for r in evals],
    }
    plan.setdefault("created", now)
    if "git_head" not in plan:
        plan["git_head"] = git_head()
    root.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    lines = ["#!/bin/sh", "set -e", f"# pilot {cfg.name}: run with python -m tools.pilot run"]
    lines += [shlex.join(own_command(root, k, i)) for k, i in ordered_steps(trains, evals)]
    (root / "commands.sh").write_text("\n".join(lines) + "\n")
    return trains, evals


def first_sample(benchmarks: list[str]) -> dict[str, int]:
    from tools.eval_power import ASSUMED_DISCORDANCE, FULL_SIZES, required_sizes

    available = {b: FULL_SIZES[b] for b in benchmarks}
    return required_sizes(dict.fromkeys(available, ASSUMED_DISCORDANCE), available)


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
        eval_max_new_tokens=args.eval_max_new_tokens or args.seq_len,
        eval_set=args.eval_set,
        eval_sample=first_sample(args.benchmarks) if args.eval_set == "verdict" else None,
        eval_batch_size=args.eval_batch_size,
        eval_extra=shlex.split(args.eval_extra),
        device=args.device,
    )
    plan_file = root / "plan.json"
    if plan_file.exists():
        old = json.loads(plan_file.read_text())["config"]
        if old != asdict(cfg) and not args.replace:
            raise SystemExit(f"{plan_file} exists with a different config; pick a new --name")
    trains, evals = write_plan(root, cfg)
    print(f"plan: {plan_file}")
    print(f"{len(trains)} training runs, {len(evals)} eval runs, nothing started")
    print(f"at most {cfg.steps * cfg.grad_accum * cfg.seq_len:,} tokens per training run")
    if cfg.eval_sample:
        sizes = ", ".join(f"{b} {n}" for b, n in cfg.eval_sample.items())
        print(f"verdict eval set: {sizes} problems; grows if the first arms disagree more")


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
    for r in sorted(trains, key=lambda t: t.arm == "separate"):
        steps.append(("train-one", r.id))
        trained.add(r.id)
        ready = [e for e in waiting if set(e.needs) <= trained]
        steps += [("eval-one", e.id) for e in ready]
        waiting = [e for e in waiting if e not in ready]
    return steps + [("eval-one", e.id) for e in waiting]


def step_argv(trains: list[TrainRun], evals: list[EvalRun]) -> dict[tuple[str, str], list[str]]:
    steps = {("train-one", r.id): r.argv for r in trains}
    return steps | {("eval-one", e.id): e.argv for e in evals}


def step_done(root: Path, kind: str, run_id: str, argv: list[str]) -> bool:
    status = read_status(root, run_id, "train" if kind == "train-one" else "eval")
    return bool(status["state"] == DONE and status.get("argv") == argv)


def pending_steps(
    root: Path, trains: list[TrainRun], evals: list[EvalRun], only: list[str]
) -> list[tuple[str, str]]:
    def wanted(run_id: str) -> bool:
        return not only or any(run_id == o or run_id.startswith(f"{o}-") for o in only)

    argv = step_argv(trains, evals)
    return [
        (k, i)
        for k, i in ordered_steps(trains, evals)
        if wanted(i) and not step_done(root, k, i, argv[(k, i)])
    ]


def scores_of(root: Path, run_id: str, bench: str) -> dict[str, bool]:
    path = root / run_id / "eval" / bench / "scores.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(r["task_id"]): bool(r["passed"]) for r in rows}


def available_sizes(root: Path, cfg: PilotConfig) -> dict[str, int]:
    from tools.eval_power import FULL_SIZES

    sizes = {b: FULL_SIZES[b] for b in cfg.benchmarks}
    for arm in VERDICT_ARMS:
        for seed in cfg.seeds:
            report = root / f"{arm}-s{seed}" / "eval" / "report.json"
            if report.exists():
                benches = json.loads(report.read_text()).get("benchmarks", {})
                for b in cfg.benchmarks:
                    if benches.get(b, {}).get("n_available"):
                        sizes[b] = int(benches[b]["n_available"])
    if cfg.eval_limit:
        sizes = {b: min(n, cfg.eval_limit) for b, n in sizes.items()}
    return sizes


def observed_discordance(root: Path, cfg: PilotConfig) -> dict[str, dict[str, Any]]:
    from tools.eval_power import MIN_SAMPLE, discordance_of

    seen: dict[str, dict[str, Any]] = {}
    for bench in cfg.benchmarks:
        pairs: dict[str, Any] = {}
        for seed in cfg.seeds:
            full = scores_of(root, f"full-s{seed}", bench)
            for other in ("sft", "moe"):
                d, n = discordance_of(full, scores_of(root, f"{other}-s{seed}", bench))
                if n >= MIN_SAMPLE:
                    pairs[f"full-{other}-s{seed}"] = {"discordance": round(d, 4), "shared": n}
        if pairs:
            seen[bench] = {"max": max(p["discordance"] for p in pairs.values()), "pairs": pairs}
    return seen


def sizing(root: Path, cfg: PilotConfig) -> dict[str, Any]:
    from tools.eval_power import ASSUMED_DISCORDANCE, power_at, required_sizes, seed_se

    available = available_sizes(root, cfg)
    seen = observed_discordance(root, cfg)
    disc = {b: seen[b]["max"] if b in seen else ASSUMED_DISCORDANCE for b in available}
    current = dict(cfg.eval_sample or available)
    current = {b: min(current.get(b, available[b]), available[b]) for b in available}
    needed = required_sizes(disc, available)
    grown = {b: max(current[b], needed[b]) for b in available}
    return {
        "discordance": disc,
        "observed": seen,
        "current": current,
        "needed": needed,
        "grown": grown,
        "power_at_current": round(power_at(seed_se(disc, current)), 3),
        "power_at_grown": round(power_at(seed_se(disc, grown)), 3),
        "power_full_suite": round(power_at(seed_se(disc, available)), 3),
    }


def resize(root: Path) -> dict[str, int] | None:
    cfg, _, _ = load_plan(root)
    if cfg.eval_set != "verdict":
        return None
    result = sizing(root, cfg)
    if result["grown"] == result["current"]:
        return None
    plan = json.loads((root / "plan.json").read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    history = [*plan.get("sizing", []), {"at": now, "from": result["current"], **result}]
    cfg.eval_sample = result["grown"]
    write_plan(root, cfg, sizing=history)
    return cast("dict[str, int]", result["grown"])


def cmd_run(args: argparse.Namespace) -> None:
    cfg, trains, evals = load_plan(args.root)
    todo = pending_steps(args.root, trains, evals, args.only)
    if args.dry_run or not todo:
        for kind, run_id in todo:
            print(shlex.join(own_command(args.root, kind, run_id)))
        print(f"{len(todo)} steps pending" + (" (dry run, nothing started)" if todo else ""))
        return
    tried: set[tuple[str, str, str]] = set()
    while True:
        cfg, trains, evals = load_plan(args.root)
        argv = step_argv(trains, evals)
        todo = [
            (k, i)
            for k, i in pending_steps(args.root, trains, evals, args.only)
            if (k, i, json.dumps(argv[(k, i)])) not in tried
        ]
        if not todo:
            return
        kind, run_id = todo[0]
        tried.add((kind, run_id, json.dumps(argv[(kind, run_id)])))
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
        elif kind == "eval-one":
            grown = resize(args.root)
            if grown:
                print(f"verdict eval set grown to {grown}; earlier evals resume", flush=True)


def cmd_size(args: argparse.Namespace) -> None:
    cfg, _, _ = load_plan(args.root)
    result = sizing(args.root, cfg)
    print(json.dumps({"eval_set": cfg.eval_set, **result}, indent=2))
    if args.apply:
        grown = resize(args.root)
        print(f"grown to {grown}" if grown else "no change")


def cmd_widen(args: argparse.Namespace) -> None:
    cfg, _, _ = load_plan(args.root)
    if cfg.eval_set == "full":
        print("already the full suite with every arm")
        return
    plan = json.loads((args.root / "plan.json").read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    history = [*plan.get("sizing", []), {"at": now, "widened_from": cfg.eval_sample}]
    cfg.eval_set, cfg.eval_sample = "full", None
    trains, evals = write_plan(args.root, cfg, sizing=history)
    todo = pending_steps(args.root, trains, evals, [])
    print(f"full suite and the separate arm added; {len(todo)} steps pending")
    print("evals resume: problems already answered are kept, only the rest are generated")


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
    if cfg.eval_sample and cfg.eval_set == "verdict":
        print("verdict eval set: " + ", ".join(f"{b} {n}" for b, n in cfg.eval_sample.items()))
    for e in evals:
        status = read_status(args.root, e.id, "eval")
        state = status["state"]
        if state == DONE and status.get("argv") != e.argv:
            state = "done, resumes for the current plan"
        print(f"{e.id:<22} eval {state}")


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


def score_table(root: Path, evals: list[EvalRun]) -> dict[str, dict[int, dict[str, float]]]:
    table: dict[str, dict[int, dict[str, float]]] = {}
    for e in evals:
        scores = eval_scores(root, e.id)
        if scores is not None:
            table.setdefault(e.arm, {})[e.seed] = scores
    return table


def cmd_compare(args: argparse.Namespace) -> None:
    cfg, _, evals = load_plan(args.root)
    table = score_table(args.root, evals)
    result = {
        "pilot": cfg.name,
        "eval_set": cfg.eval_set,
        "scores_points": {a: {str(s): v for s, v in by.items()} for a, by in table.items()},
        **verdict(table, cfg.seeds),
    }
    if cfg.eval_set == "verdict":
        sized = sizing(args.root, cfg)
        result["verdict_sample"] = {
            "problems": sized["current"],
            "discordance": sized["discordance"],
            "observed": bool(sized["observed"]),
            "power_for_a_2_point_gap": sized["power_at_current"],
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
    plan.add_argument(
        "--eval-set",
        choices=EVAL_SETS,
        default="verdict",
        help="verdict: the sample the kill rule needs, no separate arm; full: everything",
    )
    plan.add_argument(
        "--eval-max-new-tokens", type=int, help="default: --seq-len, the longest row trained on"
    )
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

    size = sub.add_parser("size", help="show the verdict sample the observed scores call for")
    size.add_argument("--root", type=Path, required=True)
    size.add_argument("--apply", action="store_true", help="grow the plan's sample to match")
    size.set_defaults(func=cmd_size)

    widen = sub.add_parser("widen", help="switch to the full suite and add the separate arm")
    widen.add_argument("--root", type=Path, required=True)
    widen.set_defaults(func=cmd_widen)

    for name, func in (("status", cmd_status), ("compare", cmd_compare)):
        view = sub.add_parser(name)
        view.add_argument("--root", type=Path, required=True)
        view.set_defaults(func=func)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
