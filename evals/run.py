from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from chained_transformer.attention import ATTENTION_CHOICES
from chained_transformer.modes import MODE_NAMES, THINK_CLOSE, THINK_OPEN
from chained_transformer.train import configure_allocator
from evals import humanevalplus, livecodebench, math500
from evals.generation import Generation, GenSettings, Runner, generate_all, load_runner
from evals.leakage import aggregate
from evals.segments import split
from evals.types import Benchmark, Loaded, Problem, Score

BENCHMARKS: dict[str, Benchmark] = {
    "math500": math500.BENCHMARK,
    "humanevalplus": humanevalplus.BENCHMARK,
    "livecodebench": livecodebench.BENCHMARK,
}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
REPORT_VERSION = 1
SAMPLE_SALT = "chained-transformer-eval-sample-v1"
RESUME_IGNORES = ("batch_size", "refill")
EMPTY_THINKING = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
SUMMED_PERF = ("seconds", "new_tokens", "generate_calls", "attention_math_fallbacks")
FINGERPRINT_READS = 16


def git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent

    def run(*cmd: str) -> str:
        res = subprocess.run(["git", *cmd], cwd=root, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def environment(device: str) -> dict[str, Any]:
    import transformers

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return env


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def as_turn(text: str, thinking: bool) -> str:
    return text if thinking or THINK_CLOSE in text else EMPTY_THINKING + text


def weights_fingerprint(paths: Sequence[str]) -> str | None:
    digest = hashlib.sha256()
    for path in paths:
        root = Path(path)
        files = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
        if not files:
            return None
        for f in files:
            size = f.stat().st_size
            digest.update(f"{f.name}:{size}".encode())
            with f.open("rb") as fh:
                for i in range(FINGERPRINT_READS):
                    fh.seek(size * i // FINGERPRINT_READS)
                    digest.update(fh.read(1 << 16))
    return digest.hexdigest()[:16]


def merge_perf(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    old = {k: v for k, v in (old or {}).items() if k not in ("resumed", "scoring_seconds")}
    if not old.get("new_tokens"):
        return new
    if not new.get("new_tokens"):
        return old
    bulk = old if old["new_tokens"] >= new["new_tokens"] else new
    out = {**new, "batch_size": bulk.get("batch_size"), "refill": bulk.get("refill")}
    for key in SUMMED_PERF:
        out[key] = (old.get(key) or 0) + (new.get(key) or 0)
    out["seconds"] = round(out["seconds"], 2)
    out["tokens_per_s"] = round(out["new_tokens"] / out["seconds"], 2) if out["seconds"] else 0.0
    for key in ("peak_alloc_gib", "peak_rss_gib"):
        peaks = [p[key] for p in (old, new) if p.get(key) is not None]
        if peaks:
            out[key] = max(peaks)
    return out


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_complete_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sample_key(name: str, task_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SALT}:{name}:{task_id}".encode()).hexdigest()


def sample_problems(name: str, problems: Sequence[Problem], n: int) -> list[Problem]:
    ranked = sorted(problems, key=lambda p: sample_key(name, p.task_id))
    keep = {p.task_id for p in ranked[:n]}
    return [p for p in problems if p.task_id in keep]


def parse_sample(items: Sequence[str], names: Sequence[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in items:
        name, sep, value = item.rpartition("=")
        if sep and name not in BENCHMARKS:
            raise SystemExit(
                f"--sample takes N or benchmark=N with a benchmark in {list(BENCHMARKS)}"
            )
        try:
            n = int(value)
        except ValueError:
            raise SystemExit(f"--sample takes N or benchmark=N, not {item!r}") from None
        if n < 1:
            raise SystemExit("--sample sizes must be at least 1")
        for target in [name] if sep else names:
            sizes[target] = n
    return sizes


def resume_mismatch(
    previous: dict[str, Any], model: Any, generation: dict[str, Any], weights: str | None = None
) -> list[str]:
    def comparable(settings: dict[str, Any] | None) -> dict[str, Any]:
        filled = {**asdict(GenSettings()), **(settings or {})}
        return {k: v for k, v in filled.items() if k not in RESUME_IGNORES}

    problems = []
    if previous.get("model") != model:
        problems.append("model")
    elif previous.get("weights") and weights and previous["weights"] != weights:
        problems.append("checkpoint (retrained since)")
    if comparable(previous.get("generation")) != comparable(generation):
        problems.append("generation settings")
    return problems


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def mode_stats(generations: Sequence[Generation], thinking: bool = True) -> dict[str, Any]:
    n = len(generations)
    if n == 0:
        return {"n": 0}
    closed = [split(as_turn(g.text, thinking)).think_closed for g in generations]
    return {
        "n": n,
        "mean_new_tokens": _mean([g.new_tokens for g in generations]),
        "mean_tokens_by_mode": {
            m: _mean([g.tokens_by_mode.get(m, 0) for g in generations]) for m in MODE_NAMES
        },
        "token_share_by_mode": {
            m: round(
                sum(g.tokens_by_mode.get(m, 0) for g in generations)
                / max(1, sum(g.new_tokens for g in generations)),
                4,
            )
            for m in MODE_NAMES
        },
        "mean_switches": _mean([g.switches for g in generations]),
        "finished_rate": round(sum(g.finished for g in generations) / n, 4),
        "loop_stop_rate": round(sum(g.stopped == "loop" for g in generations) / n, 4),
        "think_closed_rate": round(sum(closed) / n, 4),
        "mean_reprefilled_tokens": _mean([g.reprefilled_tokens for g in generations]),
    }


def score_summary(name: str, scores: Sequence[Score]) -> dict[str, Any]:
    n = len(scores)
    summary: dict[str, Any] = {
        "n": n,
        "accuracy": round(sum(s.passed for s in scores) / n, 4) if n else None,
    }
    if name == "math500":
        lenient = [bool(s.detail.get("lenient_passed", s.passed)) for s in scores]
        summary["accuracy_lenient"] = round(sum(lenient) / n, 4) if n else None
    if name == "humanevalplus":
        base = [bool(s.detail.get("base_passed")) for s in scores]
        summary["pass_at_1_base"] = round(sum(base) / n, 4) if n else None
        summary["pass_at_1_plus"] = summary["accuracy"]
    if name == "livecodebench":
        summary["pass_at_1"] = summary["accuracy"]
    return summary


def generation_from_row(row: dict[str, Any]) -> Generation:
    return Generation(**row)


def run_benchmark(
    name: str,
    loaded: Loaded,
    runner: Runner | None,
    settings: GenSettings,
    out: Path,
    workers: int,
    score_only: bool,
    oracle: bool,
    resume: bool = False,
    thinking: bool = True,
    previous_perf: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bench = BENCHMARKS[name]
    out.mkdir(parents=True, exist_ok=True)
    responses_path = out / "responses.jsonl"
    perf: dict[str, Any] = {}
    if score_only:
        if not responses_path.exists():
            raise SystemExit(f"--score-only: {responses_path} does not exist")
        generations = [generation_from_row(r) for r in read_complete_jsonl(responses_path)]
        wanted = {p.task_id for p in loaded.problems}
        generations = [g for g in generations if g.task_id in wanted]
        perf = merge_perf(previous_perf, {})
    elif oracle:
        generations = []
        for p in loaded.problems:
            text = bench.reference_response(p)
            if text is not None:
                generations.append(Generation(p.task_id, text, 0, True, {}, 0))
        write_jsonl(responses_path, [asdict(g) for g in generations])
    else:
        assert runner is not None
        kept: list[Generation] = []
        if resume and responses_path.exists():
            rows = read_complete_jsonl(responses_path)
            write_jsonl(responses_path, rows)
            kept = [generation_from_row(r) for r in rows]
        have = {g.task_id for g in kept}
        todo = [p for p in loaded.problems if p.task_id not in have]
        sink = responses_path.open("a" if kept else "w")

        def save(g: Generation) -> None:
            sink.write(json.dumps(asdict(g)) + "\n")
            sink.flush()
            have.add(g.task_id)
            print(
                f"answered {name} {len(have)}/{len(loaded.problems)} "
                f"tokens={g.new_tokens} finished={g.finished}",
                flush=True,
            )

        fresh, perf = generate_all(runner, todo, settings, save)
        sink.close()
        if kept:
            perf = merge_perf(previous_perf, perf)
        wanted = {p.task_id for p in loaded.problems}
        generations = [g for g in kept if g.task_id in wanted] + fresh
        perf["resumed"] = len(generations) - len(fresh)
    texts = {g.task_id: as_turn(g.text, thinking) for g in generations}
    problems: list[Problem] = [p for p in loaded.problems if p.task_id in texts]
    start = time.perf_counter()
    scores = bench.score(problems, texts, out / "work", workers)
    scoring_s = round(time.perf_counter() - start, 2)
    by_id = {g.task_id: g for g in generations}
    write_jsonl(
        out / "scores.jsonl",
        [
            {
                "task_id": s.task_id,
                "passed": s.passed,
                "new_tokens": by_id[s.task_id].new_tokens,
                "tokens_by_mode": by_id[s.task_id].tokens_by_mode,
                "switches": by_id[s.task_id].switches,
                **s.detail,
            }
            for s in scores
        ],
    )
    return {
        "source": loaded.source,
        "n_problems": len(loaded.problems),
        "n_scored": len(scores),
        "scores": score_summary(name, scores),
        "leakage": aggregate(list(texts.values())),
        "modes": mode_stats(generations, thinking),
        "perf": {**perf, "scoring_seconds": scoring_s},
    }


def parse_sources(items: Sequence[str]) -> dict[str, str]:
    sources = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in BENCHMARKS:
            raise SystemExit(
                f"--source takes benchmark=value with a benchmark in {list(BENCHMARKS)}"
            )
        sources[name] = value
    return sources


def batch_size_arg(value: str) -> int:
    if value == "auto":
        return 0
    size = int(value)
    if size < 1:
        raise argparse.ArgumentTypeError("batch size must be at least 1, or auto")
    return size


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate with a checkpoint and score benchmarks")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="mode checkpoint, plain checkpoint or Hub name")
    who.add_argument("--separate", nargs=3, metavar=("THINK", "CODE", "ANSWER"))
    who.add_argument("--oracle", action="store_true", help="score reference responses (no model)")
    who.add_argument(
        "--score-only", action="store_true", help="rescore OUT/<bench>/responses.jsonl"
    )
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--limit", type=int, help="first N problems of each benchmark (smoke)")
    parser.add_argument(
        "--sample",
        nargs="+",
        default=[],
        help="N or benchmark=N: a fixed pseudo-random subset, the same for every model",
    )
    parser.add_argument(
        "--resume", action="store_true", help="keep OUT's responses and generate only the rest"
    )
    parser.add_argument("--no-refill", action="store_true", help="plain static batches")
    parser.add_argument(
        "--decode",
        choices=["hf", "graph"],
        default="hf",
        help="graph: greedy decode steps replayed as CUDA graphs over a preallocated cache",
    )
    parser.add_argument(
        "--stop-loops",
        action="store_true",
        help="end a response once its last 768 tokens repeat exactly with a period of at most 256",
    )
    parser.add_argument(
        "--attention",
        choices=ATTENTION_CHOICES,
        default="auto",
        help="auto: fused kernels, in fp32 for bf16 on GPUs older than sm80; lean: the same, "
        "but one-token decode steps read the cache without copying it",
    )
    parser.add_argument("--source", nargs="*", default=[], help="benchmark=source, e.g. math500=hf")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=batch_size_arg, default=1, help="N, or auto to fill the GPU"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=list(DTYPES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stop_loops and (args.no_refill or args.separate):
        parser.error("--stop-loops needs the refill decoder: not with --no-refill or --separate")
    if args.device.startswith("cuda"):
        configure_allocator()
    dtype = (
        DTYPES[args.dtype]
        if args.dtype
        else (torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    )
    settings = GenSettings(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        seed=args.seed,
        enable_thinking=not args.no_thinking,
        refill=not args.no_refill,
        stop_loops=args.stop_loops,
        decode=args.decode,
    )
    sources = parse_sources(args.source)
    sample = parse_sample(args.sample, args.benchmarks)
    runner = None
    weights = None
    if args.model or args.separate:
        runner = load_runner(
            args.model, args.separate, args.tokenizer, dtype, args.device, args.attention
        )
        weights = weights_fingerprint(args.separate or [args.model])
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    keep_previous = args.score_only or args.resume
    previous = json.loads(report_path.read_text()) if keep_previous and report_path.exists() else {}
    if args.resume and runner is not None:
        if previous:
            differs = resume_mismatch(previous, runner.spec, asdict(settings), weights)
        else:
            held = [n for n in args.benchmarks if (args.out / n / "responses.jsonl").exists()]
            differs = ["run that left no report.json to check"] if held else []
        if differs:
            raise SystemExit(
                f"--resume: {args.out} holds responses from a different {' and '.join(differs)}; "
                "use a new --out"
            )
    if args.score_only and not sample and previous.get("sample"):
        sample = {b: n for b, n in previous["sample"]["sizes"].items() if b in args.benchmarks}
    generation = (
        asdict(settings) if not (args.oracle or args.score_only) else previous.get("generation")
    )
    thinking = bool((generation or {}).get("enable_thinking", settings.enable_thinking))
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "kind": "oracle" if args.oracle else "model",
        "smoke": args.limit is not None,
        "sample": {"sizes": sample, "salt": SAMPLE_SALT} if sample else None,
        "model": runner.spec if runner else previous.get("model"),
        "weights": weights if runner else previous.get("weights"),
        "generation": generation,
        "git": git_state(),
        "environment": environment(args.device),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": dict(previous.get("benchmarks", {})),
    }
    write_json(report_path, report)
    for name in args.benchmarks:
        loaded = BENCHMARKS[name].load(args.limit, sources.get(name))
        available = len(loaded.problems)
        if name in sample:
            loaded = Loaded(sample_problems(name, loaded.problems, sample[name]), loaded.source)
        result = run_benchmark(
            name,
            loaded,
            runner,
            settings,
            args.out / name,
            args.workers,
            args.score_only,
            args.oracle,
            args.resume,
            thinking,
            report["benchmarks"].get(name, {}).get("perf"),
        )
        result["n_available"] = available
        report["benchmarks"][name] = result
        write_json(report_path, report)
        print(name, json.dumps(report["benchmarks"][name]["scores"]), flush=True)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(report_path, report)


if __name__ == "__main__":
    main()
