from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

BASE = "Qwen/Qwen3-0.6B"
VARIANTS = ("full", "lora", "moe")
BENCHMARKS = ("math500", "humanevalplus", "livecodebench")


def step(name: str, cmd: list[str], log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (log_dir / f"{name}.log").open("w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    result = {"step": name, "returncode": code, "seconds": round(time.perf_counter() - start, 1)}
    print(json.dumps(result), flush=True)
    return result


def responses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["task_id"]: r["text"] for r in rows}


def summarize(out: Path, variants: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    conv = out / "conversion.json"
    if conv.exists():
        data = json.loads(conv.read_text())
        summary["conversion"] = {
            v: {
                "prefill_bitwise": {k: d["bitwise"] for k, d in r["prefill"].items()},
                "prefill_max_abs_diff": max(d["max_abs_diff"] for d in r["prefill"].values()),
                "decode_max_abs_diff": r["decode_switching"]["max_abs_diff"],
                "round_trip_bitwise": r.get("round_trip", {}).get("all_bitwise"),
            }
            for v, r in data["variants"].items()
        }
    oracle = out / "oracle" / "report.json"
    if oracle.exists():
        rep = json.loads(oracle.read_text())
        summary["oracle"] = {b: r["scores"] for b, r in rep["benchmarks"].items()}
    runs = {"base": out / "base", **{v: out / v for v in variants}}
    for bench in BENCHMARKS:
        base_text = responses(out / "base" / bench / "responses.jsonl")
        row: dict[str, Any] = {}
        for name, path in runs.items():
            report = path / "report.json"
            if not report.exists():
                continue
            rep = json.loads(report.read_text()).get("benchmarks", {}).get(bench)
            if rep is None:
                continue
            texts = responses(path / bench / "responses.jsonl")
            same = sum(1 for k, t in texts.items() if base_text.get(k) == t)
            row[name] = {
                "scores": rep["scores"],
                "identical_to_base": f"{same}/{len(texts)}",
                "mean_tokens_by_mode": rep["modes"].get("mean_tokens_by_mode"),
                "mean_switches": rep["modes"].get("mean_switches"),
                "think_closed_rate": rep["modes"].get("think_closed_rate"),
                "answer_think_words_per_1k": rep["leakage"].get("answer_rate_per_1k_pooled"),
                "tokens_per_s": rep["perf"].get("tokens_per_s"),
                "source_counts": {
                    k: rep["source"].get(k)
                    for k in (
                        "count",
                        "problems_in_release",
                        "release_size_in_lcb_readme",
                        "problems_in_window",
                    )
                    if k in rep["source"]
                },
            }
        summary[bench] = row
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-model checks: conversion, scorer oracles, base and fresh-mode smoke evals"
    )
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--skip-conversion-check", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source", nargs="*", default=[], help="passed to evals.run --source")
    parser.add_argument("--out", type=Path, default=Path("artifacts/real-model-smoke"))
    args = parser.parse_args()
    py = [sys.executable, "-m"]
    dtype = "bfloat16" if args.device.startswith("cuda") else "float32"
    logs = args.out / "logs"
    steps = []
    if not args.skip_conversion_check:
        steps.append(
            step(
                "conversion_check",
                [*py, "tools.conversion_check", "--base", args.base, "--round-trip"]
                + ["--out", str(args.out / "conversion.json")],
                logs,
            )
        )
    for variant in args.variants:
        steps.append(
            step(
                f"convert_{variant}",
                [*py, "chained_transformer.checkpoint", "convert", "--base", args.base]
                + [
                    "--variant",
                    variant,
                    "--dtype",
                    dtype,
                    "--out",
                    str(args.out / "ckpt" / variant),
                ],
                logs,
            )
        )
    if not args.skip_oracle:
        oracle = [b for b in args.benchmarks if b != "livecodebench"]
        if oracle:
            steps.append(
                step(
                    "oracle",
                    [*py, "evals.run", "--oracle", "--benchmarks", *oracle]
                    + (["--source", *args.source] if args.source else [])
                    + ["--workers", str(args.workers), "--out", str(args.out / "oracle")],
                    logs,
                )
            )
    common = ["--benchmarks", *args.benchmarks, "--limit", str(args.limit), "--greedy"]
    common += ["--max-new-tokens", str(args.max_new_tokens), "--batch-size", str(args.limit)]
    common += ["--device", args.device, "--dtype", dtype, "--workers", str(args.workers)]
    if args.source:
        common += ["--source", *args.source]
    steps.append(
        step(
            "eval_base",
            [*py, "evals.run", "--model", args.base, *common, "--out", str(args.out / "base")],
            logs,
        )
    )
    for variant in args.variants:
        model = str(args.out / "ckpt" / variant)
        steps.append(
            step(
                f"eval_{variant}",
                [*py, "evals.run", "--model", model, *common, "--out", str(args.out / variant)],
                logs,
            )
        )
    summary = {"steps": steps, **summarize(args.out, args.variants)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
