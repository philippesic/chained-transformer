from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, Qwen3ForCausalLM

from chained_transformer.checkpoint import ChainedQwen3Config, ChainedQwen3ForCausalLM
from chained_transformer.modes import MODE_NAMES
from chained_transformer.surgery import LoRALinear, ModeMLP, MoEMLP
from evals.switch_stats import SWITCH_BUCKETS, from_dataset, read_jsonl, summarize
from tools.pilot import (
    ARMS,
    KILL_MARGIN_POINTS,
    LORA_TIE_POINTS,
    EvalRun,
    PilotConfig,
    load_plan,
    score_table,
    verdict,
)

NA = "n/a"
BENCH_COLUMNS = {
    "math500": [("math500", "MATH500")],
    "humanevalplus": [("he_base", "HE+ base"), ("he_plus", "HE+ plus")],
    "livecodebench": [("lcb", "LCB")],
}
TRANSITIONS = tuple(f"{a}->{b}" for a in MODE_NAMES for b in MODE_NAMES if a != b)
Stats = dict[str, float | None]
TABLE1_NOTE = (
    "Mean over seeds, then each seed in parentheses. Scores are pass rates in points; mean is "
    "the mean accuracy over the benchmarks run (HE+ counts as plus). leak/1k is think words per "
    "1k answer words, pooled. train GiB is peak reserved memory in training (largest of the "
    "three runs for separate); eval GiB is peak eval memory. active counts the parameters one "
    "token passes through."
)


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def numel(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def checkpoint_params(checkpoint: Path) -> dict[str, int]:
    config = AutoConfig.from_pretrained(str(checkpoint))
    with torch.device("meta"):
        model: torch.nn.Module = (
            ChainedQwen3ForCausalLM(config)
            if isinstance(config, ChainedQwen3Config)
            else Qwen3ForCausalLM(config)
        )
    total = numel(model.parameters())
    idle = 0.0
    for module in model.modules():
        if isinstance(module, ModeMLP):
            n = len(module.modes)
            idle += numel(module.modes.parameters()) * (n - 1) / n
        elif isinstance(module, LoRALinear):
            n = len(module.lora_a)
            idle += numel([*module.lora_a, *module.lora_b]) * (n - 1) / n
        elif isinstance(module, MoEMLP):
            n = len(module.experts)
            idle += numel(module.experts.parameters()) * (n - module.top_k) / n
    return {"total": total, "active": total - round(idle)}


def run_params(root: Path, run: EvalRun, cache: dict[str, dict[str, int]]) -> Stats:
    counts = []
    for need in run.needs:
        config = root / need / "checkpoint" / "config.json"
        if not config.exists():
            return {"total_params": None, "active_params": None}
        text = config.read_text()
        if text not in cache:
            cache[text] = checkpoint_params(config.parent)
        counts.append(cache[text])
    return {
        "total_params": float(sum(c["total"] for c in counts)),
        "active_params": float(max(c["active"] for c in counts)),
    }


def weighted(pairs: Sequence[tuple[float | None, float]]) -> float | None:
    known = [(v, w) for v, w in pairs if v is not None and w > 0]
    total = sum(w for _, w in known)
    return sum(v * w for v, w in known) / total if total else None


def eval_metrics(report: dict[str, Any]) -> Stats:
    benches: dict[str, Any] = report.get("benchmarks", {})
    out: Stats = {}
    for name, bench in benches.items():
        scores = bench.get("scores", {})
        points = {
            k: 100.0 * scores[s] if scores.get(s) is not None else None
            for k, s in (
                ("math500", "accuracy"),
                ("lcb", "accuracy"),
                ("he_base", "pass_at_1_base"),
                ("he_plus", "pass_at_1_plus"),
            )
        }
        for key, _ in BENCH_COLUMNS.get(name, []):
            out[key] = points[key]
    accuracies = [
        b["scores"]["accuracy"]
        for b in benches.values()
        if b.get("scores", {}).get("accuracy") is not None
    ]
    out["mean"] = 100.0 * statistics.fmean(accuracies) if accuracies else None
    leak = [b.get("leakage", {}) for b in benches.values()]
    out["leakage"] = weighted(
        [
            (
                lk.get("answer_rate_per_1k_pooled"),
                (lk.get("n_closed") or 0) * (lk.get("answer_words_mean") or 0.0),
            )
            for lk in leak
        ]
    )
    perf = [b.get("perf", {}) for b in benches.values()]
    seconds = sum(p.get("seconds") or 0.0 for p in perf)
    out["tokens_per_s"] = sum(p.get("new_tokens") or 0 for p in perf) / seconds if seconds else None
    peaks = [p.get("peak_alloc_gib", p.get("peak_rss_gib")) for p in perf]
    out["eval_peak_gib"] = max((p for p in peaks if p is not None), default=None)
    batches = [p["batch_size"] for p in perf if p.get("batch_size") is not None]
    out["batch_size"] = float(min(batches)) if batches else None
    modes = [b.get("modes", {}) for b in benches.values()]
    for key, field in (
        ("reprefilled", "mean_reprefilled_tokens"),
        ("new_tokens", "mean_new_tokens"),
        ("switches", "mean_switches"),
    ):
        out[key] = weighted([(m.get(field), float(m.get("n") or 0)) for m in modes])
    return out


def train_peak(root: Path, run: EvalRun) -> float | None:
    peaks = []
    for need in run.needs:
        summary = read_json(root / need / "summary.json") or {}
        peak = summary.get("peak_reserved_gib")
        if peak is not None and summary.get("device") != "cpu":
            peaks.append(float(peak))
    return max(peaks) if peaks else None


def collect(root: Path, evals: list[EvalRun]) -> dict[str, dict[int, Stats]]:
    cache: dict[str, dict[str, int]] = {}
    table: dict[str, dict[int, Stats]] = {}
    for run in evals:
        report = read_json(root / run.id / "eval" / "report.json")
        stats: Stats = eval_metrics(report) if report else {}
        stats["train_peak_gib"] = train_peak(root, run)
        stats.update(run_params(root, run, cache))
        table.setdefault(run.arm, {})[run.seed] = stats
    return table


def traces(root: Path, run_ids: Sequence[str]) -> tuple[list[list[list[int]]], list[bool | None]]:
    runs: list[list[list[int]]] = []
    passed: list[bool | None] = []
    for run_id in run_ids:
        for responses in sorted((root / run_id / "eval").glob("*/responses.jsonl")):
            scores = responses.parent / "scores.jsonl"
            verdicts = (
                {s["task_id"]: bool(s["passed"]) for s in read_jsonl(scores)}
                if scores.exists()
                else {}
            )
            for row in read_jsonl(responses):
                if row.get("mode_runs"):
                    runs.append(row["mode_runs"])
                    passed.append(verdicts.get(row["task_id"]))
    return runs, passed


def switching(
    root: Path, cfg: PilotConfig, evals: list[EvalRun], data_limit: int | None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    data = Path(cfg.data)
    if data_limit != 0 and (data / "dataset_info.json").exists():
        out["data"] = from_dataset(data, data_limit)
    for arm in ARMS:
        runs, passed = traces(root, [e.id for e in evals if e.arm == arm])
        if runs:
            known = passed if any(p is not None for p in passed) else None
            out[arm] = summarize(runs, known)
    return out


def fmt(value: float | None, digits: int) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def seeded(by_seed: dict[int, Stats], seeds: list[int], key: str, digits: int) -> str:
    values = [by_seed.get(s, {}).get(key) for s in seeds]
    known = [v for v in values if v is not None]
    if not known:
        return NA
    mean = fmt(statistics.fmean(known), digits)
    if len(seeds) == 1:
        return mean
    return f"{mean} ({' / '.join(fmt(v, digits) for v in values)})"


def first(by_seed: dict[int, Stats], key: str) -> float | None:
    return next((s[key] for s in by_seed.values() if s.get(key) is not None), None)


def millions(value: float | None) -> str:
    return NA if value is None else f"{value / 1e6:.1f}M"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def table_main(cfg: PilotConfig, stats: dict[str, dict[int, Stats]]) -> list[str]:
    bench_cols = [c for b in cfg.benchmarks for c in BENCH_COLUMNS.get(b, [])]
    cols = [
        *[(k, h, 1) for k, h in bench_cols],
        ("mean", "mean", 1),
        ("leakage", "leak/1k", 2),
        ("tokens_per_s", "tok/s", 0),
        ("train_peak_gib", "train GiB", 2),
        ("eval_peak_gib", "eval GiB", 2),
    ]
    headers = ["arm", *(h for _, h, _ in cols), "params", "active"]
    rows = []
    for arm in ARMS:
        by_seed = stats.get(arm, {})
        cells = [seeded(by_seed, cfg.seeds, k, d) for k, _, d in cols]
        rows.append(
            [
                arm,
                *cells,
                millions(first(by_seed, "total_params")),
                millions(first(by_seed, "active_params")),
            ]
        )
    return markdown_table(headers, rows)


def table_kill(cfg: PilotConfig, result: dict[str, Any]) -> list[str]:
    rows = []
    for seed in cfg.seeds:
        m = result["margins"].get(str(seed), {})
        rows.append(
            [
                str(seed),
                *(
                    fmt(m.get(k), 2)
                    for k in ("full_minus_sft", "full_minus_moe", "lora_minus_full")
                ),
            ]
        )
    return markdown_table(["seed", "full - sft", "full - moe", "lora - full"], rows)


def table_switching(cfg: PilotConfig, sw: dict[str, Any]) -> list[str]:
    headers = [
        "source",
        "responses",
        "switches/resp",
        "switches/1k tok",
        "code run",
        "answer run",
        "reach code",
        "reach answer",
    ]
    rows = []
    for name, s in sw.items():
        runs = s["run_length_by_mode"]
        rows.append(
            [
                name,
                str(s["responses"]),
                fmt(s["switches"]["mean"], 2),
                fmt(s["switches_per_1k_tokens"], 2),
                fmt(runs["code"]["mean"], 1),
                fmt(runs["answer"]["mean"], 1),
                fmt(s["reaches_mode"]["code"], 3),
                fmt(s["reaches_mode"]["answer"], 3),
            ]
        )
    shares = [
        [name, *(fmt(s["transitions"].get(t, {}).get("share", 0.0), 3) for t in TRANSITIONS)]
        for name, s in sw.items()
    ]
    return [
        *markdown_table(headers, rows),
        "",
        "Transition shares:",
        "",
        *markdown_table(["source", *TRANSITIONS], shares),
    ]


def table_pass_by_switches(sw: dict[str, Any]) -> list[str]:
    buckets = [
        f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}")
        for lo, hi in SWITCH_BUCKETS
    ]
    rows = []
    for name, s in sw.items():
        by = s.get("by_outcome", {}).get("pass_rate_by_switches")
        if by is None:
            continue
        cells = [
            f"{100 * by[b]['pass_rate']:.1f} ({by[b]['n']})" if b in by else NA for b in buckets
        ]
        rows.append([name, *cells])
    return markdown_table(["arm", *buckets], rows) if rows else ["No scored traces yet."]


def table_serving(cfg: PilotConfig, stats: dict[str, dict[int, Stats]]) -> list[str]:
    rows = [
        [
            arm,
            seeded(stats.get(arm, {}), cfg.seeds, "reprefilled", 1),
            seeded(stats.get(arm, {}), cfg.seeds, "new_tokens", 0),
            seeded(stats.get(arm, {}), cfg.seeds, "tokens_per_s", 0),
            fmt(first(stats.get(arm, {}), "batch_size"), 0),
        ]
        for arm in ARMS
        if arm != "sft"
    ]
    return markdown_table(
        ["arm", "re-prefilled tok/resp", "new tok/resp", "tok/s", "eval batch"], rows
    )


def notes(root: Path, evals: list[EvalRun], stats: dict[str, dict[int, Stats]]) -> list[str]:
    out = []
    missing = [e.id for e in evals if not (root / e.id / "eval" / "report.json").exists()]
    if missing:
        out.append(f"No eval report yet for: {', '.join(missing)}.")
    reports = [read_json(root / e.id / "eval" / "report.json") for e in evals]
    present = [r for r in reports if r]
    if any(r.get("smoke") for r in present):
        out.append("Some reports used a problem limit, so scores are not full-benchmark scores.")
    devices = {r.get("environment", {}).get("device") for r in present}
    if "cpu" in devices:
        out.append("Some evals ran on CPU, so tok/s and memory are not GPU numbers.")
    sep = first(stats.get("separate", {}), "batch_size")
    others = {first(stats.get(a, {}), "batch_size") for a in ("full", "lora", "moe")} - {None}
    if sep is not None and any(b != sep for b in others):
        out.append(
            "separate decodes at batch 1 and the mode arms were batched, so tok/s in Table 5 "
            "compares batch sizes too; rerun the mode arms with --batch-size 1 for a fair "
            "serving row."
        )
    return out


def build(root: Path, data_limit: int | None) -> tuple[dict[str, Any], str]:
    cfg, trains, evals = load_plan(root)
    result = verdict(score_table(root, evals), cfg.seeds)
    stats = collect(root, evals)
    sw = switching(root, cfg, evals, data_limit)
    remarks = notes(root, evals, stats)
    lines = [
        f"# Pilot results: {cfg.name}",
        "",
        f"Verdict: {result['kill_rule']}.",
    ]
    if result["lora"]:
        lines.append(f"LoRA: {result['lora']}.")
    lines += [
        "",
        f"The rule: full must beat sft and moe by at least {KILL_MARGIN_POINTS} points of mean "
        f"accuracy on every seed; lora within {LORA_TIE_POINTS} point of full means LoRA modes.",
    ]
    if remarks:
        lines += ["", *(f"- {r}" for r in remarks)]
    lines += [
        "",
        "## Table 1, main results",
        "",
        TABLE1_NOTE,
        "",
        *table_main(cfg, stats),
        "",
        "## Table 2, kill rule margins (points)",
        "",
        *table_kill(cfg, result),
        "",
        "## Table 4, switching",
        "",
        "Every benchmark and seed pooled per arm; run lengths are mean tokens.",
        "",
        *(table_switching(cfg, sw) if sw else ["No traces yet."]),
        "",
        "## Figure 3 data, pass rate by switches",
        "",
        "Pass rate in points, with the number of responses in parentheses.",
        "",
        *table_pass_by_switches(sw),
        "",
        "## Table 5, serving",
        "",
        *table_serving(cfg, stats),
        "",
    ]
    data = {
        "pilot": cfg.name,
        "seeds": cfg.seeds,
        "benchmarks": cfg.benchmarks,
        "train_runs": [t.id for t in trains],
        **result,
        "per_run": {a: {str(s): v for s, v in by.items()} for a, by in stats.items()},
        "switching": sw,
        "notes": remarks,
    }
    return data, "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pilot tables and kill-rule verdict")
    parser.add_argument("--root", type=Path, required=True, help="artifacts/pilot/NAME")
    parser.add_argument(
        "--data-limit", type=int, default=5000, help="rows of the training data to profile; 0 skips"
    )
    args = parser.parse_args(argv)
    data, text = build(args.root, args.data_limit)
    (args.root / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.root / "results.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
