from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chained_transformer.modes import MODE_NAMES

Runs = list[list[int]]
SWITCH_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, None))


def _bucket(n: int) -> str:
    for lo, hi in SWITCH_BUCKETS:
        if n >= lo and (hi is None or n <= hi):
            return f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}")
    raise ValueError(n)


def _quantiles(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def merge_runs(runs: Runs) -> Runs:
    merged: Runs = []
    for mode, length in runs:
        if length <= 0:
            continue
        if merged and merged[-1][0] == mode:
            merged[-1][1] += length
        else:
            merged.append([mode, length])
    return merged


def runs_from_modes(modes: Sequence[int]) -> Runs:
    return merge_runs([[int(m), 1] for m in modes])


def first_position(runs: Runs, mode: int) -> int | None:
    pos = 0
    for m, length in runs:
        if m == mode:
            return pos
        pos += length
    return None


def summarize(
    traces: Sequence[Runs], passed: Sequence[bool | None] | None = None
) -> dict[str, Any]:
    traces = [merge_runs(r) for r in traces]
    switches = [max(0, len(r) - 1) for r in traces]
    tokens = [sum(length for _, length in r) for r in traces]
    pairs: Counter[str] = Counter()
    for r in traces:
        for (a, _), (b, _) in zip(r, r[1:], strict=False):
            pairs[f"{MODE_NAMES[a]}->{MODE_NAMES[b]}"] += 1
    total_pairs = sum(pairs.values())
    summary: dict[str, Any] = {
        "responses": len(traces),
        "tokens": sum(tokens),
        "switches": _quantiles(switches),
        "switches_per_1k_tokens": round(1000 * sum(switches) / sum(tokens), 3)
        if sum(tokens)
        else None,
        "switch_histogram": dict(Counter(_bucket(s) for s in switches)),
        "transitions": {
            k: {"count": v, "share": round(v / total_pairs, 4)} for k, v in pairs.most_common()
        },
        "run_length_by_mode": {
            name: _quantiles([length for r in traces for m, length in r if m == mode])
            for mode, name in enumerate(MODE_NAMES)
        },
        "token_share_by_mode": {
            name: round(
                sum(length for r in traces for m, length in r if m == mode) / max(1, sum(tokens)),
                4,
            )
            for mode, name in enumerate(MODE_NAMES)
        },
        "reaches_mode": {
            name: round(sum(any(m == mode for m, _ in r) for r in traces) / max(1, len(traces)), 4)
            for mode, name in enumerate(MODE_NAMES)
        },
        "first_position": {
            name: _quantiles([p for r in traces if (p := first_position(r, mode)) is not None])
            for mode, name in enumerate(MODE_NAMES)
        },
    }
    if passed is not None:
        known = [(s, p) for s, p in zip(switches, passed, strict=True) if p is not None]
        by_bucket: dict[str, list[bool]] = {}
        for s, p in known:
            by_bucket.setdefault(_bucket(s), []).append(bool(p))
        summary["by_outcome"] = {
            "mean_switches_passed": _mean([s for s, p in known if p]),
            "mean_switches_failed": _mean([s for s, p in known if not p]),
            "pass_rate_by_switches": {
                k: {"n": len(v), "pass_rate": round(sum(v) / len(v), 4)}
                for k, v in sorted(by_bucket.items())
            },
        }
    return summary


def _mean(values: Sequence[int]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def from_eval_dir(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for responses in sorted(path.glob("*/responses.jsonl")):
        bench = responses.parent
        rows = read_jsonl(responses)
        with_runs = [r for r in rows if r.get("mode_runs")]
        scores_file = bench / "scores.jsonl"
        verdicts = (
            {s["task_id"]: bool(s["passed"]) for s in read_jsonl(scores_file)}
            if scores_file.exists()
            else {}
        )
        passed = [verdicts.get(r["task_id"]) for r in with_runs]
        summary = summarize([r["mode_runs"] for r in with_runs], passed if verdicts else None)
        summary["responses_without_trace"] = len(rows) - len(with_runs)
        out[bench.name] = summary
    return out


def from_dataset(path: Path, limit: int | None) -> dict[str, Any]:
    from datasets import load_from_disk

    data = load_from_disk(str(path))
    n = len(data) if limit is None else min(limit, len(data))
    traces: list[Runs] = []
    for i in range(n):
        row = data[i]
        labels = row["mode_ids"][:-1]
        keep = row["loss_mask"][1:]
        traces.append(runs_from_modes([m for m, k in zip(labels, keep, strict=True) if k]))
    return summarize(traces)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mode switch statistics for evals and data")
    parser.add_argument("--eval", type=Path, nargs="*", default=[], help="evals.run output dirs")
    parser.add_argument("--data", type=Path, nargs="*", default=[], help="Arrow dataset dirs")
    parser.add_argument("--limit", type=int, help="first N dataset rows")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.eval and not args.data:
        parser.error("give --eval and/or --data")
    report: dict[str, Any] = {
        "eval": {str(p): from_eval_dir(p) for p in args.eval},
        "data": {str(p): from_dataset(p, args.limit) for p in args.data},
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
