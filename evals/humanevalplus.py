from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.math500 import run_guarded
from evals.segments import after_thinking, split
from evals.types import Loaded, Problem, Score

NAME = "humanevalplus"
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
PYTHON_TAGS = frozenset({"", "python", "py", "python3", "py3"})
PASS = "pass"
ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "XDG_CACHE_HOME", "HUMANEVAL_OVERRIDE_PATH")


@dataclass(frozen=True)
class Extracted:
    code: str
    block_found: bool
    think_closed: bool


def make_prompt(task_prompt: str) -> str:
    return f"{INSTRUCTION_PREFIX}\n```python\n{task_prompt.strip()}\n```"


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    blocks = []
    ticks_open = 0
    info = ""
    body: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        ticks = len(stripped) - len(stripped.lstrip("`"))
        if ticks_open == 0:
            if ticks >= 3 and "`" not in stripped[ticks:]:
                ticks_open, info, body = ticks, stripped[ticks:].strip(), []
        elif ticks >= ticks_open and stripped == "`" * ticks:
            blocks.append((info, "\n".join(body)))
            ticks_open = 0
        else:
            body.append(line)
    if ticks_open:
        blocks.append((info, "\n".join(body)))
    return blocks


def is_python(info: str) -> bool:
    words = info.split()
    return (words[0].lower() if words else "") in PYTHON_TAGS


def extract_code(response: str) -> Extracted:
    visible = after_thinking(response)
    blocks = [body for info, body in fenced_blocks(visible) if is_python(info)]
    closed = split(response).think_closed
    if blocks:
        return Extracted(blocks[-1], True, closed)
    return Extracted(visible, False, closed)


def sanitize_job(job: tuple[str, str]) -> tuple[str, str]:
    from evalplus.sanitize import sanitize

    code, entry_point = job
    try:
        return sanitize(code, entrypoint=entry_point), "ok"
    except Exception as e:
        return code, f"error: {type(e).__name__}"


def sanitize_all(
    jobs: Sequence[tuple[str, str]], workers: int, timeout_s: float
) -> list[tuple[str, str]]:
    importlib.import_module("evalplus.sanitize")
    done = run_guarded(sanitize_job, jobs, workers, timeout_s)
    return [r if r is not None else (j[0], "timeout") for j, r in zip(jobs, done, strict=True)]


def parse_results(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for task_id, entries in data["eval"].items():
        if not entries:
            continue
        entry = entries[0]
        base, plus = entry.get("base_status"), entry.get("plus_status")
        out[task_id] = {
            "base_status": base,
            "plus_status": plus,
            "base_passed": base == PASS,
            "plus_passed": plus == PASS,
            "passed": base == PASS and plus == PASS,
        }
    return out


def write_samples(path: Path, task_ids: Sequence[str], solutions: Mapping[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for task_id in task_ids:
            f.write(json.dumps({"task_id": task_id, "solution": solutions.get(task_id, "")}) + "\n")


def evaluator_env() -> dict[str, str]:
    return {k: os.environ[k] for k in ENV_KEEP if k in os.environ}


def run_evaluator(samples: Path, workers: int, timeout_s: float) -> Path:
    cwd = samples.parent / "cwd"
    cwd.mkdir()
    log = samples.parent / "evaluate.log"
    cmd = [
        sys.executable,
        "-I",
        "-m",
        "evalplus.evaluate",
        "--dataset",
        "humaneval",
        "--samples",
        str(samples.resolve()),
        "--parallel",
        str(max(1, workers)),
        "--i_just_wanna_run",
    ]
    with log.open("w") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=evaluator_env(),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            code = None
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    tail = log.read_text(errors="replace")[-2000:]
    if code is None:
        raise RuntimeError(f"evalplus evaluator timed out after {timeout_s}s; log {log}\n{tail}")
    if code != 0:
        raise RuntimeError(f"evalplus evaluator exited {code}; log {log}\n{tail}")
    return samples.with_name(samples.stem + "_eval_results.json")


class HumanEvalPlus:
    name: str = NAME
    sanitize_timeout_s: float = 60.0
    eval_timeout_s: float = 3600.0

    def load(self, limit: int | None, source: str | None) -> Loaded:
        if source not in (None, "evalplus", "github"):
            raise ValueError(f"humanevalplus loads only through evalplus, not {source!r}")
        import evalplus
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION, _ready_human_eval_plus_path
        from evalplus.data.utils import get_dataset_metadata

        data = get_human_eval_plus()
        path = Path(_ready_human_eval_plus_path())
        url, _ = get_dataset_metadata("HumanEvalPlus", HUMANEVAL_PLUS_VERSION, False)
        tasks = list(data.values())
        picked = tasks if limit is None else tasks[:limit]
        info = {
            "benchmark": NAME,
            "kind": "evalplus",
            "evalplus_version": evalplus.__version__,
            "dataset": "HumanEvalPlus",
            "dataset_version": HUMANEVAL_PLUS_VERSION,
            "url": url,
            "override_path": os.environ.get("HUMANEVAL_OVERRIDE_PATH"),
            "cache_path": str(path),
            "md5": get_human_eval_plus_hash(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": len(tasks),
            "loaded": len(picked),
        }
        problems = [
            Problem(
                benchmark=NAME,
                task_id=t["task_id"],
                prompt=make_prompt(t["prompt"]),
                meta={
                    "entry_point": t["entry_point"],
                    "task_prompt": t["prompt"],
                    "canonical_solution": t["canonical_solution"],
                },
            )
            for t in picked
        ]
        return Loaded(problems=problems, source=info)

    def score(
        self,
        problems: Sequence[Problem],
        responses: Mapping[str, str],
        workdir: Path,
        workers: int,
    ) -> list[Score]:
        from evalplus.data import get_human_eval_plus

        task_ids = list(get_human_eval_plus())
        unknown = [p.task_id for p in problems if p.task_id not in task_ids]
        if unknown:
            raise ValueError(f"not HumanEval+ tasks: {unknown[:5]}")
        extracted = [extract_code(responses.get(p.task_id, "")) for p in problems]
        jobs = [
            (e.code, str(p.meta["entry_point"])) for p, e in zip(problems, extracted, strict=True)
        ]
        sanitized = sanitize_all(jobs, workers, self.sanitize_timeout_s)
        solutions = {p.task_id: s[0] for p, s in zip(problems, sanitized, strict=True)}
        workdir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"{NAME}-", dir=workdir))
        samples = run_dir / "samples.jsonl"
        write_samples(samples, task_ids, solutions)
        results_path = run_evaluator(samples, workers, self.eval_timeout_s)
        results = parse_results(json.loads(results_path.read_text()))
        scores = []
        for p, e, (solution, status) in zip(problems, extracted, sanitized, strict=True):
            r = results.get(p.task_id, {})
            detail = {
                "base_passed": bool(r.get("base_passed")),
                "plus_passed": bool(r.get("plus_passed")),
                "base_status": r.get("base_status"),
                "plus_status": r.get("plus_status"),
                "block_found": e.block_found,
                "think_closed": e.think_closed,
                "sanitize": status,
                "missing_response": p.task_id not in responses,
                "solution": solution,
            }
            scores.append(Score(task_id=p.task_id, passed=bool(r.get("passed")), detail=detail))
        return scores

    def reference_response(self, problem: Problem) -> str | None:
        code = str(problem.meta["task_prompt"]) + str(problem.meta["canonical_solution"])
        return f"<think>\n</think>\n\n```python\n{code.rstrip()}\n```"


BENCHMARK = HumanEvalPlus()


def stub_response(problem: Problem) -> str:
    code = str(problem.meta["task_prompt"]).rstrip()
    return f"<think>\n</think>\n\n```python\n{code}\n    return None\n```"


def _run(problems: Sequence[Problem], responses: Mapping[str, str], workers: int) -> dict[str, Any]:
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        scores = BENCHMARK.score(problems, responses, Path(workdir), workers)
    n = len(scores)
    counts = {
        key: sum(bool(s.passed if key == "passed" else s.detail[key]) for s in scores)
        for key in ("passed", "base_passed", "plus_passed", "block_found")
    }
    return {
        "n": n,
        **counts,
        "pass_rate": counts["passed"] / n if n else None,
        "base_rate": counts["base_passed"] / n if n else None,
        "wall_s": round(time.perf_counter() - start, 2),
        "scores": [{"task_id": s.task_id, "passed": s.passed, **s.detail} for s in scores],
    }


def oracle_checks(limit: int | None, workers: int) -> dict[str, Any]:
    loaded = BENCHMARK.load(limit, None)
    problems = loaded.problems
    reference = {p.task_id: str(BENCHMARK.reference_response(p)) for p in problems}
    stubs = {p.task_id: stub_response(p) for p in problems}
    report: dict[str, Any] = {"source": loaded.source, "workers": workers}
    report["canonical"] = _run(problems, reference, workers)
    report["return_none_stub"] = _run(problems, stubs, workers)
    for name, expect in (("canonical", True), ("return_none_stub", False)):
        report[name]["unexpected"] = [
            s for s in report[name].pop("scores") if bool(s["passed"]) != expect
        ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanEval+ scorer oracle checks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = oracle_checks(args.limit, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("canonical", "return_none_stub"):
        summary = {k: v for k, v in report[name].items() if k != "unexpected"}
        print(name, summary, "unexpected", len(report[name]["unexpected"]))


if __name__ == "__main__":
    main()
