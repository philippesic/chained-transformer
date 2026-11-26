from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import serving_gpu_checklist as checklist

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="{failures}" skipped="{skipped}" tests="3">
<testcase classname="t" name="test_a"/>
<testcase classname="t" name="test_b">{b}</testcase>
<testcase classname="t" name="test_c">{c}</testcase>
</testsuite></testsuites>
"""


def junit(tmp_path: Path, b: str = "", c: str = "") -> Path:
    path = tmp_path / "r.xml"
    failures = int("<failure" in b) + int("<failure" in c)
    skipped = int("<skipped" in b) + int("<skipped" in c)
    path.write_text(JUNIT.format(failures=failures, skipped=skipped, b=b, c=c))
    return path


def args(**changes: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "vllm_python": sys.executable,
        "model": "tiny",
        "base": "Qwen/Qwen3-0.6B",
        "num_prompts": 8,
        "max_tokens": 16,
        "gpu_mem": 0.5,
        "only": [],
        "skip": [],
    }
    base.update(changes)
    return argparse.Namespace(**base)


def test_pytest_judge_passes_clean_runs_and_lists_skips(tmp_path: Path) -> None:
    clean = checklist.pytest_judge(junit(tmp_path), allow_skips=False)(0)
    assert clean.status == "pass"
    assert clean.data["tests"] == 3
    skip = '<skipped message="needs a CUDA GPU"/>'
    allowed = checklist.pytest_judge(junit(tmp_path, b=skip), allow_skips=True)(0)
    assert allowed.status == "pass"
    assert allowed.notes == ["skipped test_b: needs a CUDA GPU"]
    strict = checklist.pytest_judge(junit(tmp_path, b=skip), allow_skips=False)(0)
    assert strict.status == "fail"


def test_pytest_judge_fails_on_failures_missing_reports_and_all_skipped(tmp_path: Path) -> None:
    failed = checklist.pytest_judge(junit(tmp_path, c='<failure message="x"/>'), True)(1)
    assert failed.status == "fail"
    assert "failed: test_c" in failed.notes
    assert checklist.pytest_judge(tmp_path / "none.xml", True)(0).status == "fail"
    path = tmp_path / "all.xml"
    path.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0">'
        '<testcase name="t"><skipped message="no GPU"/></testcase></testsuite></testsuites>'
    )
    every = checklist.pytest_judge(path, allow_skips=True)(0)
    assert every.status == "fail"
    assert "nothing was checked" in " ".join(every.notes)


def kernel_report(err: float = 1e-4, graph: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = [
        {"tokens": 64, "mix": "three", "impl": "loop", "us": 90.0, "max_abs_err": 0.0},
        {"tokens": 64, "mix": "three", "impl": "triton_16x64x64", "us": 30.0, "max_abs_err": err},
        {"tokens": 64, "mix": "three", "impl": "dense_one_mode", "us": 20.0, "max_abs_err": 9.0},
        {"tokens": 64, "mix": "three", "impl": "vllm_fused_experts_top1", "error": "Bad()"},
    ]
    return {
        "gpu": "test",
        "rows": rows,
        "cuda_graph": graph or {"captured": True, "max_abs_err_after_replays": 1e-4},
    }


def test_kernel_outcome_checks_errors_and_graph_capture() -> None:
    good = checklist.kernel_outcome(kernel_report())
    assert good.status == "pass"
    assert good.data["three_modes"]["64"] == {
        "triton": "triton_16x64x64",
        "us": 30.0,
        "loop_us": 90.0,
    }
    assert any(n.startswith("info:") for n in good.notes)
    assert checklist.kernel_outcome(kernel_report(err=0.5)).status == "fail"
    lost = checklist.kernel_outcome(kernel_report(graph={"captured": False, "error": "sync"}))
    assert lost.status == "fail"
    assert "CUDA graph capture failed: sync" in lost.notes


def test_bench_vllm_outcome_flags_failed_arms_and_drift() -> None:
    arms = [
        {"arm": "base", "impl": None, "tok_per_s": 100.0},
        {"arm": "chained", "impl": "select", "tok_per_s": 90.0, "same_tokens_as_base": 7},
        {"arm": "separate", "impl": None, "error": "exit 1"},
    ]
    out = checklist.bench_vllm_outcome({"arms": arms[:2]}, prompts=8)
    assert out.status == "warn"
    assert out.data["tok_per_s"] == {"base": 100.0, "chained/select": 90.0}
    assert checklist.bench_vllm_outcome({"arms": arms}, prompts=8).status == "fail"


def test_plan_filters_steps_and_marks_missing_environments(tmp_path: Path) -> None:
    info = {"vllm_env": {"error": "missing"}}
    steps = checklist.plan(args(only=["check", "vllm-engine"]), tmp_path, info)
    assert [s.name for s in steps] == ["check", "vllm-engine"]
    assert steps[0].needs is None
    assert steps[1].needs is not None and "missing" in steps[1].needs
    skipped = checklist.plan(args(skip=["vllm-bench"]), tmp_path, {"vllm_env": {"cuda": True}})
    assert "vllm-bench" not in [s.name for s in skipped]
    assert all(s.needs is None for s in skipped)
    with pytest.raises(SystemExit, match="unknown step"):
        checklist.plan(args(only=["nope"]), tmp_path, info)


def test_run_step_logs_and_reports(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    write = f"import json; json.dump({{'arms': []}}, open({str(out)!r}, 'w'))"
    ok = checklist.Step(
        "ok",
        [[sys.executable, "-c", write]],
        checklist.json_judge(out, checklist.serving_outcome),
        "hint",
    )
    row = checklist.run_step(ok, tmp_path, timeout=60)
    assert row["status"] == "pass"
    assert "hint" not in row
    bad = checklist.Step(
        "bad",
        [[sys.executable, "-c", "print('boom'); raise SystemExit(3)"]],
        checklist.exit_judge,
        "why",
    )
    row = checklist.run_step(bad, tmp_path, timeout=60)
    assert row["status"] == "fail"
    assert row["notes"] == ["exit 3"]
    assert row["hint"] == "why"
    assert "boom" in row["log_tail"]
    skipped = checklist.Step("later", [], checklist.exit_judge, "", needs="no vLLM")
    assert checklist.run_step(skipped, tmp_path, timeout=60)["status"] == "not run"
    rows = [row, {"name": "later", "status": "not run", "notes": ["no vLLM"], "seconds": 0}]
    report = {
        "verdict": checklist.verdict(rows),
        "finished": "now",
        "out": str(tmp_path),
        "preflight": {"gpu": "test"},
        "steps": rows,
    }
    assert report["verdict"] == "FAILED"
    text = checklist.markdown(report)
    assert "- **bad:** fail" in text
    assert "what it means: why" in text
    json.dumps(report)


def test_verdicts() -> None:
    assert checklist.verdict([{"status": "pass"}]) == "PASSED"
    assert checklist.verdict([{"status": "pass"}, {"status": "warn"}]) == "PASSED with warnings"
    assert checklist.verdict([{"status": "pass"}, {"status": "not run"}]) == "INCOMPLETE"
