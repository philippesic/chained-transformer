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
        "checkpoint": None,
        "num_prompts": 8,
        "max_tokens": 16,
        "gpu_mem": 0.5,
        "max_model_len": 1024,
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
    assert checklist.kernel_outcome(kernel_report(err=float("nan"))).status == "fail"
    nan_graph = {"captured": True, "max_abs_err_after_replays": float("nan")}
    assert checklist.kernel_outcome(kernel_report(graph=nan_graph)).status == "fail"
    lost = checklist.kernel_outcome(kernel_report(graph={"captured": False, "error": "sync"}))
    assert lost.status == "fail"
    assert "CUDA graph capture failed: sync" in lost.notes


def test_bench_vllm_outcome_flags_failed_arms_and_drift() -> None:
    arms = [
        {"arm": "base", "impl": None, "tok_per_s": 100.0},
        {"arm": "chained", "impl": "select", "tok_per_s": 90.0, "same_tokens_as_base": 7},
        {"arm": "separate", "impl": None, "error": "exit 1"},
    ]
    out = checklist.bench_vllm_outcome({"arms": arms[:2]}, prompts=8, exact=True)
    assert out.status == "warn"
    assert out.data["tok_per_s"] == {"base": 100.0, "chained/select": 90.0}
    assert checklist.bench_vllm_outcome({"arms": arms[:2]}, 8, exact=False).status == "pass"
    assert checklist.bench_vllm_outcome({"arms": arms}, prompts=8, exact=True).status == "fail"


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


def test_plan_passes_a_trained_checkpoint_to_the_vllm_bench(tmp_path: Path) -> None:
    info = {"vllm_env": {"cuda": True}}
    plain = checklist.plan(args(only=["vllm-bench"]), tmp_path, info)[0]
    assert "--checkpoint" not in plain.commands[0]
    trained = checklist.plan(args(only=["vllm-bench"], checkpoint=tmp_path), tmp_path, info)[0]
    prepare = trained.commands[0]
    assert prepare[prepare.index("--checkpoint") + 1] == str(tmp_path)


def test_run_step_logs_and_reports(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    write = f"import json; json.dump({{'arms': []}}, open({str(out)!r}, 'w'))"
    ok = checklist.Step(
        "ok",
        [[sys.executable, "-c", write]],
        checklist.json_judge(out, checklist.serving_outcome),
        "what broke",
        "the fix",
    )
    row = checklist.run_step(ok, tmp_path, timeout=60)
    assert row["status"] == "pass"
    assert "broke" not in row and "fix" not in row
    bad = checklist.Step(
        "bad",
        [[sys.executable, "-c", "print('boom: CUDA out of memory'); raise SystemExit(3)"]],
        checklist.exit_judge,
        "the bad step broke",
        "do the fix",
    )
    row = checklist.run_step(bad, tmp_path, timeout=60)
    assert row["status"] == "fail"
    assert row["notes"] == ["exit 3"]
    assert row["broke"] == "the bad step broke"
    assert row["fix"] == "do the fix"
    assert "boom" in row["log_tail"]
    assert any("ran out of memory" in why for why in row["likely"])
    line = checklist.summary_line(row)
    assert line.startswith("FAIL bad (")
    assert "broke: the bad step broke" in line and "fix: do the fix" in line
    assert "likely: the GPU ran out of memory" in line
    skipped = checklist.Step("later", [], checklist.exit_judge, "", "", needs="no vLLM")
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
    assert "what broke: the bad step broke" in text
    assert "fix: do the fix" in text
    json.dumps(report)


def test_verdicts() -> None:
    assert checklist.verdict([{"status": "pass"}]) == "PASSED"
    assert checklist.verdict([{"status": "pass"}, {"status": "warn"}]) == "PASSED with warnings"
    assert checklist.verdict([{"status": "pass"}, {"status": "not run"}]) == "INCOMPLETE"


QWEN3_0_6B = {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}
RTX_2080_TI_GIB = 10.75


def test_kernel_outcome_skips_block_sizes_too_big_for_the_gpu() -> None:
    report = kernel_report()
    too_big = "OutOfResources('out of resource: shared memory, Required: 73728')"
    report["rows"].append(
        {"tokens": 64, "mix": "three", "impl": "triton_64x128x32", "error": too_big}
    )
    out = checklist.kernel_outcome(report)
    assert out.status == "pass"
    assert "info: triton_64x128x32 needs more shared memory than this GPU has; not timed" in (
        out.notes
    )
    report["rows"] = [r for r in report["rows"] if r["impl"] != "triton_16x64x64"]
    assert checklist.kernel_outcome(report).status == "fail"


def test_model_sizes_match_qwen3_0_6b() -> None:
    sizes = checklist.model_sizes(QWEN3_0_6B)
    assert sizes["base_gib"] == pytest.approx(1.11, abs=0.01)
    assert sizes["chained_gib"] == pytest.approx(2.09, abs=0.01)
    assert sizes["kv_gib_per_1k_tokens"] == pytest.approx(0.109, abs=0.001)


def test_memory_plan_fits_the_2080_ti_with_the_defaults() -> None:
    sizes = checklist.model_sizes(QWEN3_0_6B)
    problems, warnings, plan = checklist.memory_plan(RTX_2080_TI_GIB, 10.4, 0.8, 1024, sizes)
    assert problems == [] and warnings == []
    assert plan["each separate model"]["budget_gib"] == pytest.approx(2.87, abs=0.01)
    busy, _, _ = checklist.memory_plan(RTX_2080_TI_GIB, 4.0, 0.8, 1024, sizes)
    assert len(busy) == 1 and "only 4.0 GiB is free" in busy[0] and "--gpu-mem 0.34" in busy[0]
    _, tight, _ = checklist.memory_plan(RTX_2080_TI_GIB, 10.4, 0.6, 4096, sizes)
    assert any("each separate model" in w for w in tight)
    small, _, _ = checklist.memory_plan(RTX_2080_TI_GIB, 10.4, 0.3, 1024, sizes)
    assert any("each separate model engine gets 1.1 GiB" in p for p in small)
    assert checklist.memory_plan(RTX_2080_TI_GIB, 10.4, 0.8, 1024, None)[0] == []


def test_gpu_problems_know_turing() -> None:
    turing = {"gpu": "RTX 2080 Ti", "capability": [7, 5], "arch_list": ["sm_75", "sm_80"]}
    problems, notes = checklist.gpu_problems(turing)
    assert problems == []
    assert len(notes) == 1 and "sm_75 has no bfloat16" in notes[0] and "5080 only" in notes[0]
    blackwell = {"gpu": "RTX 5080", "capability": [12, 0], "arch_list": ["sm_75", "sm_120"]}
    assert checklist.gpu_problems(blackwell) == ([], [])
    pascal, _ = checklist.gpu_problems({"gpu": "GTX 1080", "capability": [6, 1]})
    assert pascal and "need sm_75 or newer" in pascal[0]
    _, missing = checklist.gpu_problems({**turing, "arch_list": ["sm_80", "sm_90"]})
    assert any("not sm_75" in n for n in missing)
    assert checklist.runs_on("compute_70", (7, 5))
    assert checklist.runs_on("sm_80", (8, 6))
    assert not checklist.runs_on("sm_86", (8, 0))


def test_diagnose_names_likely_causes() -> None:
    text = "ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0"
    assert any("5080 only" in why for why in checklist.diagnose(text))
    assert any("--gpu-mem" in why for why in checklist.diagnose("Free memory on device (4.1/10.7"))
    assert checklist.diagnose("all good") == []


def test_summary_lines_and_results_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"name": "check", "status": "pass", "notes": [], "seconds": 3.0},
        {"name": "hf-bench", "status": "warn", "notes": ["batched differs"], "seconds": 9.0},
        {"name": "vllm-bench", "status": "not run", "notes": ["no vLLM"], "seconds": 0},
    ]
    lines = [checklist.summary_line(r) for r in rows]
    assert lines == [
        "PASS check (3.0 s)",
        "PASS hf-bench (9.0 s) with warnings: batched differs",
        "SKIP vllm-bench (0 s): no vLLM",
    ]
    monkeypatch.setattr(checklist, "RUNS", tmp_path / "runs")
    info = {"repo_env": {"gpu": "RTX 2080 Ti", "capability": [7, 5], "torch": "2.14"}}
    path = checklist.write_results(checklist.results_text("PASSED", info, lines), "stamp")
    assert path == tmp_path / "runs" / "serving-gpu-checklist-stamp.txt"
    text = path.read_text().splitlines()
    assert text[0] == "serving GPU checklist: PASSED"
    assert text[1].startswith("device: RTX 2080 Ti (sm_75)")
    assert text[-3:] == lines


def test_main_writes_a_results_file_when_preflight_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checklist, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(checklist.shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "argv", ["checklist", "--out", str(tmp_path / "out")])
    assert checklist.main() == 2
    [path] = (tmp_path / "runs").iterdir()
    text = path.read_text()
    assert text.startswith("serving GPU checklist: NOT STARTED")
    assert "FAIL preflight: nvidia-smi is not on PATH" in text
