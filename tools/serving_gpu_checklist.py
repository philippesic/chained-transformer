from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
KERNEL_TOL = 1e-2
QUEUE = "pueue add -g gpu -- uv run --locked --no-sync python -m tools.serving_gpu_checklist"
VLLM_SETUP = (
    "uv venv ~/venvs/vllm && uv pip install --python ~/venvs/vllm vllm==0.34.0 pytest, "
    "or pass --vllm-python"
)


@dataclass
class Outcome:
    status: str
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    name: str
    commands: list[list[str]]
    judge: Callable[[int], Outcome]
    hint: str
    env: dict[str, str] = field(default_factory=dict)
    needs: str | None = None


def tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def exit_judge(code: int) -> Outcome:
    return Outcome("pass" if code == 0 else "fail", [] if code == 0 else [f"exit {code}"])


def junit_counts(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors")}
    skips = [
        f"{case.get('name')}: {skip.get('message', '')}"
        for case in root.iter("testcase")
        for skip in case.findall("skipped")
    ]
    failed = [
        case.get("name", "")
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    return {**counts, "skipped": len(skips), "skips": skips, "failed": failed}


def pytest_judge(xml: Path, allow_skips: bool) -> Callable[[int], Outcome]:
    def judge(code: int) -> Outcome:
        if not xml.exists():
            return Outcome("fail", [f"exit {code}, pytest wrote no report"])
        counts = junit_counts(xml)
        notes = [f"failed: {name}" for name in counts["failed"]]
        status = "pass"
        if code != 0 or counts["failures"] or counts["errors"]:
            status = "fail"
            notes.insert(0, f"exit {code}")
        if counts["tests"] == counts["skipped"]:
            status = "fail"
            notes.append("every test was skipped, so nothing was checked")
        elif counts["skipped"] and not allow_skips:
            status = "fail"
            notes.append("tests skipped that must run on a GPU")
        notes += [f"skipped {s}" for s in counts["skips"]]
        data = {k: counts[k] for k in ("tests", "failures", "errors", "skipped")}
        return Outcome(status, notes, data)

    return judge


def kernel_outcome(report: dict[str, Any], tol: float = KERNEL_TOL) -> Outcome:
    notes: list[str] = []
    rows = report.get("rows", [])
    checked = [r for r in rows if not r["impl"].startswith(("dense", "vllm_"))]
    for r in rows:
        if "error" in r:
            where = f"{r['impl']} at {r['tokens']} tokens ({r['mix']})"
            if r["impl"].startswith("vllm_"):
                notes.append(f"info: {where} did not run: {r['error']}")
            else:
                notes.append(f"{where} raised {r['error']}")
    errors = [r["max_abs_err"] for r in checked if "max_abs_err" in r]
    worst = max(errors, default=float("nan"))
    bad = [r for r in checked if "error" not in r and not r.get("max_abs_err", 0.0) <= tol]
    for r in bad:
        notes.append(
            f"{r['impl']} at {r['tokens']} tokens ({r['mix']}) is off by {r['max_abs_err']:.2e}"
        )
    triton = [r for r in checked if r["impl"].startswith("triton")]
    if not triton:
        notes.append("no Triton rows: the kernel did not run")
    graph = report.get("cuda_graph") or {}
    if not graph.get("captured"):
        notes.append(f"CUDA graph capture failed: {graph.get('error', 'not attempted')}")
    elif not graph.get("max_abs_err_after_replays", float("nan")) <= tol:
        notes.append(f"CUDA graph replays are off by {graph['max_abs_err_after_replays']:.2e}")
    failed = [n for n in notes if not n.startswith("info:")]
    best: dict[str, Any] = {}
    for n in sorted({r["tokens"] for r in rows}):
        three = [r for r in rows if r["tokens"] == n and r["mix"] == "three" and "us" in r]
        fastest = min(
            (r for r in three if r["impl"].startswith("triton")),
            key=lambda r: r["us"],
            default=None,
        )
        loop = next((r for r in three if r["impl"] == "loop"), None)
        if fastest and loop:
            best[str(n)] = {
                "triton": fastest["impl"],
                "us": round(fastest["us"], 1),
                "loop_us": round(loop["us"], 1),
            }
    data = {
        "gpu": report.get("gpu"),
        "worst_abs_err": worst,
        "cuda_graph": graph,
        "three_modes": best,
    }
    return Outcome("fail" if failed else "pass", notes, data)


def bench_vllm_outcome(report: dict[str, Any], prompts: int | None, exact: bool) -> Outcome:
    notes: list[str] = []
    status = "pass"
    speeds: dict[str, Any] = {}
    for row in report.get("arms", []):
        name = row["arm"] if not row.get("impl") else f"{row['arm']}/{row['impl']}"
        if "error" in row:
            status = "fail"
            notes.append(f"{name} failed ({row['error']}); see its log")
            continue
        speeds[name] = row.get("tok_per_s")
        same = row.get("same_tokens_as_base")
        drifted = prompts is not None and same is not None and same < prompts
        if exact and name == "chained/select" and drifted:
            if status == "pass":
                status = "warn"
            notes.append(
                f"chained/select matched the base model on {same} of {prompts} prompts; with "
                "unperturbed modes it should match all of them"
            )
        if row.get("arm") == "chained" and row.get("mixed_steps") == 0:
            notes.append(f"{name} never ran a batch with mixed modes")
    return Outcome(status, notes, {"tok_per_s": speeds})


def json_judge(path: Path, check: Callable[[dict[str, Any]], Outcome]) -> Callable[[int], Outcome]:
    def judge(code: int) -> Outcome:
        if code != 0:
            return Outcome("fail", [f"exit {code}"])
        if not path.exists():
            return Outcome("fail", [f"exit 0 but {path.name} was not written"])
        return check(json.loads(path.read_text()))

    return judge


def serving_outcome(report: dict[str, Any]) -> Outcome:
    arms = {a["arm"]: a for a in report.get("arms", [])}
    batched = arms.get("chained_batched", {})
    notes = []
    status = "pass"
    if batched.get("matches_per_sequence") is False:
        status = "warn"
        notes.append("batched chained decoding differs from per-sequence decoding")
    data = {name: a.get("tok_per_s") for name, a in arms.items()}
    return Outcome(
        status, notes, {"tok_per_s": data, "peak_alloc_gib": report.get("peak_alloc_gib")}
    )


def probe(python: str, code: str, timeout: float = 120) -> dict[str, Any]:
    try:
        done = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=timeout, cwd=REPO
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": repr(exc)}
    if done.returncode != 0:
        return {"error": (done.stderr or done.stdout).strip().splitlines()[-1:]}
    return dict(json.loads(done.stdout.strip().splitlines()[-1]))


TORCH_PROBE = (
    "import json, torch\n"
    "ok = torch.cuda.is_available()\n"
    "free, total = torch.cuda.mem_get_info() if ok else (0, 0)\n"
    "print(json.dumps({'torch': torch.__version__, 'cuda': ok,"
    " 'gpu': torch.cuda.get_device_name() if ok else None,"
    " 'capability': list(torch.cuda.get_device_capability()) if ok else None,"
    " 'free_gib': round(free / 2**30, 2), 'total_gib': round(total / 2**30, 2)}))"
)
VLLM_PROBE = (
    "import json, torch, vllm, triton\n"
    "print(json.dumps({'vllm': vllm.__version__, 'triton': triton.__version__,"
    " 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}))"
)
QWEN_PROBE = (
    "import json\nfrom transformers import AutoConfig, AutoTokenizer\n"
    "AutoConfig.from_pretrained({base!r}); AutoTokenizer.from_pretrained({base!r})\n"
    "print(json.dumps({{'ok': True}}))"
)


def preflight(args: argparse.Namespace) -> tuple[bool, dict[str, Any], list[str]]:
    from tools.gpu_smoke import other_gpu_clients

    info: dict[str, Any] = {}
    problems: list[str] = []
    if shutil.which("nvidia-smi") is None:
        return False, info, ["nvidia-smi is not on PATH; run this on the GPU machine"]
    try:
        info["other_gpu_clients"] = other_gpu_clients()
    except subprocess.CalledProcessError as exc:
        return False, info, [f"nvidia-smi failed: {exc}"]
    if info["other_gpu_clients"] and not args.allow_other_clients:
        problems.append(
            "the GPU is busy; wait for these, or queue behind them with "
            f"`{QUEUE}`: " + "; ".join(info["other_gpu_clients"])
        )
    info["repo_env"] = probe(sys.executable, TORCH_PROBE)
    if not info["repo_env"].get("cuda"):
        problems.append(f"torch in this environment sees no CUDA GPU: {info['repo_env']}")
    vllm_python = Path(args.vllm_python).expanduser()
    if vllm_python.exists():
        info["vllm_env"] = probe(str(vllm_python), VLLM_PROBE)
    else:
        info["vllm_env"] = {"error": f"{vllm_python} does not exist"}
    if args.model == "qwen":
        info["qwen"] = probe(sys.executable, QWEN_PROBE.format(base=args.base), timeout=300)
    return not problems, info, problems


def plan(args: argparse.Namespace, out: Path, info: dict[str, Any]) -> list[Step]:
    py = [sys.executable, "-m"]
    vpy = [str(Path(args.vllm_python).expanduser()), "-m"]
    venv = {"PYTHONPATH": str(REPO)}
    vllm_env = info.get("vllm_env", {})
    vllm_needs = None
    if "error" in vllm_env:
        vllm_needs = f"no usable vLLM environment ({vllm_env['error']}); set one up: {VLLM_SETUP}"
    elif not vllm_env.get("cuda"):
        vllm_needs = "torch in the vLLM environment sees no CUDA GPU"
    qwen_needs = None
    if args.model == "qwen" and "error" in info.get("qwen", {}):
        qwen_needs = (
            f"{args.base} could not be loaded ({info['qwen']['error']}); download it with "
            f"`huggingface-cli download {args.base}`, or pass --model tiny"
        )
    bench_dir = out / "vllm-bench"
    prompts = args.num_prompts
    steps = [
        Step(
            "check",
            [[*py, "tools.check"]],
            exit_judge,
            "lint, types or CPU tests failed; the log names the step",
        ),
        Step(
            "triton-gpu",
            [
                [
                    *py,
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_serving_triton.py",
                    f"--junitxml={out / 'triton-gpu.xml'}",
                ]
            ],
            pytest_judge(out / "triton-gpu.xml", allow_skips=False),
            "the Triton kernel gives wrong numbers or does not compile on this GPU; "
            "serve with mlp_impl 'select' until it is fixed",
        ),
        Step(
            "kernel-bench",
            [[*py, "tools.bench_mode_mlp", "--out", str(out / "bench_mode_mlp.json")]],
            json_judge(out / "bench_mode_mlp.json", kernel_outcome),
            "a kernel row raised or is off, or CUDA graph capture of the kernel failed",
        ),
        Step(
            "vllm-engine",
            [
                [
                    *vpy,
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_serving_vllm.py",
                    f"--junitxml={out / 'vllm-engine.xml'}",
                ]
            ],
            pytest_judge(out / "vllm-engine.xml", allow_skips=False),
            "vLLM's engine (eager) does not decode like Hugging Face with the plugin; "
            "the first failing assert says whether tokens or modes differ",
            env=venv,
            needs=vllm_needs,
        ),
        Step(
            "vllm-graphs",
            [
                [
                    *vpy,
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_serving_vllm_graphs.py",
                    f"--junitxml={out / 'vllm-graphs.xml'}",
                ]
            ],
            pytest_judge(out / "vllm-graphs.xml", allow_skips=False),
            "torch.compile or CUDA graphs break the Triton path in vLLM; "
            "eager mode (vllm-engine) is the fallback",
            env=venv,
            needs=vllm_needs,
        ),
        Step(
            "vllm-bench",
            [
                [
                    *vpy,
                    "tools.bench_vllm",
                    "prepare",
                    "--dir",
                    str(bench_dir),
                    "--model",
                    args.model,
                    "--base",
                    args.base,
                    "--num-prompts",
                    str(prompts),
                    *(["--checkpoint", str(args.checkpoint)] if args.checkpoint else []),
                ],
                [
                    *vpy,
                    "tools.bench_vllm",
                    "all",
                    "--dir",
                    str(bench_dir),
                    "--model",
                    args.model,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--gpu-mem",
                    str(args.gpu_mem),
                    "--out",
                    str(out / "bench_vllm.json"),
                ],
            ],
            json_judge(
                out / "bench_vllm.json",
                lambda r: bench_vllm_outcome(
                    r, prompts, args.model == "qwen" and args.checkpoint is None
                ),
            ),
            "an arm failed; its own log lines are in this step's log after its '+ ' command",
            env=venv,
            needs=vllm_needs or qwen_needs,
        ),
        Step(
            "paging-bench",
            [[*py, "tools.bench_paging", "--out", str(out / "bench_paging.json")]],
            json_judge(out / "bench_paging.json", lambda r: Outcome("pass", data={})),
            "host-to-device paging of idle modes failed",
        ),
        Step(
            "hf-bench",
            [
                [
                    *py,
                    "tools.bench_serving",
                    "--model",
                    args.model,
                    "--base",
                    args.base,
                    "--dtype",
                    "float16",
                    "--out",
                    str(out / "bench_serving.json"),
                ]
            ],
            json_judge(out / "bench_serving.json", serving_outcome),
            "the Hugging Face serving comparison failed",
            needs=qwen_needs,
        ),
    ]
    names = [s.name for s in steps]
    for name in [*args.only, *args.skip]:
        if name not in names:
            raise SystemExit(f"unknown step {name!r}; steps are {', '.join(names)}")
    if args.only:
        steps = [s for s in steps if s.name in args.only]
    return [s for s in steps if s.name not in args.skip]


def run_step(step: Step, out: Path, timeout: float) -> dict[str, Any]:
    log = out / f"{step.name}.log"
    started = time.perf_counter()
    if step.needs:
        return {"name": step.name, "status": "not run", "notes": [step.needs], "seconds": 0}
    code = 0
    env = {**os.environ, **step.env}
    with log.open("w") as fh:
        for cmd in step.commands:
            fh.write("+ " + " ".join(cmd) + "\n")
            fh.flush()
            try:
                code = subprocess.run(
                    cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO, timeout=timeout
                ).returncode
            except subprocess.TimeoutExpired:
                code = -1
                fh.write(f"\ntimed out after {timeout:.0f} s\n")
            if code != 0:
                break
    try:
        outcome = step.judge(code)
    except Exception as exc:
        outcome = Outcome("fail", [f"could not read the step's output: {exc!r}"])
    row: dict[str, Any] = {
        "name": step.name,
        "status": outcome.status,
        "notes": outcome.notes,
        "data": outcome.data,
        "seconds": round(time.perf_counter() - started, 1),
        "log": log.name,
    }
    if outcome.status != "pass":
        row["hint"] = step.hint
        row["log_tail"] = tail(log)
    return row


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Serving GPU checklist: {report['verdict']}", ""]
    lines.append(f"Written {report['finished']} in `{report['out']}`.")
    lines.append("")
    for row in report["steps"]:
        lines.append(f"- **{row['name']}:** {row['status']} ({row['seconds']} s)")
        lines += [f"  - {n}" for n in row["notes"]]
        if row.get("hint"):
            lines.append(f"  - what it means: {row['hint']}")
    lines += ["", "## Environment", "", "```json", json.dumps(report["preflight"], indent=2), "```"]
    for row in report["steps"]:
        if row.get("data"):
            lines += [
                "",
                f"## {row['name']}",
                "",
                "```json",
                json.dumps(row["data"], indent=2),
                "```",
            ]
        if row.get("log_tail"):
            lines += ["", f"## {row['name']} log tail", "", "```", row["log_tail"], "```"]
    return "\n".join(lines) + "\n"


def verdict(rows: list[dict[str, Any]]) -> str:
    statuses = {r["status"] for r in rows}
    if "fail" in statuses:
        return "FAILED"
    if "not run" in statuses:
        return "INCOMPLETE"
    return "PASSED with warnings" if "warn" in statuses else "PASSED"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the serving GPU checklist and write one report"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--vllm-python", default="~/venvs/vllm/bin/python")
    parser.add_argument("--model", choices=["qwen", "tiny"], default="qwen")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-mem", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--allow-other-clients", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out or Path("artifacts/serving/gpu-checklist") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for step in plan(args, out, {}):
            print(f"== {step.name}")
            for cmd in step.commands:
                print("   " + " ".join(cmd))
        return 0
    ok, info, problems = preflight(args)
    print(json.dumps(info, indent=2), flush=True)
    if not ok:
        for line in problems:
            print(f"not starting: {line}", file=sys.stderr)
        (out / "report.json").write_text(
            json.dumps({"preflight": info, "problems": problems}, indent=2) + "\n"
        )
        return 2
    rows = []
    for step in plan(args, out, info):
        print(f"== {step.name}", flush=True)
        row = run_step(step, out, args.timeout)
        print(f"   {row['status']} ({row['seconds']} s)", *row["notes"], sep="\n   ", flush=True)
        rows.append(row)
    report = {
        "verdict": verdict(rows),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "out": str(out),
        "preflight": info,
        "steps": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(markdown(report))
    print(f"{report['verdict']}: {out / 'report.md'}")
    return 0 if report["verdict"] in ("PASSED", "PASSED with warnings") else 1


if __name__ == "__main__":
    sys.exit(main())
