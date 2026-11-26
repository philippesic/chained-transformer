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
RUNS = REPO / "runs"
KERNEL_TOL = 1e-2
QUEUE = "pueue add -g ct -- uv run --locked --no-sync python -m tools.serving_gpu_checklist"
MIN_CAPABILITY = (7, 5)
BF16_CAPABILITY = (8, 0)
ENGINE_OVERHEAD_GIB = 1.0
FREE_MARGIN_GIB = 0.25
DIAGNOSES: list[tuple[tuple[str, ...], str]] = [
    (
        ("is less than desired GPU memory utilization", "Free memory on device"),
        "vLLM found less free GPU memory than --gpu-mem asks for: another process is on the "
        "GPU (check nvidia-smi), or lower --gpu-mem",
    ),
    (
        ("No available memory for the cache blocks", "larger than the maximum number of tokens"),
        "vLLM's KV cache cannot hold --max-model-len tokens: raise --gpu-mem or lower "
        "--max-model-len",
    ),
    (
        ("CUDA out of memory", "OutOfMemoryError"),
        "the GPU ran out of memory: check nvidia-smi for other jobs, then lower --gpu-mem, "
        "--max-model-len or --num-prompts",
    ),
    (
        ("Bfloat16 is only supported", "bfloat16 tl.dot needs compute capability"),
        "this GPU has no bfloat16 (compute capability below 8.0): use float16; bfloat16 is "
        "5080 only, once BitDraft is done and Philip says so",
    ),
    (
        ("out of resource: shared memory", "OutOfResources"),
        "a Triton block size needs more shared memory than this GPU has (64 KiB a block on "
        "the 2080 Ti); use smaller blocks",
    ),
    (
        ("no kernel image is available",),
        "the installed torch or vLLM wheel has no kernels for this GPU's compute capability",
    ),
    (
        ("illegal memory access", "unspecified launch failure"),
        "a GPU kernel crashed: rerun this step alone (--only) with CUDA_LAUNCH_BLOCKING=1 to "
        "see which one",
    ),
    (
        ("No module named 'vllm'",),
        "this Python has no vLLM: point --vllm-python at the vLLM virtualenv",
    ),
    (
        ("does not appear to have a file named", "We couldn't connect to", "Can't load"),
        "a model or tokenizer is not downloaded: `huggingface-cli download` it, or pass "
        "--model tiny",
    ),
    (
        ("timed out after",),
        "the step hit --timeout: raise it, or run the step alone with --only",
    ),
]
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
    broke: str
    fix: str
    env: dict[str, str] = field(default_factory=dict)
    needs: str | None = None


def diagnose(text: str) -> list[str]:
    return [why for needles, why in DIAGNOSES if any(n in text for n in needles)]


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
    too_big = [
        r
        for r in checked
        if r["impl"].startswith("triton") and "OutOfResources" in r.get("error", "")
    ]
    for r in too_big:
        notes.remove(f"{r['impl']} at {r['tokens']} tokens ({r['mix']}) raised {r['error']}")
    for impl in sorted({r["impl"] for r in too_big}):
        notes.append(f"info: {impl} needs more shared memory than this GPU has; not timed")
    checked = [r for r in checked if r not in too_big]
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
    "try:\n    import triton; tv = triton.__version__\n"
    "except ImportError:\n    tv = None\n"
    "ok = torch.cuda.is_available()\n"
    "free, total = torch.cuda.mem_get_info() if ok else (0, 0)\n"
    "print(json.dumps({'torch': torch.__version__, 'triton': tv, 'cuda': ok,"
    " 'torch_cuda': torch.version.cuda,"
    " 'gpu': torch.cuda.get_device_name() if ok else None,"
    " 'capability': list(torch.cuda.get_device_capability()) if ok else None,"
    " 'arch_list': torch.cuda.get_arch_list() if ok else [],"
    " 'free_gib': round(free / 2**30, 2), 'total_gib': round(total / 2**30, 2)}))"
)
VLLM_PROBE = (
    "import json, torch, vllm, triton\n"
    "print(json.dumps({'vllm': vllm.__version__, 'triton': triton.__version__,"
    " 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}))"
)
QWEN_PROBE = (
    "import json\nfrom transformers import AutoConfig, AutoTokenizer\n"
    "c = AutoConfig.from_pretrained({base!r}); AutoTokenizer.from_pretrained({base!r})\n"
    "keys = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',"
    " 'num_key_value_heads', 'head_dim', 'vocab_size', 'tie_word_embeddings')\n"
    "print(json.dumps({{'ok': True, 'shape': {{k: getattr(c, k) for k in keys}}}}))"
)
GIB = 2**30


def model_sizes(shape: dict[str, Any], modes: int = 3) -> dict[str, float]:
    h, inter, layers = shape["hidden_size"], shape["intermediate_size"], shape["num_hidden_layers"]
    heads, kv_heads = shape["num_attention_heads"], shape["num_key_value_heads"]
    head_dim = shape.get("head_dim") or h // heads
    attn = 2 * h * heads * head_dim + 2 * h * kv_heads * head_dim + 2 * head_dim
    mlp = 3 * h * inter
    embed = shape["vocab_size"] * h * (1 if shape.get("tie_word_embeddings", True) else 2)
    base = embed + layers * (attn + mlp + 2 * h) + h
    chained = base + (modes - 1) * layers * mlp
    kv_per_token = layers * 2 * kv_heads * head_dim * 2
    return {
        "base_gib": round(base * 2 / GIB, 2),
        "chained_gib": round(chained * 2 / GIB, 2),
        "kv_gib_per_1k_tokens": round(kv_per_token * 1024 / GIB, 3),
    }


def memory_plan(
    total_gib: float,
    free_gib: float,
    gpu_mem: float,
    max_model_len: int,
    sizes: dict[str, float] | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    problems: list[str] = []
    warnings: list[str] = []
    want = gpu_mem * total_gib
    plan: dict[str, Any] = {"vllm_reserves_gib": round(want, 2), "free_gib": free_gib}
    if free_gib < want + FREE_MARGIN_GIB:
        fits = max(0.0, (free_gib - FREE_MARGIN_GIB) / total_gib) if total_gib else 0.0
        problems.append(
            f"vLLM reserves --gpu-mem {gpu_mem} x {total_gib:.1f} GiB = {want:.1f} GiB, but only "
            f"{free_gib:.1f} GiB is free; wait for the other job (nvidia-smi), or pass "
            f"--gpu-mem {int(fits * 100) / 100:.2f}"
        )
    if sizes is None:
        return problems, warnings, plan
    kv = sizes["kv_gib_per_1k_tokens"] * max_model_len / 1024
    engines = {
        "base": (want, sizes["base_gib"]),
        "chained": (want, sizes["chained_gib"]),
        "each separate model": (want / 3, sizes["base_gib"]),
    }
    for name, (budget, weights) in engines.items():
        left = budget - weights - ENGINE_OVERHEAD_GIB
        plan[name] = {
            "budget_gib": round(budget, 2),
            "weights_gib": weights,
            "kv_cache_gib_after_about_1_gib_overhead": round(left, 2),
            "kv_needed_for_max_model_len_gib": round(kv, 2),
        }
        if budget < weights + kv:
            problems.append(
                f"the vLLM bench's {name} engine gets {budget:.1f} GiB, less than its "
                f"{weights:.1f} GiB of weights plus {kv:.2f} GiB of KV cache for "
                f"--max-model-len {max_model_len}; raise --gpu-mem or lower --max-model-len"
            )
        elif left < kv:
            warnings.append(
                f"the vLLM bench's {name} engine has about {max(left, 0.0):.2f} GiB for its KV "
                f"cache after weights and overhead, under the {kv:.2f} GiB that "
                f"--max-model-len {max_model_len} needs; it may not start"
            )
    return problems, warnings, plan


def runs_on(arch: str, capability: tuple[int, ...]) -> bool:
    kind, _, digits = arch.partition("_")
    if not digits.isdigit() or len(capability) != 2:
        return False
    built = (int(digits[:-1]), int(digits[-1]))
    if kind == "sm":
        return built[0] == capability[0] and built[1] <= capability[1]
    return kind == "compute" and built <= capability


def gpu_problems(env: dict[str, Any]) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    capability = tuple(env.get("capability") or ())
    if not capability:
        return problems, notes
    sm = f"sm_{capability[0]}{capability[1]}"
    if capability < MIN_CAPABILITY:
        problems.append(
            f"{env.get('gpu')} is {sm}; vLLM 0.34 and the Triton kernel need sm_75 or newer"
        )
    arch_list = env.get("arch_list") or []
    if arch_list and not any(runs_on(a, capability) for a in arch_list):
        notes.append(
            f"torch {env.get('torch')} was built for {', '.join(arch_list)}, not {sm}; "
            "kernels may be JIT-compiled or missing"
        )
    if capability < BF16_CAPABILITY:
        notes.append(
            f"{sm} has no bfloat16: every step runs float16 or float32, and bfloat16 checks are "
            "5080 only, once BitDraft is done and Philip says so"
        )
    return problems, notes


def preflight(args: argparse.Namespace) -> tuple[bool, dict[str, Any], list[str]]:
    from tools.gpu_smoke import other_gpu_clients

    info: dict[str, Any] = {}
    problems: list[str] = []
    info["repo_env"] = probe(sys.executable, TORCH_PROBE)
    env = info["repo_env"]
    if shutil.which("nvidia-smi") is None:
        problems.append("nvidia-smi is not on PATH; run this on the GPU machine")
    else:
        try:
            info["other_gpu_clients"] = other_gpu_clients()
        except subprocess.CalledProcessError as exc:
            problems.append(f"nvidia-smi failed: {exc}")
    if info.get("other_gpu_clients") and not args.allow_other_clients:
        problems.append(
            "another process is using the GPU (ShaderLOD's pueue group `gpu` shares the "
            f"2080 Ti); wait for it, or queue this with `{QUEUE}`: "
            + "; ".join(info["other_gpu_clients"])
        )
    if "error" in env:
        problems.append(f"torch could not be imported here: {env['error']}")
    elif not env.get("cuda"):
        problems.append(
            f"torch {env.get('torch')} here sees no CUDA GPU (built for CUDA "
            f"{env.get('torch_cuda')}); check the driver with nvidia-smi"
        )
    gpu, notes = gpu_problems(env)
    problems += gpu
    vllm_python = Path(args.vllm_python).expanduser()
    if vllm_python.exists():
        info["vllm_env"] = probe(str(vllm_python), VLLM_PROBE)
    else:
        info["vllm_env"] = {"error": f"{vllm_python} does not exist"}
    vllm_env = info["vllm_env"]
    if "error" not in vllm_env and vllm_env.get("torch") != env.get("torch"):
        notes.append(
            f"the vLLM virtualenv has torch {vllm_env.get('torch')}, this one {env.get('torch')}"
        )
    sizes = None
    if args.model == "qwen":
        info["qwen"] = probe(sys.executable, QWEN_PROBE.format(base=args.base), timeout=300)
        if "shape" in info["qwen"]:
            sizes = model_sizes(info["qwen"]["shape"])
    names = [s.name for s in plan(args, Path("."), {})]
    if env.get("cuda") and "error" not in vllm_env and any(n.startswith("vllm") for n in names):
        bench = "vllm-bench" in names
        mem_problems, warnings, info["memory_plan"] = memory_plan(
            env["total_gib"],
            env["free_gib"],
            args.gpu_mem,
            args.max_model_len,
            sizes if bench else None,
        )
        problems += mem_problems
        notes += warnings
    info["notes"] = notes
    return not problems, info, problems


def preflight_lines(info: dict[str, Any]) -> list[str]:
    env = info.get("repo_env", {})
    vllm_env = info.get("vllm_env", {"error": "not checked"})
    capability = env.get("capability")
    sm = f"sm_{capability[0]}{capability[1]}" if capability else "no GPU"
    lines = [
        f"device: {env.get('gpu')} ({sm}), {env.get('total_gib')} GiB, "
        f"{env.get('free_gib')} GiB free",
        f"this env: torch {env.get('torch')} (CUDA {env.get('torch_cuda')}), "
        f"triton {env.get('triton')}",
        "vLLM env: "
        + (
            f"vllm {vllm_env.get('vllm')}, torch {vllm_env.get('torch')}, "
            f"triton {vllm_env.get('triton')}"
            if "error" not in vllm_env
            else f"not usable ({vllm_env['error']})"
        ),
    ]
    if info.get("other_gpu_clients"):
        lines.append("other GPU processes: " + "; ".join(info["other_gpu_clients"]))
    if "memory_plan" in info:
        lines.append("memory plan: " + json.dumps(info["memory_plan"]))
    lines += [f"note: {n}" for n in info.get("notes", [])]
    return lines


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
            "lint, type checks or CPU tests failed",
            "the log names the failing tool or test; fix it before trusting the GPU steps",
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
            "the Triton kernel gives wrong numbers or does not compile on this GPU",
            "the failing test names the dtype and shape; serve with mlp_impl 'select' "
            "until it is fixed",
        ),
        Step(
            "kernel-bench",
            [[*py, "tools.bench_mode_mlp", "--out", str(out / "bench_mode_mlp.json")]],
            json_judge(out / "bench_mode_mlp.json", kernel_outcome),
            "a kernel config raised or gave wrong numbers, or CUDA graph capture failed",
            "the notes name the config; serve with a config that passed, or 'select'",
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
            "vLLM's engine (eager) with the plugin does not decode like Hugging Face",
            "the first failing assert says whether tokens or modes differ; rerun with "
            "--only vllm-engine after a fix",
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
            "torch.compile or CUDA graphs break the plugin in vLLM",
            "serve with enforce_eager=True (what vllm-engine checks) until it is fixed",
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
                    "--num-prompts",
                    str(prompts),
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
                    "--max-model-len",
                    str(args.max_model_len),
                    "--out",
                    str(out / "bench_vllm.json"),
                ],
            ],
            json_judge(
                out / "bench_vllm.json",
                lambda r: bench_vllm_outcome(r, prompts, args.model == "qwen"),
            ),
            "an arm of the vLLM benchmark failed",
            "its error is in this step's log after its '+ ' command; the preflight's memory "
            "plan says whether --gpu-mem leaves room for each engine",
            env=venv,
            needs=vllm_needs or qwen_needs,
        ),
        Step(
            "paging-bench",
            [[*py, "tools.bench_paging", "--out", str(out / "bench_paging.json")]],
            json_judge(out / "bench_paging.json", lambda r: Outcome("pass", data={})),
            "copying idle modes between host and GPU failed",
            "the log has the traceback",
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
            "the log has the traceback; out of memory means lower --num-prompts",
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
        row["broke"] = step.broke
        row["fix"] = step.fix
        row["log_tail"] = tail(log)
        row["likely"] = diagnose(log.read_text(errors="replace")) if log.exists() else []
    return row


LABELS = {"pass": "PASS", "warn": "PASS", "fail": "FAIL", "not run": "SKIP"}


def summary_line(row: dict[str, Any]) -> str:
    label = LABELS.get(row["status"], row["status"].upper())
    line = f"{label} {row['name']} ({row['seconds']} s)"
    if row["status"] == "not run":
        return f"{line}: {'; '.join(row['notes'])}"
    if row["status"] == "warn":
        return f"{line} with warnings: {'; '.join(row['notes'])}"
    if row["status"] == "fail":
        parts = [f"broke: {row['broke']}"]
        if row["notes"]:
            parts.append(f"details: {'; '.join(row['notes'][:3])}")
        parts += [f"likely: {why}" for why in row.get("likely", [])]
        parts.append(f"fix: {row['fix']}")
        return f"{line}: " + " | ".join(parts)
    return line


def results_text(verdict_line: str, info: dict[str, Any], lines: list[str]) -> str:
    head = [f"serving GPU checklist: {verdict_line}", *preflight_lines(info)]
    return "\n".join([*head, "", *lines]) + "\n"


def write_results(text: str, stamp: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"serving-gpu-checklist-{stamp}.txt"
    path.write_text(text)
    return path


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Serving GPU checklist: {report['verdict']}", ""]
    lines.append(f"Written {report['finished']} in `{report['out']}`.")
    lines.append("")
    for row in report["steps"]:
        lines.append(f"- **{row['name']}:** {row['status']} ({row['seconds']} s)")
        lines += [f"  - {n}" for n in row["notes"]]
        if row.get("broke"):
            lines.append(f"  - what broke: {row['broke']}")
            lines += [f"  - likely: {why}" for why in row.get("likely", [])]
            lines.append(f"  - fix: {row['fix']}")
    lines += ["", "## Environment", "", *[f"- {x}" for x in preflight_lines(report["preflight"])]]
    lines += ["", "```json", json.dumps(report["preflight"], indent=2), "```"]
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
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--allow-other-clients", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("artifacts/serving/gpu-checklist") / stamp
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for step in plan(args, out, {}):
            print(f"== {step.name}")
            for cmd in step.commands:
                print("   " + " ".join(cmd))
        return 0
    ok, info, problems = preflight(args)
    print("\n".join(preflight_lines(info)), flush=True)
    if not ok:
        lines = [f"FAIL preflight: {p}" for p in problems]
        for line in lines:
            print(line, file=sys.stderr)
        (out / "report.json").write_text(
            json.dumps({"preflight": info, "problems": problems}, indent=2) + "\n"
        )
        path = write_results(results_text("NOT STARTED", info, lines), stamp)
        print(f"results: {path}")
        return 2
    rows = []
    for step in plan(args, out, info):
        print(f"== {step.name}", flush=True)
        row = run_step(step, out, args.timeout)
        print(summary_line(row), flush=True)
        rows.append(row)
    report: dict[str, Any] = {
        "verdict": verdict(rows),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "out": str(out),
        "preflight": info,
        "steps": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(markdown(report))
    text = results_text(report["verdict"], info, [summary_line(r) for r in rows])
    path = write_results(text, stamp)
    print(f"{report['verdict']}: {out / 'report.md'}")
    print(f"results to paste back: {path}")
    return 0 if report["verdict"] in ("PASSED", "PASSED with warnings") else 1


if __name__ == "__main__":
    sys.exit(main())
