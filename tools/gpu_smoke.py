from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PYTHON = [sys.executable, "-m"]


def other_gpu_clients() -> list[str]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def run(name: str, cmd: list[str], log_dir: Path) -> bool:
    print(f"== {name}", flush=True)
    with (log_dir / f"{name}.log").open("w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"   {'ok' if code == 0 else f'FAILED (exit {code})'}", flush=True)
    return code == 0


def train_cmd(arm: str, out: Path, data: Path, steps: int, seq_len: int) -> list[str]:
    cmd = [
        *PYTHON,
        "chained_transformer.train",
        "--arm",
        arm,
        "--data",
        str(data),
        "--out",
        str(out),
    ]
    cmd += ["--steps", str(steps), "--seq-len", str(seq_len), "--complete-only"]
    return cmd + (["--mode", "answer"] if arm == "separate" else [])


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU qualification: equivalence, memory, 50 steps")
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/ot3-slice-4k"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-other-clients", action="store_true")
    args = parser.parse_args()
    clients = other_gpu_clients()
    if clients and not args.allow_other_clients:
        print("GPU busy; not starting. Other compute clients:", *clients, sep="\n  ")
        print("Wait for them to finish; this project queues in pueue's ct group, never gpu.")
        return 2
    out = args.out or Path("artifacts/smoke") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    steps: list[tuple[str, list[str]]] = [
        (
            "equivalence-bf16",
            [*PYTHON, "tools.gpu_equivalence", "--out", str(out / "equivalence-bf16.json")],
        ),
        (
            "equivalence-fp32-full",
            [*PYTHON, "tools.gpu_equivalence", "--variants", "full", "--dtype", "float32"]
            + ["--out", str(out / "equivalence-fp32-full.json")],
        ),
        ("train-full-50", train_cmd("full", out / "train-full-50", args.data, 50, 4096)),
    ]
    for arm in ("sft", "lora", "moe", "separate"):
        steps.append(
            (f"train-{arm}-10", train_cmd(arm, out / f"train-{arm}-10", args.data, 10, 4096))
        )
    steps.append(("train-full-16k", train_cmd("full", out / "train-full-16k", args.data, 5, 16384)))
    results = {name: run(name, cmd, out) for name, cmd in steps}
    summaries = {
        p.parent.name: json.loads(p.read_text()) for p in sorted(out.glob("train-*/summary.json"))
    }
    report = {"passed": results, "train_summaries": summaries}
    (out / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
