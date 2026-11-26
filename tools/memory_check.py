from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from chained_transformer.memory import (
    QWEN3_0_6B,
    ModelShape,
    Setup,
    estimate,
)
from chained_transformer.train import ALLOC_CONF

ARMS = ("sft", "full", "lora", "moe", "separate")
PHASES = ("forward", "backward", "optimizer")
GiB = 2**30


def synthetic_rows(seq_len: int, rows: int, vocab: int, seed: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    n = seq_len + 1
    prompt = min(64, n // 8)
    think_end = prompt + int((n - prompt) * 0.7)
    code_end = think_end + (n - think_end) // 3
    out = []
    for _ in range(rows):
        modes = [0] * think_end + [2] * 8 + [1] * (code_end - think_end - 8) + [2] * (n - code_end)
        out.append(
            {
                "input_ids": [rng.randrange(vocab) for _ in range(n)],
                "mode_ids": modes[:n],
                "loss_mask": [0] * prompt + [1] * (n - prompt),
                "complete": True,
            }
        )
    return out


def shape_config(shape: ModelShape) -> Any:
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=shape.vocab,
        hidden_size=shape.hidden,
        intermediate_size=shape.intermediate,
        num_hidden_layers=shape.layers,
        num_attention_heads=shape.heads,
        num_key_value_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        max_position_embeddings=40960,
        rope_theta=1_000_000.0,
        tie_word_embeddings=shape.tied,
        torch_dtype="bfloat16",
    )


def save_random_model(shape: ModelShape, path: Path) -> None:
    import torch
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(shape_config(shape)).to(torch.bfloat16)
    model.save_pretrained(path)


def harness_setup(name: str, arm: str, seq_len: int, loss_chunk: int | None) -> Setup:
    if name == "current-turing":
        s = Setup(arm=arm, seq_len=seq_len, optimizer="torchao-eager", attention="math-fp32")  # type: ignore[arg-type]
        s = replace(s, loss_chunk=1024)
    else:
        s = Setup(arm=arm, seq_len=seq_len)  # type: ignore[arg-type]
    if loss_chunk:
        s = replace(s, loss_chunk=loss_chunk)
    return s


def train_argv(
    setup: Setup, base: Path, data: Path, out: Path, device: str, steps: int
) -> list[str]:
    argv = [
        "--arm", setup.arm,
        "--data", str(data),
        "--out", str(out),
        "--steps", str(steps),
        "--seq-len", str(setup.seq_len),
        "--device", device,
        "--base", str(base),
        "--loss-chunk", str(setup.loss_chunk),
        "--warmup", "1",
    ]  # fmt: skip
    if setup.arm == "separate":
        argv += ["--mode", "answer"]
    if setup.optimizer == "chunked":
        argv += ["--optimizer", "chunked", "--optim-chunk", str(setup.optim_chunk)]
    else:
        argv += ["--optimizer", "torchao"]
    argv += ["--attention", "auto" if setup.attention == "fused" else "sdpa"]
    if setup.freeze_embeddings:
        argv.append("--freeze-embeddings")
    return argv


class CudaPhases:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.current = "setup"
        self.peaks: dict[str, float] = {}
        self.reserved: dict[str, float] = {}

    def __call__(self, name: str) -> None:
        cuda = self.torch.cuda
        if cuda.is_initialized():
            cuda.synchronize()
            peak = cuda.max_memory_allocated() / GiB
            reserved = cuda.max_memory_reserved() / GiB
            self.peaks[self.current] = max(self.peaks.get(self.current, 0.0), peak)
            self.reserved[self.current] = max(self.reserved.get(self.current, 0.0), reserved)
            cuda.reset_peak_memory_stats()
        self.current = name

    def summary(self) -> dict[str, Any]:
        return {
            k: {"peak_gib": round(v, 4), "reserved_gib": round(self.reserved[k], 4)}
            for k, v in self.peaks.items()
        }


@contextlib.contextmanager
def turing_emulation(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    previous = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    finally:
        torch._dynamo.config.disable = previous


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    from chained_transformer.train import main as train_main

    setup = harness_setup(args.harness, args.arm, args.seq_len, args.loss_chunk)
    if args.freeze_embeddings:
        setup = replace(setup, freeze_embeddings=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        argv = train_argv(setup, args.base, args.data, out, args.device, args.steps)
        emulate = args.harness == "current-turing" and args.device == "cpu"
        if args.device == "cpu":
            from tools.memtrace import LiveBytes

            tracker = LiveBytes()
            with turing_emulation(emulate), tracker:
                train_main(argv, on_phase=tracker.set_phase, on_model=tracker.track_module)
            measured = tracker.summary()
        else:
            phases = CudaPhases()
            status, oom_phase = "ok", None
            try:
                train_main(argv, on_phase=phases)
            except Exception as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                status, oom_phase = "oom", phases.current
            phases("done")
            measured = phases.summary()
            measured["status"] = status
            measured["oom_phase"] = oom_phase
            measured["reserved_peak_gib"] = max(
                (v["reserved_gib"] for v in measured.values() if isinstance(v, dict)), default=0.0
            )
        summary_path = out / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {"setup": setup.__dict__, "measured": measured, "train_summary": summary}


def predicted(shape: ModelShape, setup: Setup) -> dict[str, float]:
    est = estimate(shape, setup)
    ph = est.phases
    return {
        "forward": ph["forward"] / GiB,
        "backward": max(ph["loss_backward"], ph["layer_backward"]) / GiB,
        "optimizer": ph["optimizer_step"] / GiB,
        "peak": est.peak / GiB,
    }


def compare(shape: ModelShape, result: dict[str, Any]) -> dict[str, Any]:
    setup = Setup(**result["setup"])
    pred = predicted(shape, setup)
    rows = {}
    for phase in PHASES:
        got = result["measured"].get(phase, {}).get("peak_gib")
        want = pred[phase]
        rows[phase] = {
            "measured_gib": got,
            "predicted_gib": round(want, 4),
            "ratio": round(got / want, 3) if got else None,
        }
    measured_peak = max((v["measured_gib"] or 0.0) for v in rows.values())
    rows["peak"] = {
        "measured_gib": round(measured_peak, 4),
        "predicted_gib": round(pred["peak"], 4),
        "ratio": round(measured_peak / pred["peak"], 3) if measured_peak else None,
    }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure training memory per arm and compare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seq-len", type=int, nargs="+", default=[512])
    parser.add_argument("--layers", type=int, default=None, help="random model with N layers")
    parser.add_argument("--base", type=Path, default=None, help="real model dir or hub name")
    parser.add_argument("--harness", choices=["fixed", "current-turing"], default="fixed")
    parser.add_argument("--loss-chunk", type=int, default=None)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--budget-gib", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/memory-check.json"))
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", default="sft", help=argparse.SUPPRESS)
    parser.add_argument("--data", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single:
        args.seq_len = args.seq_len[0]
        result = run_single(args)
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        return 0

    shape = QWEN3_0_6B if args.layers is None else replace(QWEN3_0_6B, layers=args.layers)
    report: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = args.base
        if base is None:
            base = root / "model"
            save_random_model(shape, base)
        from datasets import Dataset

        for seq in args.seq_len:
            data = root / f"data-{seq}"
            Dataset.from_list(synthetic_rows(seq, 4, shape.vocab, seq)).save_to_disk(str(data))
            for arm in args.arms:
                result_path = root / f"{arm}-{seq}.json"
                cmd = [
                    sys.executable, "-m", "tools.memory_check", "--single",
                    "--arm", arm, "--seq-len", str(seq), "--device", args.device,
                    "--base", str(base), "--data", str(data), "--result", str(result_path),
                    "--harness", args.harness, "--steps", str(args.steps),
                ]  # fmt: skip
                if args.loss_chunk:
                    cmd += ["--loss-chunk", str(args.loss_chunk)]
                if args.freeze_embeddings:
                    cmd.append("--freeze-embeddings")
                env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
                alloc = ALLOC_CONF if args.harness == "fixed" else "expandable_segments:False"
                env["PYTORCH_ALLOC_CONF"] = env["PYTORCH_CUDA_ALLOC_CONF"] = alloc
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode != 0 or not result_path.exists():
                    tail = (proc.stderr or proc.stdout)[-2000:]
                    entry: dict[str, Any] = {"arm": arm, "seq_len": seq, "error": tail}
                    print(f"{arm:<9}{seq:>7}  FAILED\n{tail}", flush=True)
                    report.append(entry)
                    continue
                result = json.loads(result_path.read_text())
                rows = compare(shape, result)
                entry = {"arm": arm, "seq_len": seq, **result, "comparison": rows}
                verdict = ""
                if args.budget_gib is not None:
                    peak = max(
                        rows["peak"]["measured_gib"],
                        result["measured"].get("reserved_peak_gib", 0.0),
                    )
                    status = result["measured"].get("status", "ok")
                    fits = status == "ok" and peak <= args.budget_gib
                    entry["fits_budget"] = fits
                    verdict = "fits" if fits else "OVER"
                report.append(entry)
                print(
                    f"{arm:<9}{seq:>7}  measured {rows['peak']['measured_gib']:.2f} GiB"
                    f"  predicted {rows['peak']['predicted_gib']:.2f} GiB"
                    f"  ratio {rows['peak']['ratio']}  {verdict}",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    failed = any("error" in r or r.get("fits_budget") is False for r in report)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
