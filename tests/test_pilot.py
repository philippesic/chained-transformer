from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Qwen3ForCausalLM

from chained_transformer.checkpoint import load_checkpoint
from tools import pilot


def _plan(tmp_path: Path, **extra: str) -> Path:
    argv = ["plan", "--name", "p", "--data", str(tmp_path / "data"), "--root", str(tmp_path)]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", value]
    pilot.main(argv)
    return tmp_path / "p"


def test_plan_lists_every_arm_and_seed(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    plan = json.loads((root / "plan.json").read_text())
    ids = [r["id"] for r in plan["train"]]
    assert len(ids) == 14 and "separate-s1-code" in ids and "moe-s0" in ids
    assert [e["id"] for e in plan["eval"]] == [f"{a}-s{s}" for s in (0, 1) for a in pilot.ARMS]
    lora = next(r for r in plan["train"] if r["id"] == "lora-s0")["argv"]
    assert lora[lora.index("--lr") + 1] == "0.0001" and "--complete-only" in lora
    separate = next(e for e in plan["eval"] if e["id"] == "separate-s1")["argv"]
    assert separate[0] == "--separate" and separate[1].endswith("separate-s1-think/checkpoint")
    assert "--greedy" in separate
    commands = (root / "commands.sh").read_text().splitlines()
    assert sum("train-one" in c for c in commands) == 14


def test_each_eval_runs_right_after_its_last_training_run(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    commands = (root / "commands.sh").read_text().splitlines()
    steps = [(c.split()[-5], c.split()[-1]) for c in commands if "-one " in c]
    assert steps[:4] == [
        ("train-one", "sft-s0"),
        ("eval-one", "sft-s0"),
        ("train-one", "full-s0"),
        ("eval-one", "full-s0"),
    ]
    answer = steps.index(("train-one", "separate-s0-answer"))
    assert steps[answer + 1] == ("eval-one", "separate-s0")
    assert len(steps) == 24 and steps[-1] == ("eval-one", "separate-s1")
    first_separate = min(i for i, (_, run) in enumerate(steps) if run.startswith("separate"))
    assert steps.index(("eval-one", "moe-s1")) < first_separate
    sft = json.loads((root / "plan.json").read_text())["eval"][0]["argv"]
    assert sft[sft.index("--batch-size") + 1] == "auto"


def test_plan_refuses_to_change_a_fixed_config(tmp_path: Path) -> None:
    _plan(tmp_path)
    _plan(tmp_path)
    with pytest.raises(SystemExit, match="different config"):
        _plan(tmp_path, steps="5")


def test_dry_run_starts_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _plan(tmp_path)
    capsys.readouterr()
    pilot.main(["run", "--root", str(root), "--dry-run", "--only", "full-s0", "separate-s0"])
    out = capsys.readouterr().out
    assert "6 steps pending (dry run, nothing started)" in out
    assert not list(root.glob("*/train-status.json"))


def test_busy_gpu_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _plan(tmp_path)
    monkeypatch.setattr(pilot, "other_gpu_processes", lambda: ["4242, python, 12000 MiB"])
    started: list[Any] = []
    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **k: started.append(a))
    with pytest.raises(SystemExit) as stop:
        pilot.main(["run", "--root", str(root)])
    assert stop.value.code == 3 and not started


def test_train_one_saves_a_loadable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_model: Qwen3ForCausalLM
) -> None:
    from chained_transformer import train
    from chained_transformer.model import ARMS, build

    root = _plan(tmp_path, device="cpu")
    seen: dict[str, Any] = {}

    def fake_main(argv: list[str], on_model: Any = None, on_phase: Any = None) -> None:
        arm = next(a for a in ARMS if a == argv[argv.index("--arm") + 1])
        seen["argv"] = argv
        build(tiny_model, arm, pilot.surgery_for(pilot.load_plan(root)[0], arm))
        on_model(tiny_model)

    monkeypatch.setattr(train, "main", fake_main)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda name: None)
    pilot.main(["train-one", "--root", str(root), "--run", "lora-s1"])
    assert seen["argv"][seen["argv"].index("--seed") + 1] == "1"
    assert pilot.read_status(root, "lora-s1", "train")["state"] == "done"
    loaded = load_checkpoint(root / "lora-s1" / "checkpoint", torch.float32)
    assert loaded.routing is not None
    state = json.loads((root / "lora-s1" / "checkpoint" / "training_state.json").read_text())
    assert state["arm"] == "lora" and state["seed"] == 1


def test_eval_needs_trained_checkpoints(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    with pytest.raises(SystemExit, match="needs trained checkpoints"):
        pilot.main(["eval-one", "--root", str(root), "--run", "separate-s0"])


def _report(root: Path, run_id: str, math: float, code: float) -> None:
    out = root / run_id / "eval"
    out.mkdir(parents=True)
    benches = {
        "math500": {"scores": {"accuracy": math}},
        "humanevalplus": {"scores": {"accuracy": code}},
    }
    (out / "report.json").write_text(json.dumps({"benchmarks": benches}))


@pytest.mark.parametrize(
    ("full", "expected"),
    [(0.45, "full beats sft"), (0.41, "kill rule met"), (None, "mixed")],
)
def test_compare_applies_the_kill_rule(
    tmp_path: Path, full: float | None, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _plan(tmp_path)
    for seed in (0, 1):
        _report(root, f"sft-s{seed}", 0.40, 0.40)
        _report(root, f"moe-s{seed}", 0.41, 0.41)
        _report(root, f"lora-s{seed}", 0.445, 0.445)
        score = full if full is not None else (0.45 if seed == 0 else 0.40)
        _report(root, f"full-s{seed}", score, score)
    pilot.main(["compare", "--root", str(root)])
    result = json.loads((root / "compare.json").read_text())
    assert result["kill_rule"].startswith(expected)
    if full == 0.45:
        assert result["margins"]["0"] == {
            "full_minus_sft": 5.0,
            "full_minus_moe": 4.0,
            "lora_minus_full": -0.5,
        }
        assert result["lora"].startswith("lora is within")


def test_compare_waits_for_missing_reports(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    _report(root, "full-s0", 0.5, 0.5)
    pilot.main(["compare", "--root", str(root)])
    assert json.loads((root / "compare.json").read_text())["kill_rule"].startswith("incomplete")
