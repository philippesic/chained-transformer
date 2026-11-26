from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from datasets import Dataset

from chained_transformer.memory import QWEN3_0_6B
from tools import pilot, pilot_budget
from tools.pilot import PilotConfig
from tools.pilot_budget import GPUS, train_flops_per_token

SHAPE = QWEN3_0_6B


def test_flops_per_token_follow_each_arm() -> None:
    attn, mlp, head = pilot_budget.layer_params(SHAPE)
    assert SHAPE.layers * (attn + mlp) + head == 595_984_384
    L = SHAPE.layers
    sft = train_flops_per_token(SHAPE, "sft", 4096)
    assert train_flops_per_token(SHAPE, "full", 4096) == sft
    assert train_flops_per_token(SHAPE, "separate", 4096) == sft
    assert train_flops_per_token(SHAPE, "moe", 4096) - sft == 8 * L * mlp
    adapter = 3 * 64 * (SHAPE.hidden + SHAPE.intermediate)
    assert train_flops_per_token(SHAPE, "lora", 4096) - sft == 8 * L * adapter - 2 * L * mlp
    longer = sft - train_flops_per_token(SHAPE, "sft", 1024)
    assert longer == 4 * L * 4 * SHAPE.q_width * (4096 - 1024) / 2
    assert 6.5e9 < sft < 6.8e9


def test_measured_steps_scale_to_the_plan() -> None:
    cfg = PilotConfig(name="p", data="")
    gpu = GPUS["5080"]
    same = {"seq_len": 4096.0, "micro_s": 1.0, "optimizer_s": 0.5}
    lo, hi = pilot_budget.train_hours(cfg, SHAPE, gpu, "sft", 4096.0, same)
    assert lo == hi == pytest.approx(2 * (1000 * 0.5 + 8000 * 1.0) / 3600)
    sep = pilot_budget.train_hours(cfg, SHAPE, gpu, "separate", 4096.0, same)[0]
    assert sep == pytest.approx(3 * lo)
    short = {"seq_len": 2048.0, "micro_s": 1.0, "optimizer_s": 0.0}
    scale = (
        train_flops_per_token(SHAPE, "sft", 4096)
        * 4096
        / (train_flops_per_token(SHAPE, "sft", 2048) * 2048)
    )
    hours = pilot_budget.train_hours(cfg, SHAPE, gpu, "sft", 4096.0, short)[0]
    assert hours == pytest.approx(2 * 8000 * scale / 3600)
    assert 2.0 < scale < 2.4


def test_flop_estimate_gives_a_range() -> None:
    cfg = PilotConfig(name="p", data="")
    lo, hi = pilot_budget.train_hours(cfg, SHAPE, GPUS["5080"], "sft", 4096.0, None)
    tokens = 1000 * 8 * 4096
    flops = train_flops_per_token(SHAPE, "sft", 4096)
    assert lo == pytest.approx(2 * tokens * flops / (112e12 * 0.45) / 3600)
    assert hi == pytest.approx(lo * 0.45 / 0.25)


def test_auto_batch_follows_memory_and_cap() -> None:
    small = pilot_budget.auto_batch(SHAPE, GPUS["2080ti"], "full", 8192)
    big = pilot_budget.auto_batch(SHAPE, GPUS["5080"], "full", 8192)
    assert 1 < small < big <= 16
    assert pilot_budget.auto_batch(SHAPE, GPUS["5080"], "full", 1024) == 16
    assert pilot_budget.auto_batch(SHAPE, GPUS["5080"], "separate", 8192) == 1
    assert pilot_budget.auto_batch(SHAPE, GPUS["2080ti"], "sft", 8192) >= small


def test_eval_hours_use_the_limit_and_the_batch() -> None:
    full = PilotConfig(name="p", data="")
    limited = PilotConfig(name="p", data="", eval_limit=50)
    gpu = GPUS["5080"]
    lo, hi, batch = pilot_budget.eval_hours(full, SHAPE, gpu, "separate", 2000.0)
    assert batch == 1 and lo < hi
    assert pilot_budget.eval_hours(limited, SHAPE, gpu, "separate", 2000.0)[0] < lo / 10
    fixed = PilotConfig(name="p", data="", eval_batch_size=4)
    assert pilot_budget.eval_hours(fixed, SHAPE, gpu, "full", 2000.0)[2] == 4


def test_load_measured_reads_both_formats(tmp_path: Path) -> None:
    measure = {
        "seq_len": 4096,
        "arms": {"sft": {"phases": {"forward": 0.2, "backward": 0.5, "optimizer": 0.1}}},
    }
    (tmp_path / "m.json").write_text(json.dumps(measure))
    got = pilot_budget.load_measured(tmp_path / "m.json")
    assert got["sft"] == pytest.approx({"seq_len": 4096.0, "micro_s": 0.7, "optimizer_s": 0.1})
    memory_check = [
        {"arm": "moe", "seq_len": 4096, "error": "boom"},
        {"arm": "full", "seq_len": 4096, "train_summary": {"median_step_s": 3.0}},
        {"arm": "full", "seq_len": 8192, "train_summary": {"median_step_s": 7.0}},
    ]
    (tmp_path / "mc.json").write_text(json.dumps(memory_check))
    got = pilot_budget.load_measured(tmp_path / "mc.json")
    assert got == {"full": {"seq_len": 4096.0, "micro_s": 3.0, "optimizer_s": 0.0}}


def test_estimate_reads_the_plan_and_its_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {"input_ids": [1] * n, "complete": c} for n, c in ((101, True), (201, True), (9, False))
    ]
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    pilot.main(
        ["plan", "--name", "p", "--data", str(tmp_path / "data"), "--root", str(tmp_path)]
        + ["--eval-limit", "20"]
    )
    root = tmp_path / "p"
    capsys.readouterr()
    measured = tmp_path / "m.json"
    measure = {
        "seq_len": 150,
        "arms": {"sft": {"phases": {"forward": 0.1, "backward": 0.2, "optimizer": 0.05}}},
    }
    measured.write_text(json.dumps(measure))
    out = tmp_path / "budget.json"
    pilot_budget.main(
        ["estimate", "--root", str(root), "--out", str(out), "--measured", f"2080ti={measured}"]
    )
    text = capsys.readouterr().out
    assert text.startswith("# Pilot GPU time: an estimate, not a measurement")
    assert "150 tokens (mean of the data)" in text and "(first 20 problems each)" in text
    data = json.loads(out.read_text())
    assert data["estimate"] is True and data["row_tokens"] == 150
    gpu = data["gpus"]["5080"]
    assert gpu["verdict_hours"][0] < gpu["total_hours"][0]
    assert "The kill-rule verdict needs every arm but separate" in text
    arms = data["gpus"]["2080ti"]["arms"]
    assert arms["sft"]["train_source"] == "measured"
    assert arms["full"]["train_source"] == "flops"
    assert arms["sft"]["train_hours"][0] == pytest.approx(2 * (50 + 8000 * 0.3) / 3600, abs=0.01)
    with pytest.raises(SystemExit, match="GPU=PATH"):
        pilot_budget.main(["estimate", "--measured", str(measured)])


def test_phase_timer_sums_each_phase_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks: Iterator[float] = iter([0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 10.0, 10.5, 12.5, 13.0])
    monkeypatch.setattr(pilot_budget.time, "perf_counter", lambda: next(ticks))
    timer = pilot_budget.PhaseTimer("cpu")
    for name in ("forward", "backward", "forward", "backward", "optimizer", "idle"):
        timer(name)
    for name in ("forward", "backward", "optimizer", "idle"):
        timer(name)
    assert timer.steps[0] == {"forward": 2.0, "backward": 4.0, "optimizer": 1.0}
    assert timer.steps[1] == {"forward": 0.5, "backward": 2.0, "optimizer": 0.5}
    assert timer.medians(skip=1) == {"forward": 0.5, "backward": 2.0, "optimizer": 0.5}


def test_measure_leaves_a_busy_gpu_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pilot_budget, "other_gpu_processes", lambda: ["123, bitdraft, 9000 MiB"])
    with pytest.raises(SystemExit) as stop:
        pilot_budget.main(["measure", "--device", "cuda", "--out", str(tmp_path / "m.json")])
    assert stop.value.code == 3
    assert not (tmp_path / "m.json").exists()
