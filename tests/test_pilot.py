from __future__ import annotations

import json
import time
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
    root = _plan(tmp_path, eval_set="full")
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
    root = _plan(tmp_path, eval_set="full")
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
    root = _plan(tmp_path, eval_set="full")
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
    root = _plan(tmp_path, eval_set="full")
    with pytest.raises(SystemExit, match="needs trained checkpoints"):
        pilot.main(["eval-one", "--root", str(root), "--run", "separate-s0"])


def _report(root: Path, run_id: str, math: float, code: float, lcb: float | None = None) -> None:
    out = root / run_id / "eval"
    out.mkdir(parents=True, exist_ok=True)
    benches = {
        "math500": {"scores": {"accuracy": math}},
        "humanevalplus": {"scores": {"accuracy": code}},
        "livecodebench": {"scores": {"accuracy": code if lcb is None else lcb}},
    }
    (out / "report.json").write_text(json.dumps({"benchmarks": benches}))
    _done(root, "train", run_id)
    _done(root, "eval", run_id)


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


def _scores(root: Path, run_id: str, bench: str, passed: list[bool]) -> None:
    out = root / run_id / "eval" / bench
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"task_id": f"t{i}", "passed": p} for i, p in enumerate(passed)]
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _done(root: Path, kind: str, run_id: str) -> None:
    _, trains, evals = pilot.load_plan(root)
    runs = trains if kind == "train" else evals
    argv = next(r.argv for r in runs if r.id == run_id)
    pilot.write_status(root, run_id, kind, state=pilot.DONE, argv=argv)


def test_verdict_plan_samples_the_suite_and_leaves_out_separate(tmp_path: Path) -> None:
    root = _plan(tmp_path, seq_len="2048")
    plan = json.loads((root / "plan.json").read_text())
    assert [r["id"] for r in plan["train"]] == [
        f"{a}-s{s}" for s in (0, 1) for a in ("sft", "full", "lora", "moe")
    ]
    assert plan["config"]["eval_sample"] == {
        "math500": 383,
        "humanevalplus": 164,
        "livecodebench": 341,
    }
    argv = plan["eval"][0]["argv"]
    assert "--resume" in argv and argv[argv.index("--max-new-tokens") + 1] == "2048"
    assert argv[argv.index("--sample") + 1 :][:3] == [
        "math500=383",
        "humanevalplus=164",
        "livecodebench=341",
    ]


def test_a_done_step_reruns_when_its_command_changes(tmp_path: Path) -> None:
    root = _plan(tmp_path, eval_set="full")
    _done(root, "train", "sft-s0")
    _done(root, "eval", "sft-s0")
    _, trains, evals = pilot.load_plan(root)
    assert ("eval-one", "sft-s0") not in pilot.pending_steps(root, trains, evals, ["sft-s0"])
    evals[0].argv.append("--workers")
    assert pilot.pending_steps(root, trains, evals, ["sft-s0"]) == [("eval-one", "sft-s0")]


def test_resize_grows_the_sample_when_arms_disagree_and_never_shrinks(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    agree = [i % 2 == 0 for i in range(361)]
    split = [(i % 2 == 0) != (i % 3 == 0) for i in range(361)]
    for bench in ("math500", "livecodebench"):
        _scores(root, "full-s0", bench, agree)
        _scores(root, "sft-s0", bench, agree)
    assert pilot.resize(root) is None
    _scores(root, "full-s0", "math500", split)
    grown = pilot.resize(root)
    assert grown is not None and grown["math500"] == 500 and grown["livecodebench"] == 341
    plan = json.loads((root / "plan.json").read_text())
    assert plan["config"]["eval_sample"] == grown and len(plan["sizing"]) == 1
    assert plan["sizing"][0]["from"]["math500"] == 383
    _scores(root, "full-s0", "math500", agree)
    assert pilot.resize(root) is None
    assert pilot.load_plan(root)[0].eval_sample == grown


def test_run_reevaluates_earlier_arms_after_the_sample_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _plan(tmp_path, device="cpu")
    split = [i % 3 == 0 for i in range(361)]
    calls: list[tuple[str, str]] = []

    def fake_run(cmd: list[str], **_: Any) -> Any:
        kind, run_id = cmd[3], cmd[-1]
        calls.append((kind, run_id))
        if kind == "eval-one" and run_id in ("sft-s0", "full-s0"):
            passed = split if run_id == "full-s0" else [not p for p in split]
            _scores(root, run_id, "math500", passed)
        _done(root, kind.split("-")[0], run_id)
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    pilot.main(["run", "--root", str(root), "--only", "sft-s0", "full-s0"])
    assert calls == [
        ("train-one", "sft-s0"),
        ("eval-one", "sft-s0"),
        ("train-one", "full-s0"),
        ("eval-one", "full-s0"),
        ("eval-one", "sft-s0"),
        ("eval-one", "full-s0"),
    ]
    assert pilot.load_plan(root)[0].eval_sample == {
        "math500": 500,
        "humanevalplus": 164,
        "livecodebench": 341,
    }


def test_widen_adds_separate_and_keeps_trained_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _plan(tmp_path)
    _done(root, "train", "full-s0")
    pilot.main(["widen", "--root", str(root)])
    cfg, trains, evals = pilot.load_plan(root)
    assert cfg.eval_set == "full" and cfg.eval_sample is None and len(trains) == 14
    assert all("--sample" not in e.argv and "--resume" in e.argv for e in evals)
    pending = pilot.pending_steps(root, trains, evals, [])
    assert ("train-one", "full-s0") not in pending and ("train-one", "separate-s0-code") in pending
    assert "23 steps pending" in capsys.readouterr().out


def test_compare_reports_the_verdict_sample_power(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    _report(root, "full-s0", 0.5, 0.5)
    pilot.main(["compare", "--root", str(root)])
    sample = json.loads((root / "compare.json").read_text())["verdict_sample"]
    assert sample["problems"]["math500"] == 383 and sample["observed"] is False
    assert 0.79 < sample["power_for_a_2_point_gap"] < 0.85


def _table(margins: dict[int, float | None], lora: float = 0.0) -> dict[str, dict[int, Any]]:
    benches = ("math500", "humanevalplus")
    table: dict[str, dict[int, Any]] = {a: {} for a in ("full", "sft", "moe", "lora")}
    for seed, margin in margins.items():
        table["sft"][seed] = dict.fromkeys(benches, 40.0)
        table["moe"][seed] = dict.fromkeys(benches, 40.0)
        table["lora"][seed] = dict.fromkeys(benches, 40.0 + lora)
        if margin is not None:
            table["full"][seed] = dict.fromkeys(benches, 40.0 + margin)
    return table


@pytest.mark.parametrize(
    ("margins", "expected"),
    [
        ({0: 3.0, 1: 1.0}, "mixed"),
        ({0: 1.0, 1: 3.0}, "mixed"),
        ({0: 3.0, 1: 2.5, 2: 2.0}, "full beats sft"),
        ({0: 3.0, 1: 2.5, 2: 1.9}, "mixed"),
        ({0: 1.0, 1: 0.0, 2: -2.0}, "kill rule met"),
        ({0: 3.0, 1: 3.0, 2: None}, "incomplete"),
    ],
)
def test_verdict_when_seeds_disagree_and_with_a_third_seed(
    margins: dict[int, float | None], expected: str
) -> None:
    result = pilot.verdict(_table(margins), list(margins), ["math500", "humanevalplus"])
    assert result["kill_rule"].startswith(expected)
    assert set(result["margins"]) == {str(s) for s in margins}


def test_margins_of_exactly_two_and_one_point_count_despite_float_error() -> None:
    table = _table({0: 0.0, 1: 0.0})
    for seed in (0, 1):
        table["sft"][seed] = table["moe"][seed] = {"math500": 100 * 0.638}
        table["full"][seed] = {"math500": 100 * 0.658}
        table["lora"][seed] = {"math500": 100 * 0.648}
    assert 100 * 0.658 - 100 * 0.638 < 2.0
    result = pilot.verdict(table, [0, 1], ["math500"])
    assert result["kill_rule"].startswith("full beats sft")
    table["lora"][0] = {"math500": 100 * 0.668}
    assert pilot.verdict(table, [0, 1], ["math500"])["lora"].startswith("lora is within")


def test_lora_tie_is_judged_even_while_sft_is_missing() -> None:
    table = _table({0: 3.0, 1: 3.0}, lora=3.5)
    del table["sft"][1]
    result = pilot.verdict(table, [0, 1], ["math500", "humanevalplus"])
    assert result["kill_rule"].startswith("incomplete")
    assert result["lora"].startswith("lora is within")
    assert result["margins"]["1"] == {
        "full_minus_sft": None,
        "full_minus_moe": None,
        "lora_minus_full": 0.5,
    }


def test_primary_averages_the_planned_benchmarks_only(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    for seed in (0, 1):
        _report(root, f"sft-s{seed}", 0.40, 0.40)
        _report(root, f"moe-s{seed}", 0.41, 0.41)
        _report(root, f"lora-s{seed}", 0.40, 0.40)
        _report(root, f"full-s{seed}", 0.45, 0.45)
    path = root / "full-s0" / "eval" / "report.json"
    report = json.loads(path.read_text())
    del report["benchmarks"]["humanevalplus"]
    report["benchmarks"]["old"] = {"scores": {"accuracy": 0.0}}
    path.write_text(json.dumps(report))
    pilot.main(["compare", "--root", str(root)])
    result = json.loads((root / "compare.json").read_text())
    assert result["kill_rule"].startswith("incomplete")
    assert result["scores_points"]["full"]["0"] == {"math500": 45.0, "livecodebench": 45.0}


def test_failed_stale_and_retrained_evals_are_not_counted(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    for seed in (0, 1):
        for arm in ("sft", "moe", "lora", "full"):
            _report(root, f"{arm}-s{seed}", 0.45 if arm == "full" else 0.40, 0.40)
    cfg, trains, evals = pilot.load_plan(root)
    assert pilot.uncounted(root, trains, evals) == {}
    pilot.mark_failed(root, "full-s0", "eval", 2)
    assert pilot.read_status(root, "full-s0", "eval")["argv"]
    status = pilot.read_status(root, "full-s1", "eval")
    pilot.write_status(root, "full-s1", "eval", state=pilot.DONE, argv=status["argv"][:-2])
    stamp = pilot.trained_stamp(root, ["moe-s0"])
    pilot.write_status(root, "moe-s0", "eval", state=pilot.DONE, argv=evals[3].argv, trained=stamp)
    time.sleep(1.1)
    _done(root, "train", "moe-s0")
    assert pilot.uncounted(root, trains, evals) == {
        "full-s0": "eval failed",
        "full-s1": "evaluated under an earlier plan (sample or flags changed)",
        "moe-s0": "retrained after this eval",
    }
    pending = pilot.pending_steps(root, trains, evals, [])
    assert pending == [("eval-one", "full-s0"), ("eval-one", "moe-s0"), ("eval-one", "full-s1")]
    pilot.main(["compare", "--root", str(root)])
    assert json.loads((root / "compare.json").read_text())["kill_rule"].startswith("incomplete")


def test_a_retrained_arm_is_evaluated_again_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evals.run

    root = _plan(tmp_path)
    _done(root, "train", "sft-s0")
    seen: list[list[str]] = []

    def fake_eval(argv: list[str]) -> None:
        seen.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text("{}")

    monkeypatch.setattr(evals.run, "main", fake_eval)
    pilot.main(["eval-one", "--root", str(root), "--run", "sft-s0"])
    first = pilot.read_status(root, "sft-s0", "eval")["trained"]
    _, trains, evals_ = pilot.load_plan(root)
    assert ("eval-one", "sft-s0") not in pilot.pending_steps(root, trains, evals_, ["sft-s0"])
    time.sleep(1.1)
    _done(root, "train", "sft-s0")
    assert pilot.pending_steps(root, trains, evals_, ["sft-s0"]) == [("eval-one", "sft-s0")]
    pilot.main(["eval-one", "--root", str(root), "--run", "sft-s0"])
    assert pilot.read_status(root, "sft-s0", "eval")["trained"] != first
    assert len(list((root / "sft-s0").glob("eval-before-*/report.json"))) == 1
    assert len(seen) == 2


def test_eval_waits_for_training_under_the_current_plan(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    pilot.write_status(root, "sft-s0", "train", state=pilot.DONE, argv=["--old"])
    with pytest.raises(SystemExit, match="for the current plan first"):
        pilot.main(["eval-one", "--root", str(root), "--run", "sft-s0"])


def test_zero_discordance_sizes_and_compares_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _plan(tmp_path)
    same = [i % 2 == 0 for i in range(361)]
    for bench in ("math500", "humanevalplus", "livecodebench"):
        for run_id in ("full-s0", "sft-s0", "moe-s0"):
            _scores(root, run_id, bench, same[: 164 if bench == "humanevalplus" else 361])
    assert pilot.resize(root) is None
    pilot.main(["size", "--root", str(root)])
    pilot.main(["compare", "--root", str(root)])
    sample = json.loads((root / "compare.json").read_text())["verdict_sample"]
    assert sample["power_for_a_2_point_gap"] == 1.0 and sample["per_seed_se"] == 0.0


def test_compare_reports_the_odds_of_each_verdict(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    pilot.main(["compare", "--root", str(root)])
    odds = json.loads((root / "compare.json").read_text())["verdict_sample"]
    odds = odds["verdict_odds_by_true_gap"]
    assert odds["2"]["continue"] < 0.15 and odds["4"]["continue"] > 0.8
    for by_gap in odds.values():
        assert by_gap["continue"] + by_gap["mixed"] + by_gap["kill"] == pytest.approx(1, abs=0.02)


def test_train_extra_cannot_override_what_the_plan_sets(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--lora-rank"):
        _plan(tmp_path, train_extra="--lora-rank 8")
    base = ["plan", "--name", "p", "--data", "d", "--root", str(tmp_path)]
    with pytest.raises(SystemExit, match="--moe-top-k=1"):
        pilot.main([*base, "--train-extra=--moe-top-k=1"])
    root = _plan(tmp_path, lora_rank="8", moe_top_k="1", train_extra="--log-every 5")
    cfg, trains, _ = pilot.load_plan(root)
    assert cfg.lora_rank == 8 and cfg.moe_top_k == 1
    argv = next(r.argv for r in trains if r.arm == "lora")
    assert argv[argv.index("--lora-rank") + 1] == "8" and argv[-2:] == ["--log-every", "5"]


def test_same_plan_command_is_accepted_after_the_sample_grows(tmp_path: Path) -> None:
    root = _plan(tmp_path)
    cfg, _, _ = pilot.load_plan(root)
    cfg.eval_sample = {"math500": 500, "humanevalplus": 164, "livecodebench": 400}
    pilot.write_plan(root, cfg)
    _plan(tmp_path)
    assert pilot.load_plan(root)[0].eval_sample == cfg.eval_sample
    pilot.main(
        [
            "plan",
            "--name",
            "p",
            "--data",
            str(tmp_path / "data"),
            "--root",
            str(tmp_path),
            "--steps",
            "500",
            "--replace",
        ]
    )
    replaced = pilot.load_plan(root)[0]
    assert replaced.steps == 500 and replaced.eval_sample == cfg.eval_sample
    with pytest.raises(SystemExit, match="different config"):
        _plan(tmp_path, steps="700")


def test_status_reads_past_a_cut_off_training_log(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    log.write_text('{"step": 3, "loss": 1.5}\n{"step": 4, "lo')
    assert pilot.last_json_line(log) == {"step": 3, "loss": 1.5}


@pytest.mark.parametrize(
    ("margins", "every_seed", "seed_mean"),
    [
        ({0: 3.0, 1: 1.0}, "mixed", "full beats sft"),
        ({0: 2.0, 1: 0.5}, "mixed", "kill rule met"),
        ({0: 2.0, 1: 2.0}, "full beats sft", "full beats sft"),
        ({0: 1.0, 1: 1.0}, "kill rule met", "kill rule met"),
        ({0: 3.0, 1: None}, "incomplete", "incomplete"),
    ],
)
def test_seed_mean_rule_averages_each_margin_over_seeds(
    margins: dict[int, float | None], every_seed: str, seed_mean: str
) -> None:
    benches = ["math500", "humanevalplus"]
    table = _table(margins)
    assert pilot.verdict(table, [0, 1], benches)["kill_rule"].startswith(every_seed)
    result = pilot.verdict(table, [0, 1], benches, pilot.SEED_MEAN, 1.14)
    assert result["kill_rule"].startswith(seed_mean)
    assert result["rule"].startswith("seed-mean") and "1.33 points" in result["rule"]
    with pytest.raises(ValueError, match="per-seed SE"):
        pilot.verdict(table, [0, 1], benches, pilot.SEED_MEAN)


def test_plan_records_the_kill_rule_and_compare_applies_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _plan(tmp_path, kill_rule="seed-mean")
    assert json.loads((root / "plan.json").read_text())["config"]["kill_rule"] == "seed-mean"
    assert "kill rule: seed-mean" in capsys.readouterr().out
    for seed, full in ((0, 0.45), (1, 0.42)):
        _report(root, f"sft-s{seed}", 0.40, 0.40, 0.40)
        _report(root, f"moe-s{seed}", 0.41, 0.41, 0.41)
        _report(root, f"lora-s{seed}", 0.43, 0.43, 0.43)
        _report(root, f"full-s{seed}", full, full, full)
    pilot.main(["compare", "--root", str(root)])
    result = json.loads((root / "compare.json").read_text())
    assert result["kill_rule"].startswith("full beats sft and moe on the seed mean")
    odds = result["verdict_sample"]["verdict_odds_by_true_gap"]
    assert set(odds["2"]) == {"continue"} and odds["2"]["continue"] > 0.6
    with pytest.raises(SystemExit, match="different config"):
        _plan(tmp_path)
