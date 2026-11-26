from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset
from transformers import Qwen3Config

from chained_transformer.checkpoint import chained_config, meta_counts
from chained_transformer.surgery import SurgeryConfig, Variant
from tools import pilot, pilot_report

BASE = Qwen3Config(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=128,
    tie_word_embeddings=True,
)


def _plan(tmp_path: Path) -> Path:
    pilot.main(
        [
            "plan",
            "--name",
            "p",
            "--data",
            str(tmp_path / "data"),
            "--root",
            str(tmp_path),
            "--benchmarks",
            "math500",
            "humanevalplus",
            "--eval-set",
            "full",
        ]
    )
    return tmp_path / "p"


def _train(root: Path, run_id: str, arm: str, peak: float) -> None:
    out = root / run_id
    out.mkdir(parents=True, exist_ok=True)
    summary = {"arm": arm, "device": "cuda", "peak_reserved_gib": peak}
    (out / "summary.json").write_text(json.dumps(summary))
    config = (
        BASE
        if arm in ("sft", "separate")
        else chained_config(BASE, SurgeryConfig(cast(Variant, arm)))
    )
    config.save_pretrained(str(out / "checkpoint"))
    _, trains, _ = pilot.load_plan(root)
    argv = next(r.argv for r in trains if r.id == run_id)
    pilot.write_status(root, run_id, "train", state=pilot.DONE, argv=argv)


def _bench(
    accuracy: float, base: float | None, tokens: int, seconds: float, words: float, rate: float
) -> dict[str, Any]:
    scores: dict[str, Any] = {"accuracy": accuracy}
    if base is not None:
        scores.update(pass_at_1_base=base, pass_at_1_plus=accuracy)
    return {
        "scores": scores,
        "leakage": {"n_closed": 10, "answer_words_mean": words, "answer_rate_per_1k_pooled": rate},
        "modes": {
            "n": 10,
            "mean_new_tokens": tokens / 10,
            "mean_switches": 1.0,
            "mean_reprefilled_tokens": 0.0,
        },
        "perf": {"seconds": seconds, "new_tokens": tokens, "batch_size": 8, "peak_alloc_gib": 3.0},
    }


def _report(root: Path, run_id: str, math: float, code: float, batch: int = 8) -> None:
    out = root / run_id / "eval"
    out.mkdir(parents=True, exist_ok=True)
    benches = {
        "math500": _bench(math, None, 1000, 10.0, 100.0, 4.0),
        "humanevalplus": _bench(code, code + 0.05, 3000, 20.0, 300.0, 8.0),
    }
    for b in benches.values():
        b["perf"]["batch_size"] = batch
        if run_id.startswith("separate"):
            b["modes"]["mean_reprefilled_tokens"] = 250.0
    report = {"smoke": False, "environment": {"device": "cuda"}, "benchmarks": benches}
    (out / "report.json").write_text(json.dumps(report))
    _, _, evals = pilot.load_plan(root)
    argv = next(e.argv for e in evals if e.id == run_id)
    pilot.write_status(root, run_id, "eval", state=pilot.DONE, argv=argv)
    rows = [
        ("a", [[0, 5], [1, 3], [0, 2], [2, 4]], True),
        ("b", [[0, 9]], False),
        ("c", [[0, 4], [2, 6]], True),
    ]
    traces = out / "math500"
    traces.mkdir(exist_ok=True)
    (traces / "responses.jsonl").write_text(
        "".join(json.dumps({"task_id": t, "mode_runs": r}) + "\n" for t, r, _ in rows)
    )
    (traces / "scores.jsonl").write_text(
        "".join(json.dumps({"task_id": t, "passed": p}) + "\n" for t, _, p in rows)
    )


def _pilot(
    tmp_path: Path, full: tuple[float, float], lora: tuple[float, float], skip: str = ""
) -> Path:
    root = _plan(tmp_path)
    for seed in (0, 1):
        for arm in pilot.ARMS:
            for mode in pilot.SEPARATE_MODES if arm == "separate" else (None,):
                run_id = f"{arm}-s{seed}" + (f"-{mode}" if mode else "")
                _train(root, run_id, arm, 5.0 + seed + (mode == "code"))
        scores = {
            "sft": 0.40,
            "moe": 0.41,
            "separate": 0.39,
            "full": full[seed],
            "lora": lora[seed],
        }
        for arm, score in scores.items():
            run_id = f"{arm}-s{seed}"
            if run_id != skip:
                _report(root, run_id, score, score, 1 if arm == "separate" else 8)
    return root


@pytest.mark.parametrize(
    ("full", "lora", "verdict", "lora_note"),
    [
        ((0.45, 0.45), (0.445, 0.445), "full beats sft and moe", "lora is within"),
        ((0.45, 0.45), (0.40, 0.40), "full beats sft and moe", "lora does not tie"),
        ((0.41, 0.42), (0.41, 0.42), "kill rule met", "lora is within"),
        ((0.45, 0.40), (0.45, 0.40), "mixed", "lora is within"),
    ],
)
def test_verdict_heads_the_report(
    tmp_path: Path,
    full: tuple[float, float],
    lora: tuple[float, float],
    verdict: str,
    lora_note: str,
) -> None:
    root = _pilot(tmp_path, full, lora)
    data, text = pilot_report.build(root, 0)
    assert data["kill_rule"].startswith(verdict)
    assert data["lora"].startswith(lora_note)
    lines = text.splitlines()
    assert lines[2] == f"Verdict: {data['kill_rule']}."
    assert lines[3] == f"LoRA: {data['lora']}."
    assert lines[4] == "Rule: every-seed: 2.0 points on every seed."
    assert data["margins"]["0"]["full_minus_sft"] == round(100 * full[0] - 40.0, 2)
    assert data["margins_se"]["0"]["full_minus_sft"] == 0.0
    assert f"| 0 | {100 * full[0] - 40.0:.2f} ± 0.00 |" in text


def test_paired_standard_error_pools_benchmarks() -> None:
    a = {"m": {"1": True, "2": False, "3": True, "4": True}, "h": {"x": True, "y": False}}
    b = {"m": {"1": False, "2": False, "3": True, "4": False}, "h": {"x": True, "y": True}}
    expected = 100 * ((1 / 3) / 4 + 0.5 / 2) ** 0.5 / 2
    assert pilot_report.paired_se(a, b) == pytest.approx(expected)
    assert pilot_report.paired_se(a, {"other": {}}) is None
    assert pilot_report.paired_se({"m": {"1": True}}, {"m": {"1": False}}) is None


def test_missing_report_makes_the_verdict_incomplete(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.45), (0.45, 0.45), skip="moe-s1")
    data, text = pilot_report.build(root, 0)
    assert data["kill_rule"].startswith("incomplete")
    assert data["lora"].startswith("lora is within")
    assert "- No eval report yet for: moe-s1." in text
    assert "| moe | 41.0 (41.0 / n/a) |" in text


def test_main_table_scores_memory_and_parameters(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.47), (0.44, 0.44))
    data, text = pilot_report.build(root, 0)
    full = data["per_run"]["full"]
    assert full["0"]["math500"] == pytest.approx(45.0)
    assert full["1"]["he_base"] == pytest.approx(52.0)
    assert full["0"]["mean"] == pytest.approx(45.0)
    assert full["0"]["tokens_per_s"] == pytest.approx(4000 / 30)
    assert full["0"]["leakage"] == pytest.approx((4.0 * 1000 + 8.0 * 3000) / 4000)
    assert full["1"]["train_peak_gib"] == 6.0
    assert data["per_run"]["separate"]["0"]["train_peak_gib"] == 6.0
    assert "| full | 46.0 (45.0 / 47.0) | 51.0 (50.0 / 52.0) | 46.0 (45.0 / 47.0) |" in text

    base = meta_counts(BASE)
    mlp = base["mlp"]
    counts = {arm: data["per_run"][arm]["0"] for arm in pilot.ARMS}
    assert counts["sft"]["total_params"] == counts["sft"]["active_params"] == base["total"]
    assert counts["full"]["total_params"] == base["total"] + 2 * mlp
    assert counts["full"]["active_params"] == base["total"]
    adapters = counts["lora"]["total_params"] - base["total"]
    assert counts["lora"]["active_params"] == base["total"] + adapters / 3
    moe = counts["moe"]["total_params"]
    assert counts["moe"]["active_params"] == moe - mlp
    assert counts["separate"]["total_params"] == 3 * base["total"]
    assert counts["separate"]["active_params"] == base["total"]


def test_switching_serving_and_notes(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.45), (0.45, 0.45))
    row = {
        "input_ids": list(range(8)),
        "mode_ids": [0, 0, 0, 1, 1, 0, 2, 2],
        "loss_mask": [0, 1, 1, 1, 1, 1, 1, 1],
    }
    Dataset.from_list([row] * 4).save_to_disk(str(tmp_path / "data"))
    data, text = pilot_report.build(root, 10)
    sw = data["switching"]
    assert sw["data"]["responses"] == 4
    assert sw["full"]["responses"] == 6
    assert sw["full"]["switches"]["mean"] == 1.33
    rates = sw["full"]["by_outcome"]["pass_rate_by_switches"]
    assert rates == {
        "0": {"n": 2, "pass_rate": 0.0},
        "1": {"n": 2, "pass_rate": 1.0},
        "3-5": {"n": 2, "pass_rate": 1.0},
    }
    assert "| full | 0.0 (2) | 100.0 (2) | n/a | 100.0 (2) | n/a | n/a |" in text
    assert "| separate | 250.0 (250.0 / 250.0) | 200 (200 / 200) |" in text
    assert any(n.startswith("separate decodes at batch 1") for n in data["notes"])
    assert "sft" not in text.split("## Table 5, serving")[1]


def test_cli_writes_markdown_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _pilot(tmp_path, (0.45, 0.45), (0.45, 0.45))
    capsys.readouterr()
    pilot_report.main(["--root", str(root), "--data-limit", "0"])
    out = capsys.readouterr().out
    assert out == (root / "results.md").read_text()
    assert json.loads((root / "results.json").read_text())["pilot"] == "p"
    assert "## Table 1, main results" in out and "## Table 5, serving" in out


def test_failed_eval_is_shown_but_not_counted(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.45), (0.45, 0.45))
    pilot.mark_failed(root, "full-s1", "eval", 1)
    data, text = pilot_report.build(root, 0)
    assert data["kill_rule"].startswith("incomplete")
    assert data["not_counted"] == {"full-s1": "eval failed"}
    assert "Not counted in the verdict, though Table 1 shows their reports: full-s1" in text
    assert data["per_run"]["full"]["1"]["mean"] == pytest.approx(45.0)


def test_mean_needs_every_planned_benchmark_and_ignores_others(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.45), (0.45, 0.45))
    path = root / "full-s0" / "eval" / "report.json"
    report = json.loads(path.read_text())
    report["benchmarks"]["livecodebench"] = _bench(0.99, None, 10, 1.0, 1.0, 1.0)
    path.write_text(json.dumps(report))
    assert pilot_report.build(root, 0)[0]["per_run"]["full"]["0"]["mean"] == pytest.approx(45.0)
    del report["benchmarks"]["humanevalplus"]
    path.write_text(json.dumps(report))
    data, _ = pilot_report.build(root, 0)
    assert data["per_run"]["full"]["0"]["mean"] is None
    assert data["kill_rule"].startswith("incomplete")


def test_report_reads_a_cut_off_response_line_and_skips_empty_perf(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.45), (0.45, 0.45))
    responses = root / "full-s0" / "eval" / "math500" / "responses.jsonl"
    responses.write_text(responses.read_text() + '{"task_id": "d", "mode_ru')
    path = root / "full-s1" / "eval" / "report.json"
    report = json.loads(path.read_text())
    report["benchmarks"]["math500"]["perf"] = {"batch_size": 1, "new_tokens": 0, "seconds": 0.0}
    path.write_text(json.dumps(report))
    data, _ = pilot_report.build(root, 0)
    assert data["switching"]["full"]["responses"] == 6
    assert data["per_run"]["full"]["1"]["batch_size"] == 8.0
    assert data["per_run"]["full"]["1"]["tokens_per_s"] == pytest.approx(3000 / 20)


def test_report_applies_the_seed_mean_rule_when_the_plan_asks(tmp_path: Path) -> None:
    root = _pilot(tmp_path, (0.45, 0.42), (0.45, 0.42))
    assert pilot_report.build(root, 0)[0]["kill_rule"].startswith("mixed")
    plan = json.loads((root / "plan.json").read_text())
    plan["config"]["kill_rule"] = "seed-mean"
    (root / "plan.json").write_text(json.dumps(plan))
    data, text = pilot_report.build(root, 0)
    assert data["kill_rule"].startswith("full beats sft and moe on the seed mean")
    assert text.splitlines()[4].startswith("Rule: seed-mean: seed mean of each margin")
