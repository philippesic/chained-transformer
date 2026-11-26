from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.run import main, read_jsonl
from tools.tiny_checkpoint import build

MATH_ROWS = [
    {"problem": "What is $2+2$?", "answer": "4", "unique_id": "t/1", "subject": "A", "level": 1},
    {"problem": "What is $3+4$?", "answer": "7", "unique_id": "t/2", "subject": "A", "level": 1},
]


def _lcb_row(question_id: str) -> dict[str, Any]:
    tests = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
    return {
        "question_title": question_id,
        "question_content": "Add two numbers.",
        "platform": "atcoder",
        "question_id": question_id,
        "contest_id": "c",
        "contest_date": "2024-11-01T00:00:00",
        "starter_code": "",
        "difficulty": "easy",
        "public_test_cases": json.dumps(tests),
        "private_test_cases": json.dumps(tests),
        "metadata": "{}",
    }


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("data")
    math = root / "math.jsonl"
    math.write_text("".join(json.dumps(r) + "\n" for r in MATH_ROWS))
    lcb = root / "lcb"
    lcb.mkdir()
    for i, name in enumerate(["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"]):
        (lcb / name).write_text(json.dumps(_lcb_row(f"abc{i}")) + "\n")
    (lcb / "test5.jsonl").write_text("")
    return {"math": math, "lcb": lcb}


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("models")
    out = {"full": build(root / "full", "full"), "lora": build(root / "lora", "lora")}
    for seed in range(3):
        out[f"plain{seed}"] = build(root / f"plain{seed}", None, seed)
    return out


def _run(
    data: dict[str, Path], out: Path, *who: str, extra: tuple[str, ...] = ()
) -> dict[str, Any]:
    main(
        [
            *who,
            "--benchmarks",
            "math500",
            "livecodebench",
            "--source",
            f"math500={data['math']}",
            f"livecodebench={data['lcb']}",
            "--limit",
            "2",
            "--greedy",
            "--max-new-tokens",
            "12",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--workers",
            "2",
            "--out",
            str(out),
            *extra,
        ]
    )
    report: dict[str, Any] = json.loads((out / "report.json").read_text())
    return report


@pytest.mark.parametrize("variant", ["full", "lora"])
def test_model_run_writes_consistent_report(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path], variant: str
) -> None:
    report = _run(data, tmp_path, "--model", str(models[variant]))
    assert report["kind"] == "model" and report["smoke"]
    assert report["model"]["models"][0]["variant"] == variant
    assert report["generation"]["do_sample"] is False
    for name in ("math500", "livecodebench"):
        bench = report["benchmarks"][name]
        assert bench["n_scored"] == 2 and bench["scores"]["n"] == 2
        assert bench["modes"]["n"] == 2 and bench["leakage"]["n"] == 2
        for row in read_jsonl(tmp_path / name / "responses.jsonl"):
            assert sum(row["tokens_by_mode"].values()) == row["new_tokens"] <= 12
        assert len(read_jsonl(tmp_path / name / "scores.jsonl")) == 2


def test_score_only_reuses_responses(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    first = _run(data, tmp_path, "--model", str(models["full"]))
    before = read_jsonl(tmp_path / "math500" / "scores.jsonl")
    second = _run(data, tmp_path, "--score-only")
    assert second["model"] == first["model"]
    assert second["generation"] == first["generation"]
    assert read_jsonl(tmp_path / "math500" / "scores.jsonl") == before


def test_oracle_math_run_scores_everything(tmp_path: Path, data: dict[str, Path]) -> None:
    main(
        [
            "--oracle",
            "--benchmarks",
            "math500",
            "--source",
            f"math500={data['math']}",
            "--out",
            str(tmp_path),
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["kind"] == "oracle"
    assert report["benchmarks"]["math500"]["scores"]["accuracy"] == 1.0


def test_plain_and_separate_arms_run(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    plain = _run(data, tmp_path / "plain", "--model", str(models["plain0"]))
    assert plain["model"]["models"][0]["variant"] is None
    separate = _run(
        data,
        tmp_path / "separate",
        "--separate",
        *(str(models[f"plain{i}"]) for i in range(3)),
        extra=("--batch-size", "1"),
    )
    assert separate["model"]["arm"] == "separate" and len(separate["model"]["models"]) == 3
    assert separate["benchmarks"]["math500"]["modes"]["n"] == 2


def test_separate_arm_needs_greedy(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="greedy"):
        main(
            [
                "--separate",
                *(str(models[f"plain{i}"]) for i in range(3)),
                "--benchmarks",
                "math500",
                "--source",
                f"math500={data['math']}",
                "--device",
                "cpu",
                "--out",
                str(tmp_path),
            ]
        )
