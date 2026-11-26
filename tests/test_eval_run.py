from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from chained_transformer.modes import THINK_CLOSE, THINK_OPEN
from evals.run import main, read_jsonl
from tools.tiny_checkpoint import build, toy_tokenizer

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
            assert sum(n for _, n in row["mode_runs"]) == row["new_tokens"]
            assert len(row["mode_runs"]) == row["switches"] + 1
        assert len(read_jsonl(tmp_path / name / "scores.jsonl")) == 2


def test_think_tags_survive_decoding_as_qwen3_keeps_them() -> None:
    tok = toy_tokenizer()
    text = f"{THINK_OPEN}\nso{THINK_CLOSE}\n\n4"
    ids = tok(text + tok.eos_token, add_special_tokens=False).input_ids
    assert tok.decode(ids, skip_special_tokens=True) == text


def test_closed_thinking_is_counted_from_the_decoded_text(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    report = _run(
        data, tmp_path, "--model", str(models["plain1"]), extra=("--max-new-tokens", "24")
    )
    answered = []
    for name in ("math500", "livecodebench"):
        rows = read_jsonl(tmp_path / name / "responses.jsonl")
        shut = [r for r in rows if THINK_CLOSE in r["text"]]
        bench = report["benchmarks"][name]
        assert bench["modes"]["think_closed_rate"] == len(shut) / len(rows)
        assert bench["leakage"]["n_closed"] == len(shut)
        answered += [r["tokens_by_mode"]["answer"] > 0 for r in shut]
    assert any(answered)


def test_auto_batch_size_fills_the_batch_on_cpu(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    report = _run(data, tmp_path, "--model", str(models["full"]), extra=("--batch-size", "auto"))
    assert report["generation"]["batch_size"] == 0
    assert report["benchmarks"]["math500"]["perf"]["batch_size"] == 2


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


def test_sample_is_a_fixed_subset_that_grows_by_prefix(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    one = _run(data, tmp_path / "a", "--model", str(models["full"]), extra=("--sample", "1"))
    _run(data, tmp_path / "b", "--model", str(models["lora"]), extra=("--sample", "1"))
    two = _run(data, tmp_path / "c", "--model", str(models["full"]), extra=("--sample", "2"))
    assert one["sample"]["sizes"] == {"math500": 1, "livecodebench": 1}
    for name in ("math500", "livecodebench"):
        assert one["benchmarks"][name]["n_scored"] == 1
        assert one["benchmarks"][name]["n_available"] == 2

        def ids(out: str, bench: str = name) -> set[str]:
            return {r["task_id"] for r in read_jsonl(tmp_path / out / bench / "responses.jsonl")}

        assert ids("a") == ids("b") and ids("a") < ids("c")
    assert two["benchmarks"]["math500"]["n_scored"] == 2


def test_resume_generates_only_what_is_missing(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    _run(data, tmp_path, "--model", str(models["full"]), extra=("--sample", "1"))
    path = tmp_path / "math500" / "responses.jsonl"
    first = read_jsonl(path)
    path.write_text(path.read_text() + '{"task_id": "cut off')
    report = _run(
        data, tmp_path, "--model", str(models["full"]), extra=("--sample", "2", "--resume")
    )
    bench = report["benchmarks"]["math500"]
    assert bench["perf"]["resumed"] == 1 and bench["n_scored"] == 2
    rows = read_jsonl(path)
    assert rows[0] == first[0] and len(rows) == 2
    assert bench["perf"]["new_tokens"] == sum(r["new_tokens"] for r in rows)
    again = _run(
        data, tmp_path, "--model", str(models["full"]), extra=("--sample", "2", "--resume")
    )
    perf = again["benchmarks"]["math500"]["perf"]
    assert perf["resumed"] == 2 and perf["batch_size"] == bench["perf"]["batch_size"]
    assert perf["new_tokens"] == bench["perf"]["new_tokens"]
    with pytest.raises(SystemExit, match="different generation settings"):
        _run(
            data,
            tmp_path,
            "--model",
            str(models["full"]),
            extra=("--resume", "--max-new-tokens", "10"),
        )
    with pytest.raises(SystemExit, match="different model"):
        _run(data, tmp_path, "--model", str(models["lora"]), extra=("--resume",))


def test_resume_refuses_responses_it_cannot_check(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    model = tmp_path / "model"
    shutil.copytree(models["full"], model)
    _run(data, tmp_path / "out", "--model", str(model), extra=("--sample", "1"))
    weights = next(model.glob("*.safetensors"))
    raw = bytearray(weights.read_bytes())
    raw[-8:] = bytes(8)
    weights.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="retrained since"):
        _run(data, tmp_path / "out", "--model", str(model), extra=("--resume",))
    (tmp_path / "out" / "report.json").unlink()
    with pytest.raises(SystemExit, match="left no report.json"):
        _run(data, tmp_path / "out", "--model", str(model), extra=("--resume",))


def test_score_only_keeps_speed_and_sample_and_reads_a_cut_off_line(
    tmp_path: Path, data: dict[str, Path], models: dict[str, Path]
) -> None:
    first = _run(data, tmp_path, "--model", str(models["full"]), extra=("--sample", "1"))
    path = tmp_path / "math500" / "responses.jsonl"
    path.write_text(path.read_text() + '{"task_id": "cut off')
    again = _run(data, tmp_path, "--score-only")
    assert again["sample"] == first["sample"]
    for name in ("math500", "livecodebench"):
        perf = again["benchmarks"][name]["perf"]
        assert perf["new_tokens"] == first["benchmarks"][name]["perf"]["new_tokens"]
        assert again["benchmarks"][name]["n_problems"] == 1
    path.unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        _run(data, tmp_path, "--score-only")


def test_no_thinking_responses_are_scored_as_answers(tmp_path: Path, data: dict[str, Path]) -> None:
    out = tmp_path / "math500"
    out.mkdir()
    rows = [
        {"task_id": "t/1", "text": "So 2+2 is \\boxed{4}.", "new_tokens": 5},
        {"task_id": "t/2", "text": "It is \\boxed{8}.", "new_tokens": 4},
    ]
    full = [{**r, "finished": True, "tokens_by_mode": {}, "switches": 0} for r in rows]
    (out / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in full))

    def score(thinking: bool) -> dict[str, Any]:
        report = {"generation": {"enable_thinking": thinking}}
        (tmp_path / "report.json").write_text(json.dumps(report))
        main(
            [
                "--score-only",
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
        result: dict[str, Any] = json.loads((tmp_path / "report.json").read_text())
        return result["benchmarks"]["math500"]

    off = score(False)
    assert off["scores"]["accuracy"] == 0.5 and off["modes"]["think_closed_rate"] == 1.0
    assert off["leakage"]["n_closed"] == 2
    assert score(True)["scores"]["accuracy"] == 0.0


def test_a_bad_sample_size_is_a_usage_error(tmp_path: Path, data: dict[str, Path]) -> None:
    with pytest.raises(SystemExit, match="not 'math500=ten'"):
        _run(data, tmp_path, "--oracle", extra=("--sample", "math500=ten"))
