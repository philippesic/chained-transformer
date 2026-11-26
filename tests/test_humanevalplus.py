from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.humanevalplus import (
    BENCHMARK,
    INSTRUCTION_PREFIX,
    evaluator_env,
    extract_code,
    fenced_blocks,
    make_prompt,
    parse_results,
    sanitize_all,
    stub_response,
    write_samples,
)
from evals.types import Problem

TASK_PROMPT = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n'


def problem() -> Problem:
    return Problem(
        "humanevalplus",
        "HumanEval/999",
        make_prompt(TASK_PROMPT),
        {
            "entry_point": "add",
            "task_prompt": TASK_PROMPT,
            "canonical_solution": "    return a + b\n",
        },
    )


def test_prompt_mirrors_evalplus_chat_instruction() -> None:
    assert INSTRUCTION_PREFIX == (
        "Please provide a self-contained Python script that solves the following problem in a "
        "markdown code block:"
    )
    assert make_prompt("\n" + TASK_PROMPT) == (
        INSTRUCTION_PREFIX + "\n```python\n" + TASK_PROMPT.strip() + "\n```"
    )


def test_fenced_blocks_follow_tracker_fence_rules() -> None:
    text = "a\n```python\nx = 1\n```\nb\n````\ny = '```'\n````\n```bash\nls\n```\n```py\nz"
    assert fenced_blocks(text) == [
        ("python", "x = 1"),
        ("", "y = '```'"),
        ("bash", "ls"),
        ("py", "z"),
    ]


def test_extract_code_takes_last_python_block_after_thinking() -> None:
    response = (
        "<think>\n```python\ndef add(a, b):\n    return 0\n```\n</think>\n\n"
        "```python\ndef add(a, b):\n    return a - b\n```\n"
        "Better:\n```python\ndef add(a, b):\n    return a + b\n```\n```bash\npython x.py\n```\n"
    )
    e = extract_code(response)
    assert e.block_found and e.think_closed
    assert e.code == "def add(a, b):\n    return a + b"
    e = extract_code("<think>\n</think>\n\ndef add(a, b):\n    return a + b\n")
    assert not e.block_found and e.code.strip().startswith("def add")
    e = extract_code("<think>\n```python\ndef add(a, b):\n    return 1\n```\nhmm")
    assert e.block_found and not e.think_closed and "return 1" in e.code


def test_sanitize_keeps_entry_point_and_drops_test_code() -> None:
    code = (
        "import math\n\ndef helper(x):\n    return x\n\ndef unused():\n    return 0\n\n"
        "def add(a, b):\n    return helper(a) + b\n\nprint(add(1, 2))\nassert add(1, 1) == 2\n"
    )
    prose = "Here is the answer:\ndef add(a, b):\n    return a + b\nThat is all."
    (clean, status), (from_prose, status2) = sanitize_all([(code, "add"), (prose, "add")], 2, 30)
    assert status == status2 == "ok"
    assert "def add" in clean and "def helper" in clean and "import math" in clean
    assert "print(" not in clean and "assert" not in clean and "unused" not in clean
    assert from_prose == "def add(a, b):\n    return a + b"


def test_parse_results_reads_base_and_plus_separately(tmp_path: Path) -> None:
    data = {
        "date": "2026-11-26 00:00",
        "hash": "x",
        "eval": {
            "HumanEval/0": [
                {"task_id": "HumanEval/0", "base_status": "pass", "plus_status": "pass"}
            ],
            "HumanEval/1": [
                {"task_id": "HumanEval/1", "base_status": "pass", "plus_status": "fail"}
            ],
            "HumanEval/2": [
                {"task_id": "HumanEval/2", "base_status": "timeout", "plus_status": "fail"}
            ],
        },
    }
    path = tmp_path / "samples_eval_results.json"
    path.write_text(json.dumps(data))
    out = parse_results(json.loads(path.read_text()))
    assert out["HumanEval/0"]["passed"]
    assert out["HumanEval/1"]["base_passed"] and not out["HumanEval/1"]["plus_passed"]
    assert not out["HumanEval/1"]["passed"]
    assert out["HumanEval/2"]["base_status"] == "timeout" and not out["HumanEval/2"]["passed"]


def test_write_samples_fills_every_task(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    write_samples(path, ["HumanEval/0", "HumanEval/1"], {"HumanEval/1": "def f():\n    return 1"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {"task_id": "HumanEval/0", "solution": ""},
        {"task_id": "HumanEval/1", "solution": "def f():\n    return 1"},
    ]


def test_reference_and_stub_responses_wrap_code_after_empty_think() -> None:
    p = problem()
    reference = str(BENCHMARK.reference_response(p))
    assert reference == "<think>\n</think>\n\n```python\n" + TASK_PROMPT + "    return a + b\n```"
    assert extract_code(reference).code == TASK_PROMPT + "    return a + b"
    assert extract_code(stub_response(p)).code.endswith('"""\n    return None')


def test_evaluator_env_drops_unlisted_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "x")
    monkeypatch.setenv("HOME", "/home/test")
    env = evaluator_env()
    assert "SECRET_TOKEN" not in env and env["HOME"] == "/home/test"


def test_load_rejects_other_sources() -> None:
    with pytest.raises(ValueError, match="evalplus"):
        BENCHMARK.load(1, "/tmp/humaneval.jsonl")


@pytest.mark.network
def test_evalplus_scores_canonical_and_stub(tmp_path: Path) -> None:
    loaded = BENCHMARK.load(3, None)
    assert loaded.source["count"] == 164 and loaded.source["dataset_version"] == "v0.1.10"
    problems = loaded.problems
    responses = {p.task_id: str(BENCHMARK.reference_response(p)) for p in problems}
    responses[problems[1].task_id] = stub_response(problems[1])
    scores = BENCHMARK.score(problems, responses, tmp_path, 2)
    assert [s.passed for s in scores] == [True, False, True]
    assert scores[0].detail["base_passed"] and scores[0].detail["plus_passed"]
    assert scores[1].detail["base_status"] == "fail"
