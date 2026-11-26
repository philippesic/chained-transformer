from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from evals import math500
from evals.math500 import BENCHMARK, boxed_spans, extract, last_boxed, make_prompt, run_guarded
from evals.types import Problem


def problem(task_id: str, answer: str) -> Problem:
    return Problem("math500", task_id, make_prompt("q"), {"answer": answer})


def test_last_boxed_handles_nesting_multiple_and_truncation() -> None:
    assert last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("first \\boxed{3} then \\boxed{5}.") == "5"
    assert last_boxed("\\boxed{\\{1, 2\\}}") == "\\{1, 2\\}"
    assert last_boxed("\\boxed {7}") == "7"
    assert last_boxed("\\boxed{4} and \\boxed{x") == "4"
    assert last_boxed("no box here") is None
    assert boxed_spans("\\boxed{\\boxed{4}} \\boxed{\\left( 1, \\sqrt{2} \\right)}") == [
        "\\boxed{4}",
        "\\left( 1, \\sqrt{2} \\right)",
    ]


def test_extract_ignores_boxes_inside_thinking() -> None:
    ex = extract("<think>\nmaybe \\boxed{3}\n</think>\n\nSo \\boxed{4}.")
    assert ex.think_closed and ex.answer_box == "4" and ex.lenient_box == "4"
    ex = extract("<think>\nmaybe \\boxed{3}\n</think>\n\nThe answer is 4.")
    assert ex.answer_box is None and "4" in ex.answer_text and "3" not in ex.answer_text
    ex = extract("<think>\nso \\boxed{3} then \\boxed{5}")
    assert not ex.think_closed and ex.answer_box is None and ex.answer_text == ""
    assert ex.lenient_box == "5"


def test_prompt_is_qwen_math_instruction() -> None:
    assert make_prompt("What is $1+1$?") == (
        "What is $1+1$?\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def test_score_uses_math_verify_equivalence(tmp_path: Path) -> None:
    cases = [
        ("\\frac{1}{2}", "\\boxed{0.5}", True),
        ("\\frac{14}{3}", "\\boxed{\\dfrac{14}{3}}", True),
        ("(3, \\frac{\\pi}{2})", "\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", True),
        ("\\text{(C)}", "\\boxed{C}", True),
        ("3\\sqrt{13}", "\\boxed{\\sqrt{117}}", True),
        ("5", "\\boxed{x = 5}", True),
        ("10\\%", "\\boxed{10}", True),
        ("\\$32,348", "\\boxed{32348}", True),
        ("x \\in [-2,7]", "\\boxed{[-2, 7]}", True),
        ("6 - 5i", "\\boxed{6-5i}", True),
        ("2", "\\boxed{3}", False),
        ("\\text{even}", "\\boxed{\\text{odd}}", False),
        ("7", "\\boxed{7} wait no \\boxed{8}", False),
        ("9", "The final answer is $9$.", True),
    ]
    problems = [problem(f"t{i}", gold) for i, (gold, _, _) in enumerate(cases)]
    responses = {
        f"t{i}": f"<think>\nok\n</think>\n\n{text}" for i, (_, text, _) in enumerate(cases)
    }
    scores = BENCHMARK.score(problems, responses, tmp_path, 2)
    assert [s.passed for s in scores] == [want for _, _, want in cases]
    assert all(s.detail["think_closed"] and s.detail["error"] is None for s in scores)
    assert scores[0].detail["extracted"] == "0.5" and scores[0].detail["boxed"]
    assert scores[-1].detail["extracted"] == "9" and not scores[-1].detail["boxed"]


def test_unclosed_thinking_fails_primary_but_counts_lenient(tmp_path: Path) -> None:
    problems = [problem("a", "4"), problem("b", "4"), problem("c", "4")]
    responses = {
        "a": "<think>\nthe answer is \\boxed{4}",
        "b": "<think>\nhmm \\boxed{4}\n</think>\n\nI am not sure.",
    }
    a, b, c = BENCHMARK.score(problems, responses, tmp_path, 2)
    assert not a.passed and a.detail["lenient_passed"] and not a.detail["think_closed"]
    assert a.detail["extracted"] is None and a.detail["lenient_extracted"] == "4"
    assert not b.passed and not b.detail["lenient_passed"] and not b.detail["boxed"]
    assert not c.passed and c.detail["missing_response"]


def test_reference_response_round_trips(tmp_path: Path) -> None:
    golds = ["\\frac{\\sqrt{3}}{2}", "\\begin{pmatrix} 1 \\\\ -2 \\end{pmatrix}", "\\text{Evelyn}"]
    problems = [problem(f"r{i}", g) for i, g in enumerate(golds)]
    responses = {p.task_id: str(BENCHMARK.reference_response(p)) for p in problems}
    assert responses["r0"].startswith("<think>\n</think>\n\n")
    assert all(s.passed for s in BENCHMARK.score(problems, responses, tmp_path, 1))


def test_local_jsonl_source_normalises_fields(tmp_path: Path) -> None:
    rows = [
        {
            "problem": "p0",
            "answer": "1",
            "subject": "Algebra",
            "level": 2,
            "unique_id": "test/algebra/1.json",
        },
        {
            "question": "p1",
            "solution": "so \\boxed{\\frac{1}{2}}",
            "type": "Geometry",
            "level": "Level 5",
        },
    ]
    path = tmp_path / "math.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded = BENCHMARK.load(None, str(path))
    p0, p1 = loaded.problems
    assert p0.task_id == "test/algebra/1.json" and p0.meta == {
        "answer": "1",
        "subject": "Algebra",
        "level": 2,
    }
    assert p1.task_id == "test/1" and p1.meta["answer"] == "\\frac{1}{2}"
    assert p1.meta["level"] == 5 and p1.meta["subject"] == "Geometry"
    assert p0.prompt == make_prompt("p0")
    assert loaded.source["kind"] == "jsonl" and loaded.source["count"] == 2
    assert loaded.source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(BENCHMARK.load(1, str(path)).problems) == 1


def test_github_source_checks_pointer_and_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"problem": "p", "answer": "2", "unique_id": "test/x/1.json"}\n'
    sha = hashlib.sha256(content).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(content)}\n"
    served = {math500.POINTER_URL: pointer.encode(), math500.MEDIA_URL: content}
    monkeypatch.setattr(math500, "POINTER_SHA256", hashlib.sha256(pointer.encode()).hexdigest())
    monkeypatch.setattr(math500, "CONTENT_SHA256", sha)
    monkeypatch.setattr(math500, "CONTENT_SIZE", len(content))
    monkeypatch.setattr(math500, "_fetch", lambda url, dest: dest.write_bytes(served[url]))
    path = math500.download_github(tmp_path)
    assert path == math500.github_path(tmp_path) and path.read_bytes() == content
    assert not any((tmp_path / "downloads").iterdir())
    served[math500.MEDIA_URL] = b"tampered\n"
    path.unlink()
    with pytest.raises(ValueError, match="does not match"):
        math500.download_github(tmp_path)
    assert not path.exists()


def _job(x: int) -> int:
    if x == 1:
        time.sleep(60)
    if x == 2:
        os._exit(3)
    if x == 3:
        raise ValueError("boom")
    return 10 * x


def test_run_guarded_survives_hangs_crashes_and_errors() -> None:
    start = time.monotonic()
    out = run_guarded(_job, [0, 1, 2, 3, 4, 5], workers=2, timeout_s=2.0)
    assert out == [0, None, None, None, 40, 50]
    assert time.monotonic() - start < 30


@pytest.mark.network
def test_github_source_loads_pinned_math500() -> None:
    loaded = BENCHMARK.load(None, "github")
    assert loaded.source["count"] == 500 and len(loaded.problems) == 500
    assert loaded.source["sha256"] == math500.CONTENT_SHA256
    assert loaded.problems[0].prompt.endswith(math500.INSTRUCTION)
