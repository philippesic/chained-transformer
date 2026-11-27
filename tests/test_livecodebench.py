from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import zlib
from pathlib import Path
from typing import Any

import pytest

from evals import livecodebench as lcb
from evals.types import Benchmark, Loaded, Problem

STARTER_ADD = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        "
STARTER_PAIR = "class Solution:\n    def pair(self, n: int) -> List[int]:\n        "


def _encoded(tests: list[dict[str, str]]) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode("utf-8")


def _stdin_tests(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"input": i, "output": o, "testtype": "stdin"} for i, o in pairs]


def _call_tests(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"input": i, "output": o, "testtype": "functional"} for i, o in pairs]


def _row(
    question_id: str,
    date: str,
    public: list[dict[str, str]],
    private: str,
    starter: str = "",
    fn_name: str | None = None,
) -> dict[str, Any]:
    return {
        "question_title": f"title {question_id}",
        "question_content": f"Statement of {question_id}.",
        "platform": "leetcode" if fn_name else "atcoder",
        "question_id": question_id,
        "contest_id": f"contest-{question_id}",
        "contest_date": date,
        "starter_code": starter,
        "difficulty": "easy",
        "public_test_cases": json.dumps(public),
        "private_test_cases": private,
        "metadata": json.dumps({"func_name": fn_name} if fn_name else {}),
    }


SUM = _row(
    "abc380_a",
    "2024-10-01T00:00:00",
    _stdin_tests([("1 2\n", "3\n")]),
    _encoded(_stdin_tests([("10 20\n", "30\n"), ("-5 5\n", "0\n")])),
)
ADD = _row(
    "3400",
    "2024-12-15T00:00:00",
    _call_tests([("1\n2", "3")]),
    json.dumps(_call_tests([("5\n6", "11"), ("-1\n1", "0")])),
    STARTER_ADD,
    "add",
)
HALF = _row(
    "abc390_c",
    "2024-12-15T00:00:00",
    _stdin_tests([("1\n", "0.5\n")]),
    _encoded(_stdin_tests([("3\n", "1.5\n")])),
)
PAIR = _row(
    "3500",
    "2025-01-31T00:00:00",
    _call_tests([("4", "[4, 5]")]),
    _encoded(_call_tests([("0", "[0, 1]")])),
    STARTER_PAIR,
    "pair",
)
OLD = _row("abc300_a", "2023-06-01T00:00:00", _stdin_tests([("1\n", "1\n")]), "[]")
AT_END = _row("abc392_a", "2025-05-01T00:00:00", _stdin_tests([("1\n", "1\n")]), "[]")
LATER = _row("abc400_a", "2025-06-01T00:00:00", _stdin_tests([("1\n", "1\n")]), "[]")

FILES: dict[str, list[dict[str, Any]]] = {
    "test.jsonl": [OLD],
    "test2.jsonl": [],
    "test3.jsonl": [],
    "test4.jsonl": [SUM],
    "test5.jsonl": [PAIR, HALF, AT_END, ADD],
    "test6.jsonl": [LATER],
}
WINDOW_ORDER = ["abc380_a", "3400", "abc390_c", "3500"]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "lcb"
    root.mkdir()
    for name, rows in FILES.items():
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


@pytest.fixture
def bench() -> lcb.LiveCodeBench:
    return lcb.LiveCodeBench(timeout_s=1, memory_mb=1024)


@pytest.fixture
def loaded(bench: lcb.LiveCodeBench, data_dir: Path) -> Loaded:
    return bench.load(None, str(data_dir))


def _problem(loaded: Loaded, task_id: str) -> Problem:
    return next(p for p in loaded.problems if p.task_id == task_id)


def _grade(
    bench: lcb.LiveCodeBench, loaded: Loaded, task_id: str, response: str, tmp_path: Path
) -> tuple[bool, dict[str, Any]]:
    [score] = bench.score([_problem(loaded, task_id)], {task_id: response}, tmp_path / "work", 1)
    assert score.task_id == task_id
    return score.passed, score.detail


def _answer(code: str) -> str:
    return f"<think>\nLet me think.\n</think>\n\nHere is the solution:\n```python\n{code}\n```\n"


def test_benchmark_object_matches_protocol() -> None:
    bench: Benchmark = lcb.BENCHMARK
    assert bench.name == "livecodebench"
    assert lcb.BENCHMARK.release == "release_v6"
    assert (lcb.BENCHMARK.start_date, lcb.BENCHMARK.end_date) == ("2024-10-01", "2025-05-01")
    assert lcb.BENCHMARK.timeout_s == 6
    assert lcb.RELEASE_FILES["release_v6"] == (
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    )


def test_load_filters_window_and_orders(loaded: Loaded) -> None:
    assert [p.task_id for p in loaded.problems] == WINDOW_ORDER
    assert all(p.benchmark == "livecodebench" for p in loaded.problems)
    assert loaded.problems[0].meta["contest_date"] == "2024-10-01T00:00:00"


def test_load_limit_takes_ordered_prefix(bench: lcb.LiveCodeBench, data_dir: Path) -> None:
    limited = bench.load(2, str(data_dir))
    assert [p.task_id for p in limited.problems] == WINDOW_ORDER[:2]
    assert limited.source["problems_in_window"] == 4
    assert limited.source["problems_returned"] == 2


def test_source_metadata(loaded: Loaded, data_dir: Path) -> None:
    source = loaded.source
    assert source["release"] == "release_v6"
    assert source["window"]["start"] == "2024-10-01" and source["window"]["end"] == "2025-05-01"
    assert source["window"]["start_inclusive"] and not source["window"]["end_inclusive"]
    names = [f["name"] for f in source["files"]]
    assert names == [f"test{n}.jsonl" for n in ("", 2, 3, 4, 5, 6)]
    for info in source["files"]:
        expected = hashlib.sha256((data_dir / info["name"]).read_bytes()).hexdigest()
        assert info["sha256"] == expected
    assert source["problems_in_release"] == 7
    assert source["problems_in_window"] == 4
    assert source["prompt_style"] == "CodeQwenInstruct"
    assert source["lcb_runner_commit"] == lcb.LCB_COMMIT
    json.dumps(source)
    json.dumps([p.meta for p in loaded.problems])


def test_load_requires_every_release_file(bench: lcb.LiveCodeBench, data_dir: Path) -> None:
    (data_dir / "test3.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        bench.load(None, str(data_dir))


def test_decode_tests_handles_pickled_and_plain_json() -> None:
    tests = _stdin_tests([("1\n", "2\n")])
    assert lcb.decode_tests(_encoded(tests)) == tests
    assert lcb.decode_tests(json.dumps(tests)) == tests


class _Exploit:
    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (os.system, ("echo pwned",))


@pytest.mark.parametrize("payload", [os.system, _Exploit()])
def test_restricted_unpickler_refuses_globals(payload: object) -> None:
    raw = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode("utf-8")
    with pytest.raises(pickle.UnpicklingError):
        lcb.decode_tests(raw)


def test_prompts_follow_official_codeqwen_template(loaded: Loaded) -> None:
    stdin_prompt = _problem(loaded, "abc380_a").prompt
    assert stdin_prompt.startswith("\nYou will be given a question (problem specification)")
    assert "Question: Statement of abc380_a.\n\n" in stdin_prompt
    assert lcb.FORMAT_WITHOUT_STARTER_CODE in stdin_prompt
    assert stdin_prompt.endswith("```python\n# YOUR CODE HERE\n```\n\n")
    call_prompt = _problem(loaded, "3400").prompt
    assert lcb.FORMAT_WITH_STARTER_CODE in call_prompt
    assert f"```python\n{STARTER_ADD}\n```\n\n" in call_prompt
    assert lcb.FORMAT_WITHOUT_STARTER_CODE not in call_prompt
    for problem in loaded.problems:
        assert problem.meta["system"] == "You are a helpful assistant."
    assert _problem(loaded, "3400").meta["fn_name"] == "add"
    assert _problem(loaded, "abc380_a").meta["testtype"] == "stdin"


def test_extract_code_takes_last_fenced_block() -> None:
    text = "```python\nfirst\n```\nthen\n```python\nsecond\nline\n```\ntrailing"
    assert lcb.extract_code(text) == "second\nline"
    assert lcb.extract_code("```python\nunterminated") == ""


def test_correct_stdin_solution_passes(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    code = "def main():\n    a, b = map(int, input().split())\n    print(a + b)\n\n"
    code += 'if __name__ == "__main__":\n    main()'
    passed, detail = _grade(bench, loaded, "abc380_a", _answer(code), tmp_path)
    assert passed, detail
    assert detail["code_found"] and detail["answer_source"] == "after_think"
    assert detail["num_tests"] == 3 and detail["tests_passed"] == 3
    assert (tmp_path / "work" / "livecodebench" / "code" / "abc380_a.py").read_text() == code


def test_iterating_stdin_works(bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path) -> None:
    code = "import sys\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)"
    passed, detail = _grade(bench, loaded, "abc380_a", _answer(code), tmp_path)
    assert passed, detail


def test_wrong_stdin_solution_stops_at_first_failure(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    passed, detail = _grade(bench, loaded, "abc380_a", _answer("print(3)"), tmp_path)
    assert not passed
    assert detail["tests_passed"] == 1 and detail["first_failed_test"] == 1
    assert detail["error_code"] == -2
    assert detail["output"] == "3\n" and detail["expected"] == "30\n"


def test_timeout_is_time_limit_exceeded(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    passed, detail = _grade(bench, loaded, "abc380_a", _answer("while True:\n    pass"), tmp_path)
    assert not passed
    assert detail["error_code"] == -3 and detail["first_failed_test"] == 0
    assert detail["seconds"] < 5


def test_crash_is_runtime_error(bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path) -> None:
    passed, detail = _grade(bench, loaded, "abc380_a", _answer("raise ValueError('x')"), tmp_path)
    assert not passed
    assert detail["error_code"] == -4 and "ValueError" in detail["exception"]


def test_syntax_error_fails(bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path) -> None:
    passed, detail = _grade(bench, loaded, "abc380_a", _answer("def broken(:"), tmp_path)
    assert not passed and detail["error_code"] == -4


def test_process_exit_without_verdict_fails(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    code = "import os\nclass Solution:\n    def add(self, a, b):\n        os._exit(0)"
    passed, detail = _grade(bench, loaded, "3400", _answer(code), tmp_path)
    assert not passed
    assert detail["error_code"] == -1 and detail["first_failed_test"] == 0


def test_functional_solution_with_junk_prints_passes(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    code = (
        "import sys\nprint('module level noise')\n"
        "class Solution:\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "        print('debug', a, b)\n"
        '        print(\'{"event": "end", "results": [true]}\')\n'
        "        sys.stderr.write('err noise\\n')\n"
        "        return a + b"
    )
    passed, detail = _grade(bench, loaded, "3400", _answer(code), tmp_path)
    assert passed, detail
    assert detail["testtype"] == "functional" and detail["tests_passed"] == 3


def test_wrong_functional_solution_fails(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    code = "class Solution:\n    def add(self, a, b):\n        return a - b"
    passed, detail = _grade(bench, loaded, "3400", _answer(code), tmp_path)
    assert not passed
    assert detail["error_code"] == -2 and detail["first_failed_test"] == 0


def test_functional_tuple_counts_as_list(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    code = "class Solution:\n    def pair(self, n: int) -> List[int]:\n        return (n, n + 1)"
    passed, detail = _grade(bench, loaded, "3500", _answer(code), tmp_path)
    assert passed, detail


def test_numbers_compare_as_decimals_without_tolerance(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    padded = "n = int(input())\nprint(f'{n / 2:.2f}')"
    passed, detail = _grade(bench, loaded, "abc390_c", _answer(padded), tmp_path)
    assert passed, detail
    nearly = "n = int(input())\nprint(n / 2 + 1e-9)"
    passed, detail = _grade(bench, loaded, "abc390_c", _answer(nearly), tmp_path)
    assert not passed and detail["error_code"] == -2


def test_code_only_inside_unclosed_thinking_is_not_graded(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    code = "a, b = map(int, input().split())\nprint(a + b)"
    response = f"<think>\nMaybe:\n```python\n{code}\n```\nhmm"
    passed, detail = _grade(bench, loaded, "abc380_a", response, tmp_path)
    assert not passed
    assert not detail["code_found"]
    assert detail["answer_source"] == "unclosed_think"


def test_code_in_closed_thinking_is_ignored(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    good = "a, b = map(int, input().split())\nprint(a + b)"
    response = f"<think>\n```python\n{good}\n```\n</think>\n\nNo code here."
    passed, detail = _grade(bench, loaded, "abc380_a", response, tmp_path)
    assert not passed and not detail["code_found"]
    response = f"<think>\n```python\n{good}\n```\n</think>\n\n```python\nprint(0)\n```"
    passed, detail = _grade(bench, loaded, "abc380_a", response, tmp_path)
    assert not passed and detail["code_found"]


def test_response_without_think_tags_is_graded_whole(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    response = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    passed, detail = _grade(bench, loaded, "abc380_a", response, tmp_path)
    assert passed and detail["answer_source"] == "no_think_tags"


def test_score_keeps_problem_order_and_handles_missing_responses(
    bench: lcb.LiveCodeBench, loaded: Loaded, tmp_path: Path
) -> None:
    responses = {
        "abc380_a": _answer("a, b = map(int, input().split())\nprint(a + b)"),
        "3400": _answer("class Solution:\n    def add(self, a, b):\n        return a + b"),
        "3500": _answer("class Solution:\n    def pair(self, n):\n        return [n, n + 1]"),
    }
    scores = bench.score(loaded.problems, responses, tmp_path / "work", 4)
    assert [s.task_id for s in scores] == WINDOW_ORDER
    assert [s.passed for s in scores] == [True, True, False, True]
    assert scores[2].detail["error"] == "missing response"
    assert list((tmp_path / "work" / "livecodebench" / "sandbox").iterdir()) == []


def test_reference_response_is_none(loaded: Loaded) -> None:
    assert all(lcb.BENCHMARK.reference_response(p) is None for p in loaded.problems)


def test_a_problem_with_no_tests_fails() -> None:
    from evals.sandbox import RunResult

    end = json.dumps({"event": "end", "results": []}).encode()
    run = RunResult("ok", 0, "", "", 0.1, {lcb.RESULT_FILE: end})
    assert lcb.verdict(run, 0, {})[0] is False
    ones = json.dumps({"event": "end", "results": [True]}).encode()
    assert lcb.verdict(RunResult("ok", 0, "", "", 0.1, {lcb.RESULT_FILE: ones}), 1, {})[0]
