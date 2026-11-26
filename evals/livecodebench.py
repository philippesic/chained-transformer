from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chained_transformer.modes import THINK_CLOSE, THINK_OPEN
from evals.sandbox import RunResult, parallel_map, run_python
from evals.segments import after_thinking
from evals.types import Loaded, Problem, Score

LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
DATASET_REPO = "livecodebench/code_generation_lite"
DATA_FILES = (
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
)
RELEASE_FILES: dict[str, tuple[str, ...]] = {
    **{f"release_v{i}": DATA_FILES[:i] for i in range(1, len(DATA_FILES) + 1)},
    "release_latest": DATA_FILES,
}
RELEASE_SIZES = {
    "release_v1": 400,
    "release_v2": 511,
    "release_v3": 612,
    "release_v4": 713,
    "release_v5": 880,
    "release_v6": 1055,
}
RELEASE = "release_v5"
START_DATE = "2024-10-01"
END_DATE = "2025-02-01"
DATE_FORMAT = "%Y-%m-%d"
TIMEOUT_S = 6
MEMORY_MB = 4096
RESULT_FILE = "lcb_result.jsonl"
TRUNCATE = 300
STDERR_IN_DETAIL = 1000

PROMPT_STYLE = "CodeQwenInstruct"
SYSTEM_MESSAGE = "You are a helpful assistant."
QUESTION_PREAMBLE = (
    "You will be given a question (problem specification) and will generate a correct Python "
    "program that matches the specification and passes all tests. You will NOT return anything "
    "except for the program."
)
FORMAT_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose "
    "your code within delimiters."
)
FORMAT_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT."
)

DRIVER = r"""
import ast
import builtins
import json
import os
import signal
import sys
import time
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

IMPORT_STRING = "\n".join([
    "from string import *",
    "from re import *",
    "from datetime import *",
    "from collections import *",
    "from heapq import *",
    "from bisect import *",
    "from copy import *",
    "from math import *",
    "from random import *",
    "from statistics import *",
    "from itertools import *",
    "from functools import *",
    "from operator import *",
    "from io import *",
    "from sys import *",
    "from json import *",
    "from builtins import *",
    "from typing import *",
    "import string",
    "import re",
    "import datetime",
    "import collections",
    "import heapq",
    "import bisect",
    "import copy",
    "import math",
    "import random",
    "import statistics",
    "import itertools",
    "import functools",
    "import operator",
    "import io",
    "import sys",
    "import json",
    "sys.setrecursionlimit(50000)",
    "",
])


class TimeoutException(Exception):
    pass


def on_alarm(signum, frame):
    raise TimeoutException


def truncate(value, length=TRUNCATE_LIMIT):
    text = value if isinstance(value, str) else str(value)
    if len(text) <= length:
        return text
    return text[: length // 2] + "...(truncated) ..." + text[-length // 2 :]


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __iter__(self):
        return iter(StringIO(self.inputs))

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code):
    try:
        tree = ast.parse(code)
        last = tree.body[-1]
        if isinstance(last, ast.If):
            if ast.unparse(last.test).strip() == "__name__ == '__main__'":
                code = ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last.body)
    except Exception:
        pass
    return code


def make_function(code):
    try:
        imports = []
        others = []
        tree = ast.parse(code)
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            else:
                others.append(stmt)
        wrapped = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=others,
            decorator_list=[],
            lineno=-1,
        )
        return IMPORT_STRING + "\n" + ast.unparse(imports) + "\n" + ast.unparse(wrapped)
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    lines = iter(inputs.split("\n"))
    stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", stdin)
    @patch("sys.stdin.readline", lambda *args: next(lines))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def inner(fn):
        try:
            return fn()
        except SystemExit:
            pass

    return inner(method)


def get_function(compiled, name):
    try:
        assert hasattr(compiled, name)
        return getattr(compiled, name)
    except Exception:
        return None


def compile_code(code, timeout):
    signal.alarm(timeout)
    try:
        module = ModuleType("tmp_sol", "")
        exec(code, module.__dict__)
        compiled = module.Solution() if "class Solution" in code else module
        assert compiled is not None
    finally:
        signal.alarm(0)
    return compiled


def to_decimals(line):
    try:
        return True, [Decimal(item) for item in line.split()]
    except Exception:
        return False, []


def stripped_lines(value):
    return [line.strip() for line in value.strip().split("\n")]


def failure(exc, code, message, inputs, expected):
    return {
        "error": repr(exc),
        "error_code": code,
        "error_message": message,
        "inputs": truncate(inputs),
        "expected": truncate(expected),
    }


def grade_call_based(code, all_inputs, all_outputs, fn_name, timeout, emit):
    compiled = compile_code(IMPORT_STRING + "\n\n" + code, timeout)
    if compiled is None:
        return None
    method = get_function(compiled, fn_name)
    if method is None:
        return None
    all_inputs = [[json.loads(line) for line in item.split("\n")] for item in all_inputs]
    all_outputs = [json.loads(item) for item in all_outputs]
    total = 0.0
    results = []
    for index, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        emit({"event": "start", "index": index})
        signal.alarm(timeout)
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total += time.time() - start
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if not bool(prediction == gt_out):
                results.append(-2)
                return results, {
                    "output": truncate(prediction),
                    "inputs": truncate(gt_inp),
                    "expected": truncate(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
            results.append(True)
        except Exception as exc:
            signal.alarm(0)
            if "timeoutexception" in repr(exc).lower():
                results.append(-3)
                return results, failure(exc, -3, "Time Limit Exceeded", gt_inp, gt_out)
            results.append(-4)
            return results, failure(exc, -4, "Runtime Error", gt_inp, gt_out)
        finally:
            signal.alarm(0)
    return results, {"execution time": total}


def grade_stdio(code, all_inputs, all_outputs, timeout, emit):
    compiled = compile_code(make_function(clean_if_name(code)), timeout)
    if compiled is None:
        return None
    method = get_function(compiled, "wrapped_function")
    if method is None:
        return None
    results = []
    total = 0.0
    for index, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        emit({"event": "start", "index": index})
        signal.alarm(timeout)
        with Capturing() as captured:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total += time.time() - start
                signal.alarm(0)
            except Exception as exc:
                signal.alarm(0)
                if "timeoutexception" in repr(exc).lower():
                    results.append(-3)
                    return results, failure(exc, -3, "Time Limit Exceeded", gt_inp, gt_out)
                results.append(-4)
                return results, failure(exc, -4, "Runtime Error", gt_inp, gt_out)
            finally:
                signal.alarm(0)
        prediction = captured[0]
        predicted_lines = stripped_lines(prediction)
        expected_lines = stripped_lines(gt_out)
        wrong = {
            "output": truncate(prediction),
            "inputs": truncate(gt_inp),
            "expected": truncate(gt_out),
            "error_code": -2,
        }
        if len(predicted_lines) != len(expected_lines):
            results.append(-2)
            wrong["error_message"] = "Wrong answer: mismatched output length"
            return results, wrong
        for line_index, (got, want) in enumerate(zip(predicted_lines, expected_lines)):
            wrong["error_message"] = (
                f"Wrong answer at output_line_idx={line_index}: "
                f"{truncate(got)} != {truncate(want)}"
            )
            if got == want:
                continue
            ok, got_decimals = to_decimals(got)
            if not ok:
                results.append(-2)
                return results, wrong
            ok, want_decimals = to_decimals(want)
            if not ok:
                results.append(-2)
                return results, wrong
            if got_decimals == want_decimals:
                continue
            results.append(-2)
            return results, wrong
        results.append(True)
    return results, {"execution time": total}


def reliability_guard():
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid",
        "fork", "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink",
        "fchmod", "fchown", "chmod", "chown", "chroot", "lchflags", "lchmod", "lchown",
        "getcwd", "chdir",
    ):
        setattr(os, name, None)
    import shutil
    import subprocess

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    builtins.help = None
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None


def main():
    sink = open(RESULT_FILE, "w", encoding="utf-8")

    def emit(record):
        sink.write(json.dumps(record) + "\n")
        sink.flush()

    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.set_int_max_str_digits(50000)
    timeout = payload["timeout"]
    signal.signal(signal.SIGALRM, on_alarm)
    reliability_guard()
    emit({"event": "ready"})
    try:
        signal.alarm(timeout)
        if payload["fn_name"] is None:
            results, metadata = grade_stdio(
                payload["code"], payload["inputs"], payload["outputs"], timeout, emit
            )
        else:
            results, metadata = grade_call_based(
                payload["code"],
                payload["inputs"],
                payload["outputs"],
                payload["fn_name"],
                timeout,
                emit,
            )
    except Exception as exc:
        results, metadata = [-4], {
            "error_code": -4,
            "error_message": f"Error during testing: {truncate(exc)}",
        }
    finally:
        signal.alarm(0)
    emit({"event": "end", "results": results, "metadata": metadata})
    sink.close()
    os._exit(0)


main()
"""


class _JsonOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def decode_tests(raw: str) -> list[dict[str, Any]]:
    try:
        decoded: Any = json.loads(raw)
    except ValueError:
        blob = zlib.decompress(base64.b64decode(raw.encode("utf-8")))
        text = _JsonOnlyUnpickler(io.BytesIO(blob)).load()
        if not isinstance(text, str):
            raise ValueError(f"private tests unpickled to {type(text).__name__}, not str") from None
        decoded = json.loads(text)
    if not isinstance(decoded, list):
        raise ValueError("test cases are not a list")
    return decoded


DRIVER_SOURCE = f"RESULT_FILE = {RESULT_FILE!r}\nTRUNCATE_LIMIT = {TRUNCATE}\n" + DRIVER


def build_prompt(question_content: str, starter_code: str) -> str:
    prompt = f"\n{QUESTION_PREAMBLE}\n\nQuestion: {question_content}\n\n"
    if starter_code:
        prompt += f"{FORMAT_WITH_STARTER_CODE}\n```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"{FORMAT_WITHOUT_STARTER_CODE}\n```python\n# YOUR CODE HERE\n```\n\n"
    return prompt


def extract_code(text: str) -> str:
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def answer_text(response: str) -> tuple[str, str]:
    if THINK_CLOSE in response:
        return after_thinking(response), "after_think"
    if THINK_OPEN in response:
        return "", "unclosed_think"
    return response, "no_think_tags"


def _parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _download(name: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename=name, revision=revision
    )
    return Path(path)


def _safe_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:120] or "task"


@dataclass(frozen=True)
class _Row:
    record: dict[str, Any]
    contest_date: datetime
    ref: dict[str, Any]


def _read_rows(name: str, path: Path) -> tuple[list[_Row], dict[str, Any]]:
    rows: list[_Row] = []
    digest = hashlib.sha256()
    offset = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            start = offset
            offset += len(line)
            if not line.strip():
                continue
            record = json.loads(line)
            record.pop("private_test_cases", None)
            ref = {
                "path": str(path.resolve()),
                "offset": start,
                "length": len(line),
                "sha256": hashlib.sha256(line).hexdigest(),
            }
            rows.append(_Row(record, _parse_date(record["contest_date"]), ref))
    info = {
        "name": name,
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": offset,
        "problems": len(rows),
    }
    return rows, info


def _read_tests(ref: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with open(ref["path"], "rb") as handle:
        handle.seek(ref["offset"])
        line = handle.read(ref["length"])
    if hashlib.sha256(line).hexdigest() != ref["sha256"]:
        raise ValueError(f"dataset row changed since load: {ref['path']}@{ref['offset']}")
    record = json.loads(line)
    return decode_tests(record["public_test_cases"]), decode_tests(record["private_test_cases"])


def _parse_events(run: RunResult) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in run.files.get(RESULT_FILE, b"").decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


@dataclass
class LiveCodeBench:
    name: str = "livecodebench"
    release: str = RELEASE
    start_date: str = START_DATE
    end_date: str = END_DATE
    timeout_s: int = TIMEOUT_S
    memory_mb: int = MEMORY_MB
    revision: str | None = None

    def load(self, limit: int | None, source: str | None) -> Loaded:
        files = RELEASE_FILES[self.release]
        if source is None:
            paths = [_download(name, self.revision) for name in files]
        else:
            paths = [Path(source) / name for name in files]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"missing LiveCodeBench files: {missing}")
        rows: list[_Row] = []
        infos: list[dict[str, Any]] = []
        for name, path in zip(files, paths, strict=True):
            file_rows, info = _read_rows(name, path)
            rows.extend(file_rows)
            infos.append(info)
        start = datetime.strptime(self.start_date, DATE_FORMAT)
        end = datetime.strptime(self.end_date, DATE_FORMAT)
        window = [row for row in rows if start <= row.contest_date < end]
        window.sort(key=lambda row: (row.contest_date, str(row.record["question_id"])))
        chosen = window if limit is None else window[:limit]
        problems = [self._problem(row) for row in chosen]
        meta = {
            "dataset": DATASET_REPO,
            "origin": "huggingface_hub" if source is None else str(Path(source).resolve()),
            "revision": self.revision,
            "release": self.release,
            "files": infos,
            "window": {
                "start": self.start_date,
                "end": self.end_date,
                "start_inclusive": True,
                "end_inclusive": False,
            },
            "problems_in_release": len(rows),
            "release_size_in_lcb_readme": RELEASE_SIZES.get(self.release),
            "problems_in_window": len(window),
            "problems_returned": len(problems),
            "limit": limit,
            "order": ["contest_date", "question_id"],
            "prompt_style": PROMPT_STYLE,
            "lcb_runner_commit": LCB_COMMIT,
            "timeout_s": self.timeout_s,
            "memory_mb": self.memory_mb,
        }
        return Loaded(problems=problems, source=meta)

    def _problem(self, row: _Row) -> Problem:
        record = row.record
        metadata = json.loads(record.get("metadata") or "{}")
        fn_name = metadata.get("func_name")
        starter_code = record.get("starter_code") or ""
        return Problem(
            benchmark=self.name,
            task_id=str(record["question_id"]),
            prompt=build_prompt(record["question_content"], starter_code),
            meta={
                "system": SYSTEM_MESSAGE,
                "question_id": str(record["question_id"]),
                "question_title": record.get("question_title"),
                "platform": record.get("platform"),
                "contest_id": record.get("contest_id"),
                "contest_date": row.contest_date.isoformat(),
                "difficulty": record.get("difficulty"),
                "starter_code": starter_code,
                "fn_name": fn_name,
                "testtype": "stdin" if fn_name is None else "functional",
                "tests_ref": row.ref,
            },
        )

    def reference_response(self, problem: Problem) -> str | None:
        return None

    def score(
        self,
        problems: Sequence[Problem],
        responses: Mapping[str, str],
        workdir: Path,
        workers: int,
    ) -> list[Score]:
        root = workdir / self.name
        (root / "code").mkdir(parents=True, exist_ok=True)

        def one(problem: Problem) -> Score:
            return self.score_one(problem, responses.get(problem.task_id), root)

        return parallel_map(one, problems, workers)

    def score_one(self, problem: Problem, response: str | None, root: Path) -> Score:
        fn_name = problem.meta.get("fn_name")
        detail: dict[str, Any] = {
            "testtype": "stdin" if fn_name is None else "functional",
            "code_found": False,
        }
        if response is None:
            detail["error"] = "missing response"
            return Score(problem.task_id, False, detail)
        text, answer_source = answer_text(response)
        code = extract_code(text)
        detail["answer_source"] = answer_source
        detail["code_found"] = bool(code)
        if not code:
            detail["error"] = "no fenced code block after thinking"
            return Score(problem.task_id, False, detail)
        code_dir = root / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / f"{_safe_name(problem.task_id)}.py").write_text(code, encoding="utf-8")
        public, private = _read_tests(problem.meta["tests_ref"])
        tests = public + private
        num_tests = len(tests)
        payload = json.dumps(
            {
                "code": code,
                "fn_name": fn_name,
                "inputs": [t["input"] for t in tests],
                "outputs": [t["output"] for t in tests],
                "timeout": self.timeout_s,
            }
        ).encode("utf-8")
        del public, private, tests
        run = run_python(
            DRIVER_SOURCE,
            payload,
            (self.timeout_s + 1) * num_tests + 5,
            self.memory_mb,
            collect=(RESULT_FILE,),
            tmp_parent=root / "sandbox",
        )
        return Score(problem.task_id, *verdict(run, num_tests, detail))


def verdict(run: RunResult, num_tests: int, detail: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    events = _parse_events(run)
    started = [e["index"] for e in events if e.get("event") == "start"]
    end = next((e for e in reversed(events) if e.get("event") == "end"), None)
    detail.update(
        num_tests=num_tests,
        sandbox_status=run.status,
        returncode=run.returncode,
        seconds=round(run.seconds, 3),
    )
    if end is None:
        detail.update(
            error_code=-1,
            error_message="sandbox ended without a verdict",
            tests_passed=max(len(started) - 1, 0),
            first_failed_test=started[-1] if started else None,
            stderr_tail=run.stderr[-STDERR_IN_DETAIL:],
        )
        return False, detail
    results = list(end.get("results") or [])
    metadata = end.get("metadata") or {}
    passed = num_tests > 0 and len(results) == num_tests and all(r is True for r in results)
    failed = [i for i, r in enumerate(results) if r is not True]
    detail.update(
        tests_passed=sum(1 for r in results if r is True),
        first_failed_test=failed[0] if failed else None,
    )
    for key in ("error_code", "error_message", "output", "expected", "inputs"):
        if key in metadata:
            detail[key] = metadata[key]
    if "error" in metadata:
        detail["exception"] = metadata["error"]
    if "execution time" in metadata:
        detail["execution_time"] = round(float(metadata["execution time"]), 4)
    if not passed and "error_code" not in detail:
        detail["error_code"] = -4
    return passed, detail


BENCHMARK = LiveCodeBench()
