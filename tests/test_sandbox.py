from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from evals.sandbox import (
    MEMORY,
    OK,
    OUTPUT_LIMIT,
    RUNTIME_ERROR,
    TIMEOUT,
    Job,
    run_many,
    run_python,
)

SPAWN_SLEEPER = (
    "import subprocess, sys\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "print(child.pid, flush=True)\n"
)


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _gone(pid: int, wait_s: float = 5.0) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_stdin_round_trip() -> None:
    result = run_python("import sys\nprint(sys.stdin.read().strip()[::-1])", "hello\n", 10)
    assert result.status == OK and result.ok
    assert result.returncode == 0
    assert result.stdout == "olleh\n"
    assert result.stderr == ""


def test_large_stdin_and_output_round_trip() -> None:
    data = "x" * (3 * 2**20)
    result = run_python("import sys\nsys.stdout.write(sys.stdin.read())", data, 20)
    assert result.status == OK
    assert result.stdout == data


def test_runtime_error_reports_traceback() -> None:
    result = run_python("print('before')\nraise ValueError('boom')", "", 10)
    assert result.status == RUNTIME_ERROR
    assert result.returncode == 1
    assert result.stdout == "before\n"
    assert "ValueError: boom" in result.stderr


def test_timeout_kills_process_group_including_children() -> None:
    start = time.monotonic()
    result = run_python(SPAWN_SLEEPER + "import time\ntime.sleep(60)\n", "", 1.5)
    assert result.status == TIMEOUT
    assert time.monotonic() - start < 10
    child = int(result.stdout.split()[0])
    assert _gone(child)


def test_children_left_behind_are_killed_after_normal_exit() -> None:
    start = time.monotonic()
    result = run_python(SPAWN_SLEEPER, "", 10)
    assert result.status == OK
    assert time.monotonic() - start < 5
    assert _gone(int(result.stdout.split()[0]))


def test_busy_loop_times_out() -> None:
    result = run_python("while True:\n    pass\n", "", 1)
    assert result.status == TIMEOUT
    assert result.seconds < 5


def test_memory_limit() -> None:
    result = run_python("block = bytearray(1024 * 2**20)\nprint('allocated')", "", 10, 256)
    assert result.status == MEMORY
    assert "allocated" not in result.stdout
    assert "MemoryError" in result.stderr


def test_output_cap_stops_runaway_printing() -> None:
    result = run_python("while True:\n    print('y' * 1000)\n", "", 10, output_limit=50_000)
    assert result.status == OUTPUT_LIMIT
    assert len(result.stdout) == 50_000
    assert result.seconds < 5


def test_stderr_is_truncated_to_its_tail() -> None:
    code = "import sys\nfor i in range(200000):\n    sys.stderr.write(f'line {i}\\n')\n"
    result = run_python(code, "", 20)
    assert result.status == OK
    assert len(result.stderr) < 100_000
    assert result.stderr.endswith("line 199999\n")


def test_file_size_limit() -> None:
    code = "with open('big.bin', 'wb') as f:\n    f.write(b'0' * (64 * 2**20))\n"
    result = run_python(code, "", 10)
    assert result.status == RUNTIME_ERROR
    assert "File too large" in result.stderr


def test_minimal_environment_and_fresh_workdir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    code = (
        "import json, os, sys\n"
        "print(json.dumps([sorted(os.environ), os.getcwd(), sys.flags.isolated]))"
    )
    first = run_python(code, "", 10)
    second = run_python(code, "", 10)
    assert first.status == OK and second.status == OK
    names, cwd, isolated = json.loads(first.stdout)
    assert isolated == 1
    assert "HF_TOKEN" not in names and "HTTPS_PROXY" not in names
    assert {"PATH", "PYTHONHASHSEED", "OMP_NUM_THREADS"} <= set(names)
    assert not any("proxy" in n.lower() or "token" in n.lower() for n in names)
    assert not os.path.exists(cwd)
    assert cwd != json.loads(second.stdout)[1]


def test_collect_reads_files_from_the_workdir(tmp_path: Path) -> None:
    code = "open('out.txt', 'w').write('result')\nprint('noise')"
    result = run_python(code, "", 10, collect=("out.txt", "missing.txt"), tmp_parent=tmp_path)
    assert result.status == OK
    assert result.files == {"out.txt": b"result"}
    assert list(tmp_path.iterdir()) == []


def test_run_many_preserves_order() -> None:
    jobs = [Job(code=f"import time\ntime.sleep(0.{3 - i})\nprint({i})") for i in range(4)]
    results = run_many(jobs, workers=4)
    assert [r.stdout.strip() for r in results] == ["0", "1", "2", "3"]
    assert all(r.status == OK for r in results)
