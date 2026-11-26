from __future__ import annotations

import contextlib
import math
import os
import resource
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

OK = "ok"
TIMEOUT = "timeout"
RUNTIME_ERROR = "runtime_error"
MEMORY = "memory"
OUTPUT_LIMIT = "output_limit"

MIB = 2**20
STDOUT_LIMIT = 4 * MIB
STDERR_TAIL = 64 * 1024
FILE_LIMIT = 16 * MIB
COLLECT_LIMIT = 4 * MIB
CHUNK = 64 * 1024
POLL_S = 0.05
SCRIPT_NAME = "main.py"
MEMORY_MARKERS = ("MemoryError", "Cannot allocate memory", "std::bad_alloc", "out of memory")

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class RunResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass(frozen=True)
class Job:
    code: str
    stdin: str | bytes = ""
    timeout_s: float = 10.0
    memory_mb: int = 2048


def sandbox_env(workdir: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LANG": "C.UTF-8",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
    }


def _bounded(kind: int, value: int) -> tuple[int, int]:
    _, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    return value, value


def _limits(memory_mb: int, cpu_s: int) -> Callable[[], None]:
    memory = _bounded(resource.RLIMIT_AS, memory_mb * MIB)
    cpu_soft, cpu_hard = _bounded(resource.RLIMIT_CPU, cpu_s + 1)
    cpu = (min(cpu_s, cpu_soft), cpu_hard)
    fsize = _bounded(resource.RLIMIT_FSIZE, FILE_LIMIT)

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, memory)
        resource.setrlimit(resource.RLIMIT_CPU, cpu)
        resource.setrlimit(resource.RLIMIT_FSIZE, fsize)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _exited(pid: int) -> bool:
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _drain(fd: int, out: bytearray) -> bool:
    try:
        chunk = os.read(fd, CHUNK)
    except BlockingIOError:
        return False
    out.extend(chunk)
    return bool(chunk)


def _read(fd: int, out: bytearray) -> bool:
    try:
        chunk = os.read(fd, CHUNK)
    except BlockingIOError:
        return True
    out.extend(chunk)
    return bool(chunk)


def _trim(err: bytearray) -> None:
    if len(err) > 2 * STDERR_TAIL:
        del err[:-STDERR_TAIL]


def _pump(
    proc: subprocess.Popen[bytes], data: bytes, deadline: float, limit: int
) -> tuple[bytes, bytes, str | None]:
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    out = bytearray()
    err = bytearray()
    view = memoryview(data)
    offset = 0
    reason: str | None = None
    sel = selectors.DefaultSelector()
    in_fd, out_fd, err_fd = proc.stdin.fileno(), proc.stdout.fileno(), proc.stderr.fileno()
    for fd in (in_fd, out_fd, err_fd):
        os.set_blocking(fd, False)
    if data:
        sel.register(in_fd, selectors.EVENT_WRITE)
    else:
        proc.stdin.close()
    sel.register(out_fd, selectors.EVENT_READ)
    sel.register(err_fd, selectors.EVENT_READ)
    try:
        while len(sel.get_map()) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = TIMEOUT
                break
            if _exited(proc.pid):
                for fd, target in ((out_fd, out), (err_fd, err)):
                    budget = limit + CHUNK if fd == out_fd else 2 * STDERR_TAIL
                    while fd in sel.get_map() and len(target) < budget and _drain(fd, target):
                        pass
                break
            for key, _ in sel.select(min(remaining, POLL_S)):
                fd = int(key.fd)
                if fd == in_fd:
                    try:
                        offset += os.write(fd, view[offset : offset + CHUNK])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        offset = len(data)
                    if offset >= len(data):
                        sel.unregister(fd)
                        proc.stdin.close()
                    continue
                if not _read(fd, out if fd == out_fd else err):
                    sel.unregister(fd)
                _trim(err)
            if len(out) > limit:
                break
    finally:
        sel.close()
    _trim(err)
    if len(out) > limit:
        reason = OUTPUT_LIMIT
    return bytes(out[:limit]), bytes(err), reason


def _wait_leader(pid: int, deadline: float) -> bool:
    delay = 0.001
    while not _exited(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(delay)
        delay = min(delay * 2, POLL_S)
    return True


def _collect(workdir: Path, names: Sequence[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for name in names:
        path = workdir / name
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                continue
            chunks = bytearray()
            while len(chunks) <= COLLECT_LIMIT:
                chunk = os.read(fd, CHUNK)
                if not chunk:
                    break
                chunks.extend(chunk)
            found[name] = bytes(chunks[:COLLECT_LIMIT])
        finally:
            os.close(fd)
    return found


def _remove(workdir: Path) -> None:
    try:
        shutil.rmtree(workdir)
        return
    except OSError:
        pass
    with contextlib.suppress(OSError):
        os.chmod(workdir, 0o700)
    for root, dirs, _ in os.walk(workdir):
        for name in dirs:
            path = os.path.join(root, name)
            if not os.path.islink(path):
                with contextlib.suppress(OSError):
                    os.chmod(path, 0o700)
    shutil.rmtree(workdir, ignore_errors=True)


def _status(reason: str | None, finished: bool, returncode: int | None, stderr: str) -> str:
    if reason is not None:
        return reason
    if not finished:
        return TIMEOUT
    if returncode == 0:
        return OK
    if returncode == -signal.SIGXCPU:
        return TIMEOUT
    if returncode == -signal.SIGKILL or any(m in stderr for m in MEMORY_MARKERS):
        return MEMORY
    return RUNTIME_ERROR


def run_python(
    code: str,
    stdin: str | bytes = "",
    timeout_s: float = 10.0,
    memory_mb: int = 2048,
    *,
    output_limit: int = STDOUT_LIMIT,
    collect: Sequence[str] = (),
    tmp_parent: Path | None = None,
) -> RunResult:
    data = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="sandbox-", dir=tmp_parent))
    try:
        (workdir / SCRIPT_NAME).write_text(code, encoding="utf-8")
        start = time.monotonic()
        deadline = start + timeout_s
        proc = subprocess.Popen(
            [sys.executable, "-I", SCRIPT_NAME],
            cwd=workdir,
            env=sandbox_env(workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            preexec_fn=_limits(memory_mb, math.ceil(timeout_s) + 1),
        )
        try:
            out, err, reason = _pump(proc, data, deadline, output_limit)
            finished = reason is None and _wait_leader(proc.pid, deadline)
        finally:
            _kill_group(proc.pid)
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        seconds = time.monotonic() - start
        stderr = err.decode("utf-8", errors="replace")
        if len(err) > STDERR_TAIL:
            stderr = "...[truncated]\n" + err[-STDERR_TAIL:].decode("utf-8", errors="replace")
        return RunResult(
            status=_status(reason, finished, proc.returncode, stderr),
            returncode=proc.returncode,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=stderr,
            seconds=seconds,
            files=_collect(workdir, collect),
        )
    finally:
        _remove(workdir)


def parallel_map(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))


def run_many(jobs: Sequence[Job], workers: int) -> list[RunResult]:
    return parallel_map(
        lambda job: run_python(job.code, job.stdin, job.timeout_s, job.memory_mb), jobs, workers
    )
