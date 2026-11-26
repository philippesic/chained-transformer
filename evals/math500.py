from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import multiprocessing as mp
import re
import shutil
import tempfile
import time
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, TypeVar

from chained_transformer.modes import THINK_CLOSE
from evals.types import Loaded, Problem, Score

T = TypeVar("T")
R = TypeVar("R")

NAME = "math500"
HF_DATASET = "HuggingFaceH4/MATH-500"
HF_SPLIT = "test"
PRM800K_REPO = "https://github.com/openai/prm800k"
PRM800K_COMMIT = "7ecc794703b2877f63226f2477a49b34f9b25163"
PRM800K_PATH = "prm800k/math_splits/test.jsonl"
POINTER_URL = f"https://raw.githubusercontent.com/openai/prm800k/{PRM800K_COMMIT}/{PRM800K_PATH}"
MEDIA_URL = (
    f"https://media.githubusercontent.com/media/openai/prm800k/{PRM800K_COMMIT}/{PRM800K_PATH}"
)
POINTER_SHA256 = "bf4bf2b73edea199aed01c5b3bdc90404d7f2419daf6931fc9a79622fcbf0cf2"
CONTENT_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
CONTENT_SIZE = 446564
EXPECTED_COUNT = 500
MAX_DOWNLOAD_BYTES = 16 * 2**20
CACHE_ROOT = Path.home() / ".cache" / "chained-transformer"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
BOXED_RE = re.compile(r"\\boxed\s*\{")
LEVEL_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Record:
    unique_id: str
    problem: str
    answer: str
    subject: str | None
    level: int | None


@dataclass(frozen=True)
class Extraction:
    think_closed: bool
    answer_text: str
    answer_box: str | None
    lenient_text: str
    lenient_box: str | None


@dataclass
class _Slot:
    proc: BaseProcess
    index: int
    since: float


def make_prompt(problem: str) -> str:
    return f"{problem}\n\n{INSTRUCTION}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced(text: str, start: int) -> int | None:
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def boxed_spans(text: str) -> list[str]:
    found = []
    pos = 0
    while (m := BOXED_RE.search(text, pos)) is not None:
        end = _balanced(text, m.end())
        if end is None:
            pos = m.end()
            continue
        found.append(text[m.end() : end])
        pos = end + 1
    return found


def last_boxed(text: str) -> str | None:
    spans = boxed_spans(text)
    return spans[-1] if spans else None


def extract(response: str) -> Extraction:
    head, sep, tail = response.partition(THINK_CLOSE)
    closed = bool(sep)
    answer_text = tail if closed else ""
    answer_box = last_boxed(answer_text)
    if closed:
        return Extraction(True, answer_text, answer_box, answer_text, answer_box)
    return Extraction(False, "", None, response, last_boxed(response))


def _level(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and (m := LEVEL_RE.search(value)):
        return int(m.group(1))
    return None


def normalise(row: Mapping[str, Any], index: int) -> Record:
    problem = row.get("problem") or row.get("question")
    answer = row.get("answer")
    if not answer and isinstance(row.get("solution"), str):
        answer = last_boxed(row["solution"])
    if not isinstance(problem, str) or not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"MATH-500 row {index} has no problem or answer: keys {sorted(row)}")
    uid = row.get("unique_id") or row.get("id") or f"{HF_SPLIT}/{index}"
    subject = row.get("subject") or row.get("type")
    return Record(
        unique_id=str(uid),
        problem=problem,
        answer=answer.strip(),
        subject=str(subject) if subject else None,
        level=_level(row.get("level")),
    )


def content_sha256(records: Sequence[Record]) -> str:
    rows = [[r.unique_id, r.problem, r.answer] for r in records]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
        total = 0
        while chunk := r.read(2**16):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
            f.write(chunk)


def parse_lfs_pointer(text: str) -> tuple[str, int]:
    fields = dict(line.split(" ", 1) for line in text.splitlines() if " " in line)
    oid = fields.get("oid", "")
    if not fields.get("version", "").startswith("https://git-lfs") or not oid.startswith("sha256:"):
        raise ValueError("not a git-lfs pointer")
    return oid.removeprefix("sha256:"), int(fields["size"])


def github_path(cache_root: Path) -> Path:
    return cache_root / NAME / f"prm800k-{PRM800K_COMMIT[:12]}" / "test.jsonl"


def download_github(cache_root: Path) -> Path:
    final = github_path(cache_root)
    if final.exists():
        if sha256_file(final) != CONTENT_SHA256:
            raise ValueError(f"cached {final} does not match sha256 {CONTENT_SHA256}; delete it")
        return final
    downloads = cache_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{NAME}-", dir=downloads))
    try:
        pointer = tmp / "pointer"
        _fetch(POINTER_URL, pointer)
        if sha256_file(pointer) != POINTER_SHA256:
            raise ValueError(f"{POINTER_URL} does not match sha256 {POINTER_SHA256}")
        oid, size = parse_lfs_pointer(pointer.read_text(encoding="utf-8"))
        if (oid, size) != (CONTENT_SHA256, CONTENT_SIZE):
            raise ValueError(f"git-lfs pointer names {oid} ({size} bytes), expected pinned one")
        content = tmp / "test.jsonl"
        _fetch(MEDIA_URL, content)
        if content.stat().st_size != CONTENT_SIZE or sha256_file(content) != CONTENT_SHA256:
            raise ValueError(f"{MEDIA_URL} does not match sha256 {CONTENT_SHA256}")
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(content), final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def _load_hf() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(HF_DATASET, split=HF_SPLIT)
    info: dict[str, Any] = {"kind": "hf", "dataset": HF_DATASET, "split": HF_SPLIT}
    info["fingerprint"] = getattr(rows, "_fingerprint", None)
    return [dict(r) for r in rows], info


def _load_github(cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = download_github(cache_root)
    info: dict[str, Any] = {
        "kind": "github",
        "repo": PRM800K_REPO,
        "commit": PRM800K_COMMIT,
        "path": PRM800K_PATH,
        "pointer_url": POINTER_URL,
        "pointer_sha256": POINTER_SHA256,
        "url": MEDIA_URL,
        "sha256": CONTENT_SHA256,
        "cache_path": str(path),
    }
    return read_jsonl(path), info


def _load_local(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "kind": "jsonl",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    return read_jsonl(path), info


def judge(job: tuple[str, str | None, str]) -> dict[str, Any]:
    from math_verify import parse, verify

    gold, box, text = job
    gold_parsed = parse(f"${gold}$")
    pred = parse(f"\\boxed{{{box}}}") if box is not None else parse(text)
    extracted = box if box is not None else next((p for p in pred if isinstance(p, str)), None)
    return {
        "passed": bool(gold_parsed) and bool(pred) and bool(verify(gold_parsed, pred)),
        "extracted": extracted,
        "gold_parsed": bool(gold_parsed),
    }


def _serve(fn: Callable[[Any], Any], conn: Connection) -> None:
    while (item := conn.recv()) is not None:
        try:
            value = fn(item[0])
        except Exception:
            value = None
        conn.send(value)
    conn.close()


def run_guarded(
    fn: Callable[[T], R], items: Sequence[T], workers: int, timeout_s: float
) -> list[R | None]:
    results: list[R | None] = [None] * len(items)
    queue = deque(range(len(items)))
    ctx = mp.get_context("fork")
    active: dict[Connection, _Slot] = {}

    def feed(conn: Connection, proc: BaseProcess) -> None:
        if queue:
            index = queue.popleft()
            conn.send((items[index],))
            active[conn] = _Slot(proc, index, time.monotonic())
            return
        with contextlib.suppress(OSError):
            conn.send(None)
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        conn.close()

    def launch() -> None:
        if not queue:
            return
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_serve, args=(fn, child), daemon=True)
        proc.start()
        child.close()
        feed(parent, proc)

    def drop(conn: Connection, slot: _Slot) -> None:
        slot.proc.kill()
        slot.proc.join()
        conn.close()
        launch()

    for _ in range(max(1, min(workers, len(items)))):
        launch()
    try:
        while active:
            for ready in wait(list(active), timeout=0.5):
                assert isinstance(ready, Connection)
                slot = active.pop(ready)
                try:
                    results[slot.index] = ready.recv()
                except (EOFError, OSError):
                    drop(ready, slot)
                    continue
                feed(ready, slot.proc)
            now = time.monotonic()
            for conn, slot in list(active.items()):
                if now - slot.since > timeout_s:
                    del active[conn]
                    drop(conn, slot)
    finally:
        for conn, slot in active.items():
            slot.proc.kill()
            conn.close()
    return results


def _outcome(result: dict[str, Any] | None) -> tuple[bool, str | None, str | None]:
    if result is None:
        return False, None, "math_verify timed out or crashed"
    return bool(result["passed"]), result["extracted"], None


class Math500:
    name: str = NAME
    item_timeout_s: float = 60.0
    cache_root: Path = CACHE_ROOT

    def load(self, limit: int | None, source: str | None) -> Loaded:
        kind = source or "hf"
        if kind == "hf":
            rows, info = _load_hf()
        elif kind == "github":
            rows, info = _load_github(self.cache_root)
        else:
            rows, info = _load_local(Path(kind))
        records = [normalise(row, i) for i, row in enumerate(rows)]
        if kind == "github" and len(records) != EXPECTED_COUNT:
            raise ValueError(f"expected {EXPECTED_COUNT} problems, got {len(records)}")
        picked = records if limit is None else records[:limit]
        info.update(
            benchmark=NAME,
            count=len(records),
            loaded=len(picked),
            content_sha256=content_sha256(records),
        )
        problems = [
            Problem(
                benchmark=NAME,
                task_id=r.unique_id,
                prompt=make_prompt(r.problem),
                meta={"answer": r.answer, "subject": r.subject, "level": r.level},
            )
            for r in picked
        ]
        return Loaded(problems=problems, source=info)

    def score(
        self,
        problems: Sequence[Problem],
        responses: Mapping[str, str],
        workdir: Path,
        workers: int,
    ) -> list[Score]:
        importlib.import_module("math_verify")

        extractions = [extract(responses.get(p.task_id, "")) for p in problems]
        jobs: list[tuple[str, str | None, str]] = []
        job_of: list[int | None] = []
        for p, ex in zip(problems, extractions, strict=True):
            if ex.lenient_box is None and not ex.lenient_text.strip():
                job_of.append(None)
                continue
            job_of.append(len(jobs))
            jobs.append((str(p.meta["answer"]), ex.lenient_box, ex.lenient_text))
        results = run_guarded(judge, jobs, workers, self.item_timeout_s)
        scores = []
        for p, ex, j in zip(problems, extractions, job_of, strict=True):
            lenient, extracted, error = (
                _outcome(results[j]) if j is not None else (False, None, None)
            )
            primary = lenient if ex.think_closed else False
            detail: dict[str, Any] = {
                "gold": p.meta["answer"],
                "think_closed": ex.think_closed,
                "extracted": extracted if ex.think_closed else None,
                "boxed": ex.answer_box is not None,
                "lenient_passed": lenient,
                "lenient_extracted": extracted,
                "lenient_boxed": ex.lenient_box is not None,
                "missing_response": p.task_id not in responses,
                "gold_parsed": (results[j] or {}).get("gold_parsed") if j is not None else None,
                "error": error,
            }
            scores.append(Score(task_id=p.task_id, passed=primary, detail=detail))
        return scores

    def reference_response(self, problem: Problem) -> str | None:
        return f"<think>\n</think>\n\nThe answer is \\boxed{{{problem.meta['answer']}}}."


BENCHMARK = Math500()


def _summary(scores: Sequence[Score], key: str = "passed") -> dict[str, Any]:
    n = len(scores)
    hits = sum(bool(s.passed if key == "passed" else s.detail[key]) for s in scores)
    return {"n": n, "passed": hits, "rate": hits / n if n else None}


def _failures(scores: Sequence[Score], expect: bool, key: str = "passed") -> list[dict[str, Any]]:
    out = []
    for s in scores:
        value = s.passed if key == "passed" else s.detail[key]
        if bool(value) != expect:
            out.append({"task_id": s.task_id, **s.detail})
    return out


def oracle_checks(
    source: str | None, limit: int | None, workers: int, workdir: Path
) -> dict[str, Any]:
    loaded = BENCHMARK.load(limit, source)
    problems = loaded.problems
    reference = {p.task_id: str(BENCHMARK.reference_response(p)) for p in problems}
    shifted = {
        p.task_id: reference[problems[(i + 1) % len(problems)].task_id]
        for i, p in enumerate(problems)
    }
    unclosed = {
        p.task_id: f"<think>\nSo the answer is \\boxed{{{p.meta['answer']}}}.\n" for p in problems
    }
    report: dict[str, Any] = {"source": loaded.source, "workers": workers}
    for name, responses, expect in (
        ("reference", reference, True),
        ("shifted_answers", shifted, False),
        ("box_only_in_unclosed_thinking", unclosed, False),
    ):
        start = time.perf_counter()
        scores = BENCHMARK.score(problems, responses, workdir, workers)
        report[name] = {
            "expect": "all pass" if expect else "none pass",
            "primary": _summary(scores),
            "lenient": _summary(scores, "lenient_passed"),
            "unexpected": _failures(scores, expect),
            "lenient_unexpected": _failures(scores, name != "shifted_answers", "lenient_passed"),
            "wall_s": round(time.perf_counter() - start, 2),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MATH-500 scorer oracle checks")
    parser.add_argument("--source", default="github")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as workdir:
        report = oracle_checks(args.source, args.limit, args.workers, Path(workdir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name in ("reference", "shifted_answers", "box_only_in_unclosed_thinking"):
        print(name, report[name]["primary"], "lenient", report[name]["lenient"])


if __name__ == "__main__":
    main()
