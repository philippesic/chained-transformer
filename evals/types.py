from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Problem:
    benchmark: str
    task_id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    task_id: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Loaded:
    problems: list[Problem]
    source: dict[str, Any]


class Benchmark(Protocol):
    name: str

    def load(self, limit: int | None, source: str | None) -> Loaded: ...

    def score(
        self,
        problems: Sequence[Problem],
        responses: Mapping[str, str],
        workdir: Path,
        workers: int,
    ) -> list[Score]: ...

    def reference_response(self, problem: Problem) -> str | None: ...
