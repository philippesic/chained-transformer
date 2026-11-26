from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import cast

import pytest
import torch

from evals import generation
from evals.generation import Generation, GenSettings, Runner, generate_all, load_runner
from evals.types import Problem
from tools.tiny_checkpoint import build

PROMPTS = [
    "What is 2+2?",
    "Add two numbers and print the sum, reading them from standard input line by line.",
    "Please reason step by step.",
    "class Solution: return the indices of two numbers adding up to target.",
    "Hmm, alternatively we could check it.",
    "Let me think step by step about the final answer within boxed.",
    "import sys",
    "Wait, actually the answer is 12.",
]
SETTINGS = GenSettings(max_new_tokens=24, do_sample=False, batch_size=1, refill=False)


@pytest.fixture(scope="module")
def runners(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Runner]]:
    root = tmp_path_factory.mktemp("refill")
    paths = {"full": build(root / "full", "full"), "plain": build(root / "plain", None, 1)}
    yield {k: load_runner(str(v), None, None, torch.float32, "cpu") for k, v in paths.items()}


def _problems() -> list[Problem]:
    return [Problem("math500", f"t{i}", p) for i, p in enumerate(PROMPTS)]


def _by_id(runner: Runner, settings: GenSettings) -> dict[str, Generation]:
    out, _ = generate_all(runner, _problems(), settings)
    return {g.task_id: g for g in out}


def _stop_token(runner: Runner, monkeypatch: pytest.MonkeyPatch) -> int:
    rows: list[list[int]] = []
    trim = generation._trim

    def spy(new: list[int], eos: set[int]) -> tuple[list[int], bool]:
        rows.append(list(new))
        return trim(new, eos)

    monkeypatch.setattr(generation, "_trim", spy)
    _by_id(runner, SETTINGS)
    monkeypatch.setattr(generation, "_trim", trim)

    def spread(token: int) -> int:
        firsts = {row.index(token) if token in row else len(row) for row in rows}
        return len(firsts) if min(firsts) > 0 else 0

    best = max(sorted({t for row in rows for t in row}), key=spread)
    assert spread(best) > 3, "no token ends the responses at varied lengths"
    return best


@pytest.mark.parametrize("arm", ["full", "plain"])
def test_refill_matches_one_problem_at_a_time(
    runners: dict[str, Runner], arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runners[arm]
    stop = _stop_token(runner, monkeypatch)
    monkeypatch.setattr(generation, "_eos_ids", lambda tok: [stop])
    alone = _by_id(runner, SETTINGS)
    lengths = {g.new_tokens for g in alone.values()}
    assert len(lengths) > 2 and any(g.finished for g in alone.values())
    refilled, perf = generate_all(runner, _problems(), replace(SETTINGS, batch_size=3, refill=True))
    assert perf["refill"] and perf["batch_size"] == 3
    assert perf["generate_calls"] > 3
    assert sorted(g.task_id for g in refilled) == sorted(alone)
    for g in refilled:
        assert g == alone[g.task_id]


def test_static_batches_are_kept_behind_a_flag(runners: dict[str, Runner]) -> None:
    _, perf = generate_all(runners["plain"], _problems(), replace(SETTINGS, batch_size=3))
    assert not perf["refill"] and perf["generate_calls"] == 3


def _stops(stop: generation._StopOrRefill, rows: list[list[int]]) -> list[bool]:
    ids = cast(torch.LongTensor, torch.tensor(rows))
    flags: list[bool] = stop(ids, cast(torch.FloatTensor, torch.zeros(len(rows), 1))).tolist()
    return flags


def test_refill_stops_only_while_problems_wait() -> None:
    stop = generation._StopOrRefill(2, [5, 5, 5, 5], [9], low=2, device=torch.device("cpu"))
    ids = [[1, 1, 9], [1, 1, 3], [1, 1, 3], [1, 1, 3]]
    assert _stops(stop, ids) == [True, False, False, False]
    ids = [[1, 1, 9, 0], [1, 1, 3, 9], [1, 1, 3, 3], [1, 1, 3, 3]]
    assert _stops(stop, ids) == [True, True, True, True] and stop.early
    last = generation._StopOrRefill(2, [1, 5], [9], low=-1, device=torch.device("cpu"))
    assert _stops(last, [[1, 1, 3], [1, 1, 3]]) == [True, False]
