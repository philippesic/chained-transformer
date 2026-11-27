from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast

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


@pytest.mark.parametrize("arm", ["full", "plain"])
def test_refill_stays_inside_the_budgeted_width(
    runners: dict[str, Runner], arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runners[arm]
    stop = _stop_token(runner, monkeypatch)
    monkeypatch.setattr(generation, "_eos_ids", lambda tok: [stop])
    alone = _by_id(runner, SETTINGS)
    monkeypatch.setattr(generation, "PREFILL_CHUNK", 5)
    model = runner.models[0]
    generate = model.generate
    calls: list[tuple[int, int, int, int | None]] = []

    def spy(**kwargs: Any) -> torch.Tensor:
        out = cast(torch.Tensor, generate(**kwargs))
        config = kwargs["generation_config"]
        calls.append(
            (
                kwargs["input_ids"].shape[1],
                out.shape[1],
                config.max_new_tokens,
                config.prefill_chunk_size,
            )
        )
        return out

    monkeypatch.setattr(model, "generate", spy)
    refilled, _ = generate_all(runner, _problems(), replace(SETTINGS, batch_size=3, refill=True))
    tok = runner.tokenizer
    prompts = [generation.chat_prompt(tok, p, SETTINGS.enable_thinking) for p in _problems()]
    longest = max(len(tok(p, add_special_tokens=False).input_ids) for p in prompts)
    budgeted = longest + SETTINGS.max_new_tokens
    assert all(after <= budgeted for _, after, _, _ in calls)
    assert any(before + SETTINGS.max_new_tokens > budgeted for before, _, _, _ in calls)
    assert any(before > 2 * chunk for before, _, _, chunk in calls if chunk)
    assert all(chunk == 5 for _, _, _, chunk in calls)
    for g in refilled:
        assert g == alone[g.task_id]


def _slot(length: int, new: int, order: int = 0) -> generation._Slot:
    return generation._Slot(
        Problem("math500", f"s{order}", ""), [0] * (length - new), [0] * new, order
    )


def test_groups_hold_rows_of_one_length_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "MIN_CALL", 4)
    assert [generation.length_class(n, 150) for n in (0, 149, 150, 299, 300, 600)] == [
        0,
        0,
        1,
        1,
        2,
        3,
    ]
    lengths = [(20, 5), (40, 5), (160, 100), (170, 100), (180, 100), (320, 150)]
    pool = [_slot(length, new, i) for i, (length, new) in enumerate(lengths)]
    group, full = generation.pick_group(pool, rows=4, capacity=1200, cap=200)
    assert not full and [s.length for s in group] == [320]
    pool += [_slot(30, 5, 6), _slot(10, 5, 7)]
    group, full = generation.pick_group(pool, rows=4, capacity=1200, cap=200)
    assert full and [s.length for s in group] == [40, 30, 20, 10]
    pool.append(_slot(175, 100, 8))
    group, full = generation.pick_group(pool, rows=4, capacity=1200, cap=200)
    assert full and [s.length for s in group] == [180, 175, 170, 160]
    pool += [_slot(310, 150, 9), _slot(330, 150, 10)]
    group, full = generation.pick_group(pool, rows=4, capacity=1200, cap=200)
    assert full and [s.length for s in group] == [330, 320, 310]


def test_a_long_row_alone_still_gets_a_call() -> None:
    group, full = generation.pick_group([_slot(590, 199)], rows=4, capacity=600, cap=200)
    assert full and len(group) == 1


@pytest.mark.parametrize("arm", ["full", "plain"])
def test_grouped_refill_runs_more_rows_inside_the_same_capacity(
    runners: dict[str, Runner], arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runners[arm]
    stop = _stop_token(runner, monkeypatch)
    monkeypatch.setattr(generation, "_eos_ids", lambda tok: [stop])
    alone = _by_id(runner, SETTINGS)
    monkeypatch.setattr(generation, "MIN_CALL", 2)
    model = runner.models[0]
    generate = model.generate
    calls: list[tuple[int, int, int]] = []

    def spy(**kwargs: Any) -> torch.Tensor:
        out = cast(torch.Tensor, generate(**kwargs))
        calls.append((out.shape[0], kwargs["input_ids"].shape[1], out.shape[1]))
        return out

    monkeypatch.setattr(model, "generate", spy)
    tok = runner.tokenizer
    prompts = [generation.chat_prompt(tok, p, SETTINGS.enable_thinking) for p in _problems()]
    longest = max(len(tok(p, add_special_tokens=False).input_ids) for p in prompts)
    capacity = 2 * (longest + SETTINGS.max_new_tokens)
    texts = generation.TokenTexts.from_tokenizer(tok)
    done: list[Generation] = []
    settings = replace(SETTINGS, batch_size=2, refill=True)
    generation._generate_refill(runner, texts, _problems(), settings, 2, done.append, rows=6)
    assert all(rows * after <= capacity for rows, _, after in calls)
    assert max(rows for rows, _, _ in calls) > 2
    assert sorted(g.task_id for g in done) == sorted(alone)
    for g in done:
        assert g == alone[g.task_id]


@pytest.mark.parametrize("arm", ["full", "plain"])
def test_loop_stop_cuts_a_repeating_row_where_the_repeat_is_proven(
    runners: dict[str, Runner], arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runners[arm]
    monkeypatch.setattr(generation, "_eos_ids", lambda tok: [len(runner.tokenizer) + 7])
    monkeypatch.setattr(generation, "LOOP_TAIL", 12)
    monkeypatch.setattr(generation, "LOOP_PERIOD", 6)
    monkeypatch.setattr(generation, "LOOP_CHECK", 1)
    finish = generation._finish
    tokens: dict[bool, dict[str, list[int]]] = {False: {}, True: {}}
    settings = replace(SETTINGS, max_new_tokens=80, batch_size=3, refill=True)
    runs: dict[bool, list[Generation]] = {}
    for stop_loops in (False, True):

        def spy(
            slot: generation._Slot, *args: Any, into: dict[str, list[int]] = tokens[stop_loops]
        ) -> Generation:
            into[slot.problem.task_id] = list(slot.new)
            return finish(slot, *args)

        monkeypatch.setattr(generation, "_finish", spy)
        runs[stop_loops], perf = generate_all(
            runner, _problems(), replace(settings, stop_loops=stop_loops)
        )
        assert perf["loop_stops"] == sum(g.stopped == "loop" for g in runs[stop_loops])
    assert all(g.stopped is None and g.new_tokens == 80 for g in runs[False])
    cut = 0
    for g in runs[True]:
        whole, kept = tokens[False][g.task_id], tokens[True][g.task_id]
        proven = [m for m in range(18, 81) if generation.loop_period(whole[:m]) is not None]
        assert kept == whole[: proven[0] if proven else 80]
        assert g.stopped == ("loop" if proven and proven[0] < 80 else None) and not g.finished
        cut += g.stopped == "loop"
    assert 0 < cut < len(runs[True])
