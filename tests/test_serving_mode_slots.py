from __future__ import annotations

import random

import numpy as np
import pytest

from chained_transformer.modes import Mode
from serving.mode_slots import SlotModes
from serving.tiny import TINY_PROMPTS, tiny_texts, tiny_token_texts


def sequence(seed: int, prompt: list[int], length: int = 40) -> list[int]:
    rng = random.Random(seed)
    pool = [2, 3, 3, 4, 4, 4, 5] + list(range(6, 40))
    return prompt + [rng.choice(pool) for _ in range(length)]


def expected(seq: list[int]) -> list[int]:
    return tiny_texts().modes(seq)


def run_steps(slots: SlotModes, slot: int, req_id: str, seq: list[int], steps: list[int]) -> list:
    got: list[int] = []
    pos = 0
    for n in steps:
        got += slots.row_modes(slot, req_id, pos, seq[pos : pos + n])
        pos += n
    return got


@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_chunked_prefill_then_decode(chunk: int) -> None:
    for i, prompt in enumerate(TINY_PROMPTS):
        seq = sequence(i, prompt)
        slots = SlotModes(tiny_token_texts())
        slots.add(5, "r", prompt)
        steps = [chunk] * (len(prompt) // chunk) + [len(prompt) % chunk]
        steps = [s for s in steps if s] + [1] * (len(seq) - len(prompt))
        assert run_steps(slots, 5, "r", seq, steps) == expected(seq)


def test_tiny_sequences_switch_modes() -> None:
    modes = [set(expected(sequence(i, p))) for i, p in enumerate(TINY_PROMPTS)]
    assert set.union(*modes) == {0, 1, 2}


def test_prefix_cache_hit_feeds_cached_tokens_first() -> None:
    seq = sequence(7, TINY_PROMPTS[2])
    slots = SlotModes(tiny_token_texts())
    slots.add(0, "r", seq[:20])
    got = slots.row_modes(0, "r", 16, seq[16:20])
    for pos in range(20, len(seq)):
        got += slots.row_modes(0, "r", pos, seq[pos : pos + 1])
    assert got == expected(seq)[16:]


def test_preempted_request_resumes_with_generated_tokens() -> None:
    seq = sequence(3, TINY_PROMPTS[1])
    slots = SlotModes(tiny_token_texts())
    slots.add(2, "r", TINY_PROMPTS[1])
    cut = len(TINY_PROMPTS[1]) + 11
    first = run_steps(slots, 2, "r", seq, [len(TINY_PROMPTS[1])] + [1] * 11)
    slots.remove("r")
    slots.add(6, "r", seq[:cut])
    rest = slots.row_modes(6, "r", 0, seq[:cut])
    for pos in range(cut, len(seq)):
        rest += slots.row_modes(6, "r", pos, seq[pos : pos + 1])
    want = expected(seq)
    assert first == want[:cut]
    assert rest == want


def test_rejected_draft_tokens_rewind_the_tracker() -> None:
    seq = sequence(11, TINY_PROMPTS[0])
    slots = SlotModes(tiny_token_texts(), record=True)
    slots.add(1, "r", TINY_PROMPTS[0])
    pos = len(TINY_PROMPTS[0])
    got = slots.row_modes(1, "r", 0, seq[:pos])
    rng = random.Random(0)
    while pos < len(seq) - 4:
        drafts = [seq[pos + j] if rng.random() < 0.6 else 3 for j in range(4)]
        modes = slots.row_modes(1, "r", pos, drafts)
        accepted = 0
        while accepted < 4 and drafts[accepted] == seq[pos + accepted]:
            accepted += 1
        got += modes[:accepted]
        pos += accepted
        got += slots.row_modes(1, "r", pos, [seq[pos]])
        pos += 1
    assert got == expected(seq)[:pos]
    assert slots.traces()["r"] == expected(seq)[:pos]
    assert slots.slots[1].rebuilds > 0


def test_dummy_rows_get_mode_zero_and_leave_state_alone() -> None:
    seq = sequence(4, TINY_PROMPTS[2])
    slots = SlotModes(tiny_token_texts())
    slots.add(0, "real", TINY_PROMPTS[2])
    got = slots.row_modes(0, "real", 0, TINY_PROMPTS[2])
    assert slots.row_modes(0, "req_0_dummy", 0, [3, 3, 2]) == [0, 0, 0]
    assert slots.row_modes(9, "req_9_dummy", 0, [3]) == [0]
    for pos in range(len(TINY_PROMPTS[2]), len(seq)):
        got += slots.row_modes(0, "real", pos, seq[pos : pos + 1])
    assert got == expected(seq)


def test_slot_reuse_retires_the_old_request() -> None:
    slots = SlotModes(tiny_token_texts(), record=True)
    slots.add(0, "a", TINY_PROMPTS[2])
    slots.row_modes(0, "a", 0, TINY_PROMPTS[2])
    slots.add(0, "b", TINY_PROMPTS[0])
    assert not slots.owns(0, "a")
    assert slots.owns(0, "b")
    assert slots.traces()["a"] == expected(TINY_PROMPTS[2])
    slots.remove("a")
    assert slots.owns(0, "b")


def test_batch_modes_matches_each_sequence() -> None:
    seqs = [sequence(20 + i, p) for i, p in enumerate(TINY_PROMPTS)]
    slots = SlotModes(tiny_token_texts(), prompt_mode=Mode.ANSWER)
    slot_ids = np.array([3, 0, 7, 1])
    for slot, i in zip(slot_ids, range(len(seqs)), strict=True):
        slots.add(int(slot), f"r{i}", TINY_PROMPTS[i])
    got: list[list[int]] = [[] for _ in seqs]
    pos = [0] * len(seqs)
    first = True
    while min(len(s) - p for s, p in zip(seqs, pos, strict=True)) > 0:
        order = [2, 0, 3, 1] if first else [0, 1, 2, 3]
        counts = [len(TINY_PROMPTS[i]) if first else 1 for i in order]
        qsl = np.concatenate([[0], np.cumsum(counts)])
        ids = np.concatenate(
            [seqs[i][pos[i] : pos[i] + c] for i, c in zip(order, counts, strict=True)]
        )
        modes = slots.batch_modes(
            [f"r{i}" for i in order],
            slot_ids[order],
            np.array([pos[i] for i in order]),
            qsl,
            ids,
        )
        for row, i in enumerate(order):
            got[i] += modes[qsl[row] : qsl[row + 1]].tolist()
            pos[i] += counts[row]
        first = False
    for seq, g in zip(seqs, got, strict=True):
        assert g == tiny_texts().modes(seq[: len(g)], Mode.ANSWER)


def test_out_of_vocab_tokens_read_as_empty_text() -> None:
    slots = SlotModes(tiny_token_texts())
    slots.add(0, "r", [0, 1])
    assert slots.row_modes(0, "r", 0, [0, 1, 999, 2]) == [0, 0, 0, 2]
