from __future__ import annotations

import random

import pytest

from chained_transformer.modes import Mode, ModeTracker, position_modes, segment_labels

T, C, A = Mode.THINK, Mode.CODE, Mode.ANSWER
ASSISTANT = "<|im_start|>assistant\n"


def segments(content: str, prefix: str = ASSISTANT) -> list[tuple[Mode, str]]:
    tracker = ModeTracker()
    tracker.feed(prefix)
    out: list[tuple[Mode, str]] = []
    for ch in content:
        producer = tracker.mode
        tracker.feed(ch)
        if out and out[-1][0] == producer:
            out[-1] = (producer, out[-1][1] + ch)
        else:
            out.append((producer, ch))
    return out


def test_think_code_answer_round_trip() -> None:
    text = "<think>\nhmm\n```python\nx=1\n```\nok</think>\n\nAnswer:\n```\ny\n```\ndone<|im_end|>"
    assert segments(text) == [
        (T, "<think>\nhmm\n```python\n"),
        (C, "x=1\n```\n"),
        (T, "ok</think>"),
        (A, "\n\nAnswer:\n```\n"),
        (C, "y\n```\n"),
        (A, "done<|im_end|>"),
    ]


def test_missing_think_close_keeps_truncated_trace_in_think() -> None:
    assert segments("<think>\nlong reasoning that was cut off") == [
        (T, "<think>\nlong reasoning that was cut off")
    ]


def test_four_tick_fence_contains_three_tick_lines() -> None:
    text = "</think>````md\n```python\nx\n```\n````\nafter"
    assert segments(text) == [
        (T, "</think>"),
        (A, "````md\n"),
        (C, "```python\nx\n```\n````\n"),
        (A, "after"),
    ]


def test_info_string_line_does_not_close_a_fence() -> None:
    text = "</think>```\na\n```python\nb\n```\nc"
    assert segments(text)[-2:] == [(C, "a\n```python\nb\n```\n"), (A, "c")]


def test_inline_triple_backticks_are_not_fences() -> None:
    text = "</think>use ```x``` inline\nand `y`\n"
    assert segments(text) == [(T, "</think>"), (A, "use ```x``` inline\nand `y`\n")]


def test_indented_fence_and_trailing_spaces_count() -> None:
    text = "</think>1. step\n   ```js\n   f()\n   ```  \nnext"
    assert segments(text)[-2:] == [(C, "   f()\n   ```  \n"), (A, "next")]


def test_unclosed_fence_in_think_ends_at_think_close() -> None:
    text = "<think>\n```\ndraft</think>\nfinal"
    assert segments(text) == [(T, "<think>\n```\n"), (C, "draft</think>"), (A, "\nfinal")]


def test_unclosed_fence_in_answer_stays_code_until_turn_end() -> None:
    text = "</think>```\ncode<|im_end|>\n<|im_start|>user\nq"
    got = segments(text)
    assert got[1] == (A, "```\n")
    assert got[2] == (C, "code<|im_end|>")


def test_second_think_close_in_answer_is_plain_text() -> None:
    text = "</think>the tag </think> appears\n"
    assert segments(text) == [(T, "</think>"), (A, "the tag </think> appears\n")]


def test_prompt_fences_and_tags_do_not_switch_modes() -> None:
    tracker = ModeTracker(prompt_mode=Mode.ANSWER)
    tracker.feed("<|im_start|>user\nfix this\n```\nx\n```\n</think>\n<|im_end|>\n")
    assert tracker.mode == Mode.ANSWER
    tracker.feed("<|im_start|>assistant\n")
    assert tracker.mode == Mode.THINK


def test_tilde_fences_are_not_tracked() -> None:
    assert segments("</think>~~~\nx\n~~~\n")[1] == (A, "~~~\nx\n~~~\n")


@pytest.mark.parametrize("seed", range(5))
def test_modes_do_not_depend_on_token_boundaries(seed: int) -> None:
    text = (
        "<|im_start|>user\nq```\n<|im_end|>\n"
        + ASSISTANT
        + "<think>\na\n```py\nb\n```\nc</think>\n\nd\n````\ne\n```\n````\nf<|im_end|>\n"
    )
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, len(text)), 25))
    pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)], strict=True)]
    per_char = position_modes(list(text))
    ends = [j - 1 for j in [*cuts, len(text)]]
    assert position_modes(pieces) == [per_char[e] for e in ends]


def test_segment_labels_shift_position_modes() -> None:
    assert segment_labels([0, 1, 2]) == [int(T), 0, 1]
