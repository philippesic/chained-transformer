from __future__ import annotations

from chained_transformer.evals import (
    extract_code,
    final_answer,
    score_math,
    segment_text,
    think_word_leakage,
)

RESPONSE = (
    "<think>\nWait, let me try 2+2.\n```python\nprint(2+2)\n```\nHmm ok</think>\n\n"
    "The answer is \\boxed{4}.\n```python\ndef f():\n    return 4\n```\n"
)


def test_segments_split_think_code_answer() -> None:
    parts = segment_text(RESPONSE)
    assert "Wait" in parts["think"] and "Hmm" in parts["think"]
    assert "print(2+2)" in parts["code"] and "return 4" in parts["code"]
    assert "boxed{4}" in parts["answer"] and "Wait" not in parts["answer"]


def test_math_scoring_reads_only_the_answer() -> None:
    assert score_math(RESPONSE, "4")
    assert not score_math(RESPONSE, "5")
    assert not score_math("<think>\n\\boxed{4} maybe</think>\n\nIt is \\boxed{5}.", "4")


def test_truncated_thinking_falls_back_to_full_text() -> None:
    assert final_answer("<think>\nstill going") == "<think>\nstill going"


def test_leakage_counts_think_words_in_answer_only() -> None:
    assert think_word_leakage(RESPONSE) == 0.0
    leaky = "<think>\nx</think>\n\nWait, actually let me redo this. Hmm."
    assert think_word_leakage(leaky) > 0


def test_extract_code_takes_last_answer_block() -> None:
    assert extract_code(RESPONSE).strip() == "def f():\n    return 4"
