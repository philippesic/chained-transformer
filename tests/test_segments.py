from __future__ import annotations

from evals.segments import after_thinking, split

RESPONSE = (
    "<think>\nWait.\n```python\nprint(1)\n```\nok</think>\n\n"
    "Answer \\boxed{4}.\n```python\ndef f():\n    return 4\n```\nDone."
)


def test_split_separates_code_by_side_of_think_close() -> None:
    s = split(RESPONSE)
    assert s.think_closed
    assert "Wait" in s.think and "Wait" not in s.answer
    assert "print(1)" in s.code_in_think and "print(1)" not in s.code_in_answer
    assert "return 4" in s.code_in_answer and "return 4" in s.code
    assert "boxed{4}" in s.answer and "Done." in s.answer
    assert "return 4" in s.visible


def test_unclosed_thinking_has_no_answer() -> None:
    s = split("<think>\nstill going")
    assert not s.think_closed and s.answer == "" and "still" in s.think
    assert after_thinking("<think>\nstill going") == "<think>\nstill going"
    assert after_thinking("<think>a</think>b") == "b"
