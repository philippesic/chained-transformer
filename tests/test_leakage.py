from __future__ import annotations

import pytest

from evals.leakage import THINK_WORDS, aggregate, count_think_words, leakage

RESPONSE = (
    "<think>\nWait, let me check. Hmm.\n```python\nx = 1\n```\nActually fine.\n</think>\n\n"
    "The answer is 4. Wait, I think so.\n```python\nwait = 2\n```\nDone."
)


def test_think_word_list_and_whole_word_matching() -> None:
    assert THINK_WORDS == (
        "wait",
        "hmm",
        "let me",
        "alternatively",
        "actually",
        "double-check",
        "i think",
    )
    assert count_think_words("WAIT. Let me double-check; I think, hmm, actually alternatively") == 7
    assert count_think_words("awaiting hmmm letme factually thinking") == 0


def test_leakage_counts_each_segment() -> None:
    r = leakage(RESPONSE)
    assert r.think_closed
    assert (r.think_words, r.think_think_words, r.think_rate_per_1k) == (8, 4, 500.0)
    assert (r.answer_words, r.answer_think_words, r.answer_rate_per_1k) == (10, 2, 200.0)
    assert r.answer_has_think_word
    assert r.code_words == 8
    assert (r.code_chars_in_think, r.code_chars_in_answer) == (len("x = 1\n```\n"), 13)
    assert r.as_dict()["answer_rate_per_1k"] == 200.0


def test_unclosed_thinking_has_no_answer_leakage() -> None:
    r = leakage("<think>\nhmm wait")
    assert not r.think_closed and r.answer_words == 0 and not r.answer_has_think_word
    assert r.think_rate_per_1k == 1000.0


def test_aggregate_counts_answer_rates_over_closed_only() -> None:
    out = aggregate(
        [
            RESPONSE,
            "<think>\nhmm wait",
            "<think></think>The answer is 3.",
            leakage("<think>ok</think>"),
        ]
    )
    assert (out["n"], out["n_closed"], out["n_unclosed"]) == (4, 3, 1)
    assert out["unclosed_rate"] == 0.25
    assert out["n_closed_with_answer_words"] == 2
    assert out["answer_rate_per_1k_mean"] == 100.0
    assert out["answer_rate_per_1k_pooled"] == pytest.approx(1000 * 2 / 14)
    assert out["answer_has_think_word_rate"] == pytest.approx(1 / 3)
    assert out["think_rate_per_1k_mean"] == 500.0
    assert out["think_rate_per_1k_pooled"] == pytest.approx(1000 * 6 / 11)
    assert out["think_words_mean"] == 2.75
    assert out["answer_words_mean"] == pytest.approx(14 / 3)
    assert out["code_chars_in_think_mean"] == 2.5
    assert out["code_chars_in_answer_mean"] == pytest.approx(13 / 3)


def test_aggregate_of_nothing_is_empty() -> None:
    out = aggregate([])
    assert out["n"] == 0 and out["unclosed_rate"] is None and out["answer_rate_per_1k_mean"] is None
