from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from chained_transformer.modes import THINK_CLOSE, THINK_OPEN
from evals.segments import split

THINK_WORDS = ("wait", "hmm", "let me", "alternatively", "actually", "double-check", "i think")
THINK_WORD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in THINK_WORDS) + r")\b", re.I)


@dataclass(frozen=True)
class Leakage:
    think_closed: bool
    think_words: int
    answer_words: int
    code_words: int
    answer_think_words: int
    think_think_words: int
    answer_rate_per_1k: float
    think_rate_per_1k: float
    answer_has_think_word: bool
    code_chars_in_think: int
    code_chars_in_answer: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_think_words(text: str) -> int:
    return len(THINK_WORD_RE.findall(text))


def rate_per_1k(hits: int, words: int) -> float:
    return 1000.0 * hits / words if words else 0.0


def leakage(response: str) -> Leakage:
    s = split(response)
    think = s.think.replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")
    answer_words = len(s.answer.split())
    think_words = len(think.split())
    answer_hits = count_think_words(s.answer)
    think_hits = count_think_words(think)
    return Leakage(
        think_closed=s.think_closed,
        think_words=think_words,
        answer_words=answer_words,
        code_words=len(s.code.split()),
        answer_think_words=answer_hits,
        think_think_words=think_hits,
        answer_rate_per_1k=rate_per_1k(answer_hits, answer_words),
        think_rate_per_1k=rate_per_1k(think_hits, think_words),
        answer_has_think_word=answer_hits > 0,
        code_chars_in_think=len(s.code_in_think),
        code_chars_in_answer=len(s.code_in_answer),
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(items: Iterable[Leakage | str]) -> dict[str, Any]:
    rows = [leakage(x) if isinstance(x, str) else x for x in items]
    closed = [r for r in rows if r.think_closed]
    worded = [r for r in closed if r.answer_words]
    thought = [r for r in rows if r.think_words]
    answer_words = sum(r.answer_words for r in closed)
    think_words = sum(r.think_words for r in rows)
    return {
        "n": len(rows),
        "n_closed": len(closed),
        "n_unclosed": len(rows) - len(closed),
        "unclosed_rate": (len(rows) - len(closed)) / len(rows) if rows else None,
        "n_closed_with_answer_words": len(worded),
        "answer_rate_per_1k_mean": _mean([r.answer_rate_per_1k for r in worded]),
        "answer_rate_per_1k_pooled": (
            rate_per_1k(sum(r.answer_think_words for r in closed), answer_words)
            if answer_words
            else None
        ),
        "answer_has_think_word_rate": (
            sum(r.answer_has_think_word for r in closed) / len(closed) if closed else None
        ),
        "think_rate_per_1k_mean": _mean([r.think_rate_per_1k for r in thought]),
        "think_rate_per_1k_pooled": (
            rate_per_1k(sum(r.think_think_words for r in rows), think_words)
            if think_words
            else None
        ),
        "think_words_mean": _mean([r.think_words for r in rows]),
        "answer_words_mean": _mean([r.answer_words for r in closed]),
        "code_words_mean": _mean([r.code_words for r in rows]),
        "code_chars_in_think_mean": _mean([r.code_chars_in_think for r in rows]),
        "code_chars_in_answer_mean": _mean([r.code_chars_in_answer for r in closed]),
    }
