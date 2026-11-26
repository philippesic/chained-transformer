from __future__ import annotations

from dataclasses import dataclass

from chained_transformer.modes import MODE_NAMES, THINK_CLOSE, ModeTracker

ASSISTANT_PREFIX = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class Segments:
    think: str
    code: str
    answer: str
    code_in_answer: str
    code_in_think: str
    think_closed: bool

    @property
    def visible(self) -> str:
        return self.answer + self.code_in_answer


def split(response: str) -> Segments:
    tracker = ModeTracker()
    tracker.feed(ASSISTANT_PREFIX)
    parts: dict[str, list[str]] = {name: [] for name in MODE_NAMES}
    code_after: list[str] = []
    code_before: list[str] = []
    for ch in response:
        name = MODE_NAMES[tracker.mode]
        parts[name].append(ch)
        if name == "code":
            (code_after if tracker.thinking_done else code_before).append(ch)
        tracker.feed(ch)
    return Segments(
        think="".join(parts["think"]),
        code="".join(parts["code"]),
        answer="".join(parts["answer"]),
        code_in_answer="".join(code_after),
        code_in_think="".join(code_before),
        think_closed=THINK_CLOSE in response,
    )


def after_thinking(response: str) -> str:
    head, sep, tail = response.partition(THINK_CLOSE)
    return tail if sep else head
