from __future__ import annotations

import enum
from dataclasses import dataclass


class Mode(enum.IntEnum):
    THINK = 0
    CODE = 1
    ANSWER = 2


MODE_NAMES: tuple[str, ...] = tuple(m.name.lower() for m in Mode)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TURN_OPEN = "<|im_start|>"
TURN_CLOSE = "<|im_end|>"
ASSISTANT_ROLE = "assistant"
FENCE_CHAR = "`"
MIN_FENCE = 3
MARKER_WINDOW = max(len(THINK_CLOSE), len(TURN_OPEN), len(TURN_CLOSE))


@dataclass
class ModeTracker:
    prompt_mode: Mode = Mode.THINK
    mode: Mode = Mode.THINK
    in_assistant: bool = False
    thinking_done: bool = False
    fence_len: int = 0
    outer_mode: Mode = Mode.THINK
    line: str = ""
    recent: str = ""
    pending_role: bool = False
    role: str = ""

    def __post_init__(self) -> None:
        self.mode = self.prompt_mode
        self.outer_mode = self.prompt_mode

    def feed(self, text: str) -> Mode:
        if self.pending_role or ">" in text or "\n" in text:
            for ch in text:
                self._step(ch)
        else:
            self.recent = (self.recent + text)[-MARKER_WINDOW:]
            if self.in_assistant:
                self.line += text
        return self.mode

    def _step(self, ch: str) -> None:
        self.recent = (self.recent + ch)[-MARKER_WINDOW:]
        if self.pending_role:
            self._read_role(ch)
            return
        if self.recent.endswith(TURN_OPEN):
            self._start_turn()
            return
        if not self.in_assistant:
            return
        if self.recent.endswith(TURN_CLOSE):
            self._end_turn()
            return
        if not self.thinking_done and self.recent.endswith(THINK_CLOSE):
            self.thinking_done = True
            self.fence_len = 0
            self.mode = Mode.ANSWER
            self.outer_mode = Mode.ANSWER
            self.line = ""
            return
        if ch == "\n":
            self._end_line()
        else:
            self.line += ch

    def _start_turn(self) -> None:
        self.pending_role = True
        self.role = ""
        self.in_assistant = False
        self.mode = self.prompt_mode
        self.line = ""

    def _read_role(self, ch: str) -> None:
        if ch == "\n":
            self.pending_role = False
            if self.role.strip() == ASSISTANT_ROLE:
                self.in_assistant = True
                self.thinking_done = False
                self.fence_len = 0
                self.mode = Mode.THINK
                self.outer_mode = Mode.THINK
                self.line = ""
        else:
            self.role += ch

    def _end_turn(self) -> None:
        self.in_assistant = False
        self.fence_len = 0
        self.mode = self.prompt_mode
        self.line = ""

    def _end_line(self) -> None:
        stripped = self.line.strip()
        self.line = ""
        ticks = len(stripped) - len(stripped.lstrip(FENCE_CHAR))
        if self.fence_len == 0:
            if ticks >= MIN_FENCE and FENCE_CHAR not in stripped[ticks:]:
                self.fence_len = ticks
                self.outer_mode = self.mode
                self.mode = Mode.CODE
        elif ticks >= self.fence_len and stripped == FENCE_CHAR * ticks:
            self.fence_len = 0
            self.mode = self.outer_mode


def position_modes(token_texts: list[str], prompt_mode: Mode = Mode.THINK) -> list[int]:
    tracker = ModeTracker(prompt_mode=prompt_mode)
    return [int(tracker.feed(t)) for t in token_texts]


def segment_labels(position_mode_ids: list[int], prompt_mode: Mode = Mode.THINK) -> list[int]:
    return [int(prompt_mode), *position_mode_ids[:-1]]
