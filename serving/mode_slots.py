from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from chained_transformer.modes import Mode, ModeTracker


@dataclass
class Slot:
    req_id: str
    tokens: list[int]
    prefill_len: int
    tracker: ModeTracker
    fed: int = 0
    rebuilds: int = 0
    trace: list[int] = field(default_factory=list)


class SlotModes:
    def __init__(
        self, texts: Sequence[str], prompt_mode: Mode = Mode.THINK, record: bool = False
    ) -> None:
        self.texts = texts
        self.prompt_mode = prompt_mode
        self.record = record
        self.slots: dict[int, Slot] = {}
        self.slot_of: dict[str, int] = {}
        self.finished: dict[str, list[int]] = {}
        self.first_starts: dict[str, int] = {}

    def _text(self, token_id: int) -> str:
        return self.texts[token_id] if 0 <= token_id < len(self.texts) else ""

    def _tracker(self) -> ModeTracker:
        return ModeTracker(prompt_mode=self.prompt_mode)

    def add(self, slot: int, req_id: str, token_ids: Sequence[int]) -> None:
        old = self.slots.get(slot)
        if old is not None:
            self._retire(old)
        if req_id in self.slot_of:
            self.remove(req_id)
        tokens = list(token_ids)
        self.slots[slot] = Slot(req_id, tokens, len(tokens), self._tracker())
        self.slot_of[req_id] = slot

    def _retire(self, st: Slot) -> None:
        self.slot_of.pop(st.req_id, None)
        if self.record:
            self.finished[st.req_id] = st.trace[: st.fed]

    def remove(self, req_id: str) -> None:
        slot = self.slot_of.get(req_id)
        if slot is None:
            return
        st = self.slots.pop(slot)
        self._retire(st)

    def owns(self, slot: int, req_id: str) -> bool:
        st = self.slots.get(slot)
        return st is not None and st.req_id == req_id

    def row_modes(self, slot: int, req_id: str, start: int, token_ids: Sequence[int]) -> list[int]:
        st = self.slots.get(slot)
        if st is None or st.req_id != req_id:
            return [int(Mode.THINK)] * len(token_ids)
        if self.record:
            self.first_starts.setdefault(req_id, start)
        if st.fed > start:
            st.tracker = self._tracker()
            st.fed = 0
            st.rebuilds += 1
            del st.tokens[max(start, st.prefill_len) :]
            del st.trace[:]
        if start > len(st.tokens):
            raise ValueError(
                f"request {req_id} starts at position {start} but only "
                f"{len(st.tokens)} of its tokens are known"
            )
        while st.fed < start:
            mode = int(st.tracker.feed(self._text(st.tokens[st.fed])))
            self._note(st, st.fed, mode)
            st.fed += 1
        modes = []
        for offset, token in enumerate(token_ids):
            pos = start + offset
            if pos < len(st.tokens):
                st.tokens[pos] = int(token)
            else:
                st.tokens.append(int(token))
            mode = int(st.tracker.feed(self._text(int(token))))
            self._note(st, pos, mode)
            modes.append(mode)
        st.fed = start + len(token_ids)
        return modes

    def _note(self, st: Slot, pos: int, mode: int) -> None:
        if not self.record:
            return
        if pos < len(st.trace):
            st.trace[pos] = mode
            del st.trace[pos + 1 :]
        else:
            st.trace.append(mode)

    def batch_modes(
        self,
        req_ids: Sequence[str],
        slots: np.ndarray,
        starts: np.ndarray,
        query_start_loc: np.ndarray,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros(int(query_start_loc[len(req_ids)]), dtype=np.int64)
        for row, req_id in enumerate(req_ids):
            lo, hi = int(query_start_loc[row]), int(query_start_loc[row + 1])
            if hi == lo:
                continue
            out[lo:hi] = self.row_modes(
                int(slots[row]), req_id, int(starts[row]), token_ids[lo:hi].tolist()
            )
        return out

    def traces(self) -> dict[str, list[int]]:
        live = {st.req_id: st.trace[: st.fed] for st in self.slots.values()}
        return {**self.finished, **live}
