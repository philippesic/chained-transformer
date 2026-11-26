from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


@dataclass
class Allocation:
    nbytes: int
    op: str
    phase: str


@dataclass
class PhasePeak:
    bytes: int = 0
    top: list[tuple[int, str, str]] = field(default_factory=list)


class LiveBytes(TorchDispatchMode):
    def __init__(self, existing: Iterable[torch.Tensor] = (), keep_top: int = 8) -> None:
        super().__init__()
        self.live: dict[int, Allocation] = {}
        self.current = 0
        self.phase = "setup"
        self.peaks: dict[str, PhasePeak] = {}
        self.keep_top = keep_top
        for t in existing:
            self._track(t, "existing")

    def set_phase(self, name: str) -> None:
        self.phase = name
        self.peaks.setdefault(name, PhasePeak(self.current, self._top()))

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor):
                self._track(t, str(func.__name__))
        return out

    def _track(self, t: torch.Tensor, op: str) -> None:
        storage = t.untyped_storage()
        key = storage._cdata
        if key in self.live:
            return
        nbytes = storage.nbytes()
        self.live[key] = Allocation(nbytes, op, self.phase)
        self.current += nbytes
        weakref.finalize(storage, self._free, key)
        peak = self.peaks.setdefault(self.phase, PhasePeak())
        if self.current > peak.bytes:
            peak.bytes = self.current
            if nbytes >= 1 << 20:
                peak.top = self._top()

    def _free(self, key: int) -> None:
        alloc = self.live.pop(key, None)
        if alloc is not None:
            self.current -= alloc.nbytes

    def _top(self) -> list[tuple[int, str, str]]:
        biggest = sorted(self.live.values(), key=lambda a: a.nbytes, reverse=True)
        return [(a.nbytes, a.op, a.phase) for a in biggest[: self.keep_top]]

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "peak_gib": round(p.bytes / 2**30, 4),
                "top": [
                    {"gib": round(n / 2**30, 4), "op": op, "born": phase} for n, op, phase in p.top
                ],
            }
            for name, p in self.peaks.items()
        }
