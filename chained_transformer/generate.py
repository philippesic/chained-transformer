from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache, PreTrainedTokenizerBase

from chained_transformer.modes import Mode, ModeTracker
from chained_transformer.surgery import Routing

PREFILL_CHUNK = 2048


class TokenTexts:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> TokenTexts:
        size = len(tokenizer)
        return cls(tokenizer.batch_decode([[i] for i in range(size)], skip_special_tokens=False))

    def __getitem__(self, token_id: int) -> str:
        return self.texts[token_id]

    def modes(self, token_ids: Sequence[int], prompt_mode: Mode = Mode.THINK) -> list[int]:
        tracker = ModeTracker(prompt_mode=prompt_mode)
        return [int(tracker.feed(self.texts[t])) for t in token_ids]


@dataclass
class ModeHook:
    routing: Routing
    texts: TokenTexts
    prompt_mode: Mode = Mode.THINK
    trackers: list[ModeTracker] = field(default_factory=list)
    trace: list[list[int]] = field(default_factory=list)

    def __call__(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise RuntimeError("mode tracking needs input_ids; inputs_embeds is not supported")
        cache = kwargs.get("cache", kwargs.get("past_key_values"))
        if cache is None or cache.get_seq_length() == 0:
            self.trackers = [ModeTracker(prompt_mode=self.prompt_mode) for _ in input_ids]
            self.trace = [[] for _ in input_ids]
        rows = input_ids.tolist()
        modes = [
            [int(tracker.feed(self.texts[t])) for t in row]
            for tracker, row in zip(self.trackers, rows, strict=True)
        ]
        for history, row_modes in zip(self.trace, modes, strict=True):
            history.extend(row_modes)
        self.routing.set(torch.tensor(modes, dtype=torch.long, device=input_ids.device))


@contextmanager
def mode_tracking(
    model: nn.Module, routing: Routing, texts: TokenTexts, prompt_mode: Mode = Mode.THINK
) -> Iterator[ModeHook]:
    hook = ModeHook(routing, texts, prompt_mode)
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield hook
    finally:
        handle.remove()
        routing.clear()


@dataclass
class SeparateModelsResult:
    token_ids: list[int]
    modes: list[int]
    switches: int
    reprefilled_tokens: int


@torch.no_grad()
def generate_separate_models(
    models: Sequence[nn.Module],
    texts: TokenTexts,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_ids: set[int],
    prompt_mode: Mode = Mode.THINK,
    prefill_chunk: int = PREFILL_CHUNK,
) -> SeparateModelsResult:
    device = next(models[0].parameters()).device
    tracker = ModeTracker(prompt_mode=prompt_mode)
    tokens = list(prompt_ids)
    modes = [int(tracker.feed(texts[t])) for t in tokens]
    caches = [DynamicCache() for _ in models]
    seen = [0 for _ in models]
    switches = 0
    reprefilled = 0
    for step in range(max_new_tokens):
        mode = modes[-1]
        if len(modes) > len(prompt_ids) and modes[-1] != modes[-2]:
            switches += 1
        model = models[mode]
        pending = tokens[seen[mode] :]
        if step > 0:
            reprefilled += len(pending) - 1
        for start in range(0, len(pending), prefill_chunk):
            ids = torch.tensor(
                [pending[start : start + prefill_chunk]], dtype=torch.long, device=device
            )
            logits = model(
                input_ids=ids, cache=caches[mode], use_cache=True, logits_to_keep=1
            ).logits
        seen[mode] = len(tokens)
        next_id = int(logits[0, -1].argmax())
        tokens.append(next_id)
        modes.append(int(tracker.feed(texts[next_id])))
        if next_id in eos_ids:
            break
    return SeparateModelsResult(tokens[len(prompt_ids) :], modes, switches, reprefilled)
