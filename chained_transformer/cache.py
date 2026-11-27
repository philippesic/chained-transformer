from __future__ import annotations

from functools import partial
from typing import Any, cast

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


class RoomyLayer(DynamicLayer):
    def __init__(self, room: int) -> None:
        super().__init__()
        self.room = room
        self._store: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view: torch.Tensor | None = None

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        keys, values = cast(torch.Tensor, self.keys), cast(torch.Tensor, self.values)
        have = keys.shape[-2] if keys.numel() else 0
        new = key_states.shape[-2]
        store = self._store
        if store is None or keys is not self._view or have + new > store[0].shape[-2]:
            size = max(self.room, have + new)
            shape = (*key_states.shape[:-2], size, key_states.shape[-1])
            store = (key_states.new_empty(shape), value_states.new_empty(shape))
            if have:
                store[0][..., :have, :] = keys
                store[1][..., :have, :] = values
            self._store = store
        store[0][..., have : have + new, :] = key_states
        store[1][..., have : have + new, :] = value_states
        self.keys = self._view = store[0][..., : have + new, :]
        self.values = store[1][..., : have + new, :]
        return self.keys, self.values

    def reset(self) -> None:
        self._store = self._view = None
        super().reset()


def roomy_cache(room: int) -> DynamicCache:
    cache = DynamicCache()
    cache.layer_class_to_replicate = cast(Any, partial(RoomyLayer, room))
    return cache
