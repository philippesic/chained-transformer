from __future__ import annotations

from typing import Any

import torch
from transformers import DynamicCache, Qwen3ForCausalLM

from chained_transformer.cache import RoomyLayer, roomy_cache


def _states(length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, length, 4, generator=gen), torch.randn(2, 3, length, 4, generator=gen)


def test_layer_appends_in_place_until_the_room_runs_out() -> None:
    layer = RoomyLayer(room=6)
    parts = [_states(n, i) for i, n in enumerate((4, 1, 1, 3))]
    stores = []
    for i, (k, v) in enumerate(parts):
        keys, values = layer.update(k, v)
        torch.testing.assert_close(keys, torch.cat([p[0] for p in parts[: i + 1]], dim=-2))
        torch.testing.assert_close(values, torch.cat([p[1] for p in parts[: i + 1]], dim=-2))
        assert layer.get_seq_length() == keys.shape[-2]
        stores.append(keys.data_ptr())
    assert stores[0] == stores[1] == stores[2] != stores[3]


def test_layer_copies_again_after_the_cache_is_cut_or_reordered() -> None:
    layer = RoomyLayer(room=16)
    first, second, third = (_states(n, i) for i, n in enumerate((5, 2, 1)))
    layer.update(*first)
    layer.update(*second)
    layer.crop(-2)
    keys, _ = layer.update(*third)
    torch.testing.assert_close(keys, torch.cat([first[0], third[0]], dim=-2))
    layer.batch_select_indices(torch.tensor([1, 0]))
    keys, values = layer.update(*third)
    want = torch.cat([first[0], third[0], third[0]], dim=-2)
    want[..., :6, :] = want[..., :6, :][[1, 0]]
    torch.testing.assert_close(keys, want)
    layer.reset()
    assert layer.get_seq_length() == 0
    keys, _ = layer.update(*second)
    torch.testing.assert_close(keys, second[0])


def test_generation_matches_the_default_cache(tiny_model: Qwen3ForCausalLM) -> None:
    ids = torch.randint(1, 64, (3, 9), generator=torch.Generator().manual_seed(4))
    mask = torch.ones_like(ids)
    mask[1, :4] = 0
    kwargs: dict[str, Any] = {"attention_mask": mask, "max_new_tokens": 14, "do_sample": False}
    kwargs |= {"pad_token_id": 0, "eos_token_id": None, "prefill_chunk_size": 4}
    want = tiny_model.generate(input_ids=ids, cache=DynamicCache(), **kwargs)
    for room in (9 + 14, 5):
        cache = roomy_cache(room)
        got = tiny_model.generate(input_ids=ids, cache=cache, **kwargs)
        assert got.tolist() == want.tolist()
        assert all(isinstance(layer, RoomyLayer) for layer in cache.layers)
        assert cache.get_seq_length() == 9 + 14 - 1
