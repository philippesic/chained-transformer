from __future__ import annotations

import copy

import pytest
import torch

from chained_transformer.surgery import Routing
from serving.mixed import StackedWeights, batched_greedy_decode, mode_mlp_loop, use_grouped_mlps
from serving.paging import ModePager, use_paged_mlps
from serving.tiny import TINY_PAD, TINY_PROMPTS, tiny_chained_model, tiny_texts


def layers(n: int = 2, modes: int = 3) -> list[StackedWeights]:
    g = torch.Generator().manual_seed(0)
    return [
        StackedWeights(
            torch.randn(modes, 2 * 24, 16, generator=g), torch.randn(modes, 16, 24, generator=g)
        )
        for _ in range(n)
    ]


def test_pager_evicts_least_recently_used() -> None:
    pager = ModePager(layers(), capacity=2, device="cpu")
    pager.ensure([0])
    pager.ensure([1])
    pager.ensure([0])
    pager.ensure([2])
    assert set(pager.slots) == {0, 2}
    assert pager.stats.loads == 3 and pager.stats.evictions == 1 and pager.stats.hits == 1
    assert pager.slot_table.tolist()[1] == -1


def test_pager_refuses_more_modes_than_slots() -> None:
    pager = ModePager(layers(), capacity=2, device="cpu")
    with pytest.raises(RuntimeError, match="only 2 fit"):
        pager.ensure([0, 1, 2])
    pager.ensure([0, 1])
    assert not pager.prefetch(2, keep=[0, 1])
    assert pager.prefetch(2, keep=[1])
    assert set(pager.slots) == {1, 2}


def test_paged_weights_give_the_same_outputs() -> None:
    w = layers(1)[0]
    pager = ModePager([w], capacity=2, device="cpu")
    x = torch.randn(9, 16)
    for ids in ([0, 0, 2, 2, 0, 2, 0, 0, 2], [1] * 9, [2, 1, 1, 2, 2, 1, 2, 1, 1]):
        t = torch.tensor(ids)
        pager.ensure(set(ids))
        got = mode_mlp_loop(x, pager.remap(t), pager.weights(0))
        torch.testing.assert_close(got, mode_mlp_loop(x, t, w), rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [1, 2])
def test_paged_decode_matches_resident_decode(capacity: int) -> None:
    model, _ = tiny_chained_model()
    resident = copy.deepcopy(model)
    resident_routing = Routing()
    use_grouped_mlps(resident, resident_routing)
    paged_routing = Routing()
    pager = use_paged_mlps(model, paged_routing, capacity, "cpu")
    for prompt in TINY_PROMPTS:
        want = batched_greedy_decode(
            resident, resident_routing, tiny_texts(), [prompt], 24, TINY_PAD
        )
        got = batched_greedy_decode(model, paged_routing, tiny_texts(), [prompt], 24, TINY_PAD)
        assert got.token_ids == want.token_ids
        assert got.modes == want.modes
    assert pager.stats.loads >= 3
    assert pager.stats.over_capacity_steps > 0
    assert pager.stats.streamed_bytes > 0
    if capacity == 1:
        assert pager.stats.evictions == pager.stats.loads - 1


def test_mixed_batch_over_capacity_streams_the_rest() -> None:
    model, _ = tiny_chained_model()
    resident = copy.deepcopy(model)
    resident_routing = Routing()
    use_grouped_mlps(resident, resident_routing)
    paged_routing = Routing()
    pager = use_paged_mlps(model, paged_routing, 2, "cpu")
    want = batched_greedy_decode(
        resident, resident_routing, tiny_texts(), TINY_PROMPTS, 24, TINY_PAD
    )
    got = batched_greedy_decode(model, paged_routing, tiny_texts(), TINY_PROMPTS, 24, TINY_PAD)
    assert got.token_ids == want.token_ids
    assert got.mixed_steps > 0
    assert pager.stats.over_capacity_steps > 0
