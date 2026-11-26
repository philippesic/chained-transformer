from __future__ import annotations

import copy

import pytest
import torch
from transformers import DynamicCache, Qwen3ForCausalLM

from mode_mlp.surgery import (
    MoEMLP,
    SurgeryConfig,
    Variant,
    apply_surgery,
    freeze_all_but_mode,
    moe_aux_loss,
)

VARIANTS: list[Variant] = ["full", "lora", "moe"]


def _ids(seed: int, shape: tuple[int, int], high: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, high, shape, generator=g)


@pytest.mark.parametrize("variant", VARIANTS)
def test_identical_modes_match_base(tiny_model: Qwen3ForCausalLM, variant: Variant) -> None:
    base = copy.deepcopy(tiny_model)
    routing = apply_surgery(tiny_model, SurgeryConfig(variant=variant, lora_rank=4))
    ids = _ids(1, (2, 12), 64)
    routing.set(_ids(2, (2, 12), 3))
    with torch.no_grad():
        want = base(input_ids=ids).logits
        got = tiny_model(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_modes_route_tokens_to_their_own_mlp(tiny_model: Qwen3ForCausalLM) -> None:
    routing = apply_surgery(tiny_model, SurgeryConfig(variant="full"))
    with torch.no_grad():
        for layer in tiny_model.model.layers:
            layer.mlp.modes[1].down_proj.weight.add_(0.5)
    ids = _ids(3, (1, 10), 64)
    modes = torch.zeros(1, 10, dtype=torch.long)
    modes[0, 6:] = 1
    routing.set(modes)
    with torch.no_grad():
        mixed = tiny_model(input_ids=ids).logits
        routing.set(torch.zeros(1, 10, dtype=torch.long))
        think_only = tiny_model(input_ids=ids).logits
    torch.testing.assert_close(mixed[:, :6], think_only[:, :6])
    assert not torch.allclose(mixed[:, 6:], think_only[:, 6:])


def test_one_cache_serves_mode_switches(tiny_model: Qwen3ForCausalLM) -> None:
    routing = apply_surgery(tiny_model, SurgeryConfig(variant="full"))
    with torch.no_grad():
        for m, scale in ((1, 0.3), (2, -0.3)):
            for layer in tiny_model.model.layers:
                layer.mlp.modes[m].up_proj.weight.add_(scale)
    ids = _ids(4, (1, 9), 64)
    modes = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 0]])
    with torch.no_grad():
        routing.set(modes)
        full = tiny_model(input_ids=ids).logits
        cache = DynamicCache()
        steps = []
        for t in range(9):
            routing.set(modes[:, t : t + 1])
            out = tiny_model(input_ids=ids[:, t : t + 1], cache=cache, use_cache=True)
            steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-5)


def test_missing_mode_ids_fail_with_repair_hint(tiny_model: Qwen3ForCausalLM) -> None:
    apply_surgery(tiny_model, SurgeryConfig(variant="full"))
    with pytest.raises(RuntimeError, match="Routing.set"):
        tiny_model(input_ids=_ids(5, (1, 4), 64))


@pytest.mark.parametrize("variant", ["full", "lora"])
def test_single_mode_training_leaves_other_modes_unchanged(
    tiny_model: Qwen3ForCausalLM, variant: Variant
) -> None:
    routing = apply_surgery(tiny_model, SurgeryConfig(variant=variant, lora_rank=4))
    trainable = freeze_all_but_mode(tiny_model, 1)
    assert trainable > 0
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    opt = torch.optim.SGD([p for p in tiny_model.parameters() if p.requires_grad], lr=0.1)
    ids = _ids(6, (2, 8), 64)
    routing.set(_ids(7, (2, 8), 3))
    tiny_model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    changed = {n for n, p in tiny_model.named_parameters() if not torch.equal(p, before[n])}
    assert changed
    marker = ".modes.1." if variant == "full" else "lora_"
    assert all(marker in n for n in changed)
    if variant == "lora":
        assert all(n.endswith(".1") for n in changed)


def test_moe_router_gets_gradient_and_aux_loss(tiny_model: Qwen3ForCausalLM) -> None:
    apply_surgery(tiny_model, SurgeryConfig(variant="moe", moe_experts=3, moe_top_k=2))
    ids = _ids(8, (2, 8), 64)
    loss = tiny_model(input_ids=ids, labels=ids).loss
    aux = moe_aux_loss(tiny_model)
    assert aux is not None and torch.isfinite(aux)
    (loss + 0.01 * aux).backward()
    routers = [m.router.weight.grad for m in tiny_model.modules() if isinstance(m, MoEMLP)]
    assert all(g is not None and g.abs().sum() > 0 for g in routers)
