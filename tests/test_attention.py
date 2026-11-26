from __future__ import annotations

import copy

import torch
from transformers import Qwen3ForCausalLM

from chained_transformer.attention import NAME, attention_backend, use_attention


def test_fused_attention_matches_eager(tiny_model: Qwen3ForCausalLM) -> None:
    eager = copy.deepcopy(tiny_model)
    use_attention(eager, "eager")
    use_attention(tiny_model, "auto")
    assert attention_backend(tiny_model) == NAME
    ids = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = eager(input_ids=ids).logits
        got = tiny_model(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_fused_attention_backward_matches_eager(tiny_model: Qwen3ForCausalLM) -> None:
    eager = copy.deepcopy(tiny_model)
    use_attention(eager, "eager")
    use_attention(tiny_model, "auto")
    ids = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(1))
    eager(input_ids=ids, labels=ids).loss.backward()
    tiny_model(input_ids=ids, labels=ids).loss.backward()
    for (name, a), (_, b) in zip(
        eager.named_parameters(), tiny_model.named_parameters(), strict=True
    ):
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-5, msg=name)
