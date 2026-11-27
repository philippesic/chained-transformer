from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM

from chained_transformer import attention
from chained_transformer.attention import LEAN, NAME, attention_backend, decode_step, use_attention


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


def test_lean_decode_step_matches_sdpa_over_repeated_heads() -> None:
    gen = torch.Generator().manual_seed(2)
    query = torch.randn(3, 8, 1, 16, generator=gen).to(torch.bfloat16)
    key = torch.randn(3, 2, 11, 16, generator=gen).to(torch.bfloat16)
    value = torch.randn(3, 2, 11, 16, generator=gen).to(torch.bfloat16)
    mask = torch.ones(3, 1, 1, 11, dtype=torch.bool)
    mask[1, ..., :4] = False
    mask[2, ..., :9] = False
    for attn_mask in (None, mask, torch.zeros(3, 1, 1, 11).masked_fill(~mask, float("-inf"))):
        want = F.scaled_dot_product_attention(
            query.float(),
            key.float().repeat_interleave(4, dim=1),
            value.float().repeat_interleave(4, dim=1),
            attn_mask=attn_mask,
        )
        got = decode_step(query, key, value, attn_mask, None)
        assert got.dtype == torch.float32
        strided = query.transpose(1, 2).contiguous().transpose(1, 2)
        torch.testing.assert_close(decode_step(strided, key, value, attn_mask, None), got)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    scaled = decode_step(query, key, value, None, 0.5)
    want = F.scaled_dot_product_attention(
        query.float(),
        key.float().repeat_interleave(4, dim=1),
        value.float().repeat_interleave(4, dim=1),
        scale=0.5,
    )
    torch.testing.assert_close(scaled, want, rtol=1e-5, atol=1e-6)


def test_lean_attention_decodes_like_fused_with_a_padded_batch(
    tiny_model: Qwen3ForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    fused = copy.deepcopy(tiny_model)
    use_attention(fused, "auto")
    use_attention(tiny_model, "lean")
    assert attention_backend(tiny_model) == LEAN
    ids = torch.randint(1, 64, (3, 9), generator=torch.Generator().manual_seed(3))
    mask = torch.ones_like(ids)
    mask[1, :3] = 0
    mask[2, :7] = 0
    calls: list[int] = []
    step = attention.decode_step

    def spy(query: torch.Tensor, *args: Any) -> torch.Tensor:
        calls.append(query.shape[2])
        return step(query, *args)

    monkeypatch.setattr(attention, "decode_step", spy)
    kwargs: dict[str, Any] = {"attention_mask": mask, "max_new_tokens": 12, "do_sample": False}
    kwargs |= {"pad_token_id": 0, "eos_token_id": None, "output_logits": True}
    got = tiny_model.generate(input_ids=ids, return_dict_in_generate=True, **kwargs)
    want = fused.generate(input_ids=ids, return_dict_in_generate=True, **kwargs)
    assert calls and set(calls) == {1}
    assert got.sequences.tolist() == want.sequences.tolist()
    for a, b in zip(got.logits, want.logits, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
