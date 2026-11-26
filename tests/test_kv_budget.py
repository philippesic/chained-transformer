from __future__ import annotations

import pytest
import torch
from transformers import Qwen3ForCausalLM

from chained_transformer.checkpoint import shape_config
from evals.generation import GiB, fit_batch, kv_bytes_per_token, sequence_bytes


@pytest.fixture(scope="module")
def qwen3_0_6b() -> Qwen3ForCausalLM:
    with torch.device("meta"):
        return Qwen3ForCausalLM(shape_config("Qwen/Qwen3-0.6B")).bfloat16()


def test_kv_bytes_per_token_matches_qwen3_0_6b(qwen3_0_6b: Qwen3ForCausalLM) -> None:
    per_model = 2 * 28 * 8 * 128 * 2
    assert kv_bytes_per_token([qwen3_0_6b]) == per_model
    assert kv_bytes_per_token([qwen3_0_6b] * 3) == 3 * per_model


def test_batch_16_at_8k_tokens_does_not_fit_an_11_gb_card(qwen3_0_6b: Qwen3ForCausalLM) -> None:
    per_sequence = sequence_bytes([qwen3_0_6b], 1500, 8192)
    free = int(8.2 * GiB)
    with pytest.raises(RuntimeError, match="--batch-size 5"):
        fit_batch(16, per_sequence, free, auto=False)
    assert fit_batch(16, per_sequence, free, auto=True) == 5
    assert fit_batch(4, per_sequence, free, auto=False) == 4


def test_no_batch_fits_asks_for_fewer_tokens(qwen3_0_6b: Qwen3ForCausalLM) -> None:
    per_sequence = sequence_bytes([qwen3_0_6b], 1500, 32768)
    with pytest.raises(RuntimeError, match="--max-new-tokens"):
        fit_batch(4, per_sequence, 2 * GiB, auto=True)
