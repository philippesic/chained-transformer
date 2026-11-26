from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model
