from __future__ import annotations

from dataclasses import replace

import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from chained_transformer.memory import ARM_NAMES, QWEN3_0_6B, ModelShape, Setup, estimate
from chained_transformer.model import ARMS, build
from chained_transformer.surgery import SurgeryConfig
from chained_transformer.train import configure_trainable

SMALL = ModelShape(
    vocab=512, hidden=64, intermediate=128, layers=2, heads=4, kv_heads=2, head_dim=16
)


def _model(shape: ModelShape) -> Qwen3ForCausalLM:
    cfg = Qwen3Config(
        vocab_size=shape.vocab,
        hidden_size=shape.hidden,
        intermediate_size=shape.intermediate,
        num_hidden_layers=shape.layers,
        num_attention_heads=shape.heads,
        num_key_value_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg)


def test_arm_names_match_training_arms() -> None:
    assert tuple(ARM_NAMES) == tuple(ARMS)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("freeze", [False, True])
def test_parameter_counts_match_real_surgery(arm: str, freeze: bool) -> None:
    model = _model(SMALL)
    build(model, arm, SurgeryConfig(lora_rank=8))  # type: ignore[arg-type]
    configure_trainable(model, arm, None, freeze)  # type: ignore[arg-type]
    est = estimate(SMALL, Setup(arm=arm, seq_len=16, lora_rank=8, freeze_embeddings=freeze))  # type: ignore[arg-type]
    assert est.params_total == sum(p.numel() for p in model.parameters())
    assert est.params_trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_qwen3_0_6b_counts_match_the_readme() -> None:
    full = estimate(QWEN3_0_6B, Setup(arm="full", seq_len=16))
    lora = estimate(QWEN3_0_6B, Setup(arm="lora", seq_len=16))
    sft = estimate(QWEN3_0_6B, Setup(arm="sft", seq_len=16))
    assert round(sft.params_total / 1e6, 1) == pytest.approx(596.0, abs=0.2)
    assert round(full.params_total / 1e6, 1) == pytest.approx(1124.5, abs=0.2)
    assert round(lora.params_total / 1e6, 1) == pytest.approx(662.1, abs=0.2)


def test_fixed_harness_is_smaller_than_current_on_turing() -> None:
    for arm in ARM_NAMES:
        fixed = estimate(QWEN3_0_6B, Setup(arm=arm, seq_len=4096))
        current = estimate(
            QWEN3_0_6B,
            replace(
                Setup(arm=arm, seq_len=4096),
                optimizer="torchao-eager",
                attention="math-fp32",
                loss_chunk=1024,
            ),
        )
        assert fixed.peak < current.peak
