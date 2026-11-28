from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from chained_transformer.attention import FALLBACKS, use_attention
from chained_transformer.model import ARMS
from chained_transformer.optim import BLOCK, ChunkedAdamW8bit
from chained_transformer.surgery import SurgeryConfig, Variant, apply_surgery, moe_aux_loss
from chained_transformer.train import main

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]
CUDA = torch.device("cuda")
MiB = 2**20


def _model(layers: int = 2, device: torch.device | None = None) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    with device or torch.device("cpu"):
        return Qwen3ForCausalLM(cfg).float().eval()


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 128, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_attention_on_cuda_matches_eager_without_math_fallback(dtype: torch.dtype) -> None:
    eager = _model(device=CUDA)
    if dtype == torch.bfloat16:
        eager = eager.bfloat16()
    use_attention(eager, "eager")
    fused = copy.deepcopy(eager)
    use_attention(fused, "auto")
    ids = _ids(512, 0).to(CUDA)
    with torch.no_grad():
        want = eager(input_ids=ids).logits
    before = len(FALLBACKS)
    out = fused(input_ids=ids, labels=ids)
    out.loss.backward()
    assert len(FALLBACKS) == before, FALLBACKS[before:]
    tol = 3e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out.logits, want, rtol=tol, atol=tol)


def test_chunked_adamw8bit_on_cuda_tracks_cpu() -> None:
    g = torch.Generator().manual_seed(0)
    init = torch.randn(BLOCK * 64, generator=g)
    grads = [torch.randn(BLOCK * 64, generator=g) for _ in range(3)]
    results = []
    for device in (torch.device("cpu"), CUDA):
        p = torch.nn.Parameter(init.clone().to(device))
        opt = ChunkedAdamW8bit([p], lr=1e-2, chunk_numel=BLOCK * 16)
        for grad in grads:
            p.grad = grad.to(device)
            opt.step()
        assert opt.state[p]["m_codes"].device.type == device.type
        results.append(p.detach().cpu())
    diff = (results[1] - results[0]).abs()
    assert int((diff > 1e-5).sum()) <= 16, diff.max()


def test_bf16_stochastic_rounding_on_cuda_is_unbiased() -> None:
    p = torch.nn.Parameter(torch.ones(BLOCK * 1024, dtype=torch.bfloat16, device=CUDA))
    opt = ChunkedAdamW8bit([p], lr=1e-4)
    p.grad = torch.ones_like(p)
    opt.step()
    assert p.dtype == torch.bfloat16
    assert float((1 - p.float()).mean()) == pytest.approx(1e-4, rel=0.1)


def test_optimizer_step_temporaries_stay_within_a_chunk() -> None:
    p = torch.nn.Parameter(torch.zeros(1 << 24, device=CUDA))
    p.grad = torch.ones_like(p)
    opt = ChunkedAdamW8bit([p], lr=1e-3, chunk_numel=1 << 19)
    opt.step()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    opt.step()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base < 64 * MiB


def test_moe_backward_with_checkpointing_frees_recomputed_activations() -> None:
    model = _model(layers=4, device=CUDA)
    apply_surgery(model, SurgeryConfig(variant="moe", moe_experts=3, moe_top_k=2))
    use_attention(model, "auto")
    model.activation_checkpointing_enable({"use_reentrant": False})
    model.train()
    ids = _ids(1024, 1).to(CUDA)

    def step() -> None:
        loss = model(input_ids=ids, labels=ids).loss
        aux = moe_aux_loss(model)
        assert aux is not None
        (loss + 0.01 * aux).backward()

    step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_bytes = sum(g.numel() * g.element_size() for g in grads)
    assert torch.cuda.memory_allocated() - before < grad_bytes + 4 * MiB


def test_moe_matches_base_over_a_long_batched_greedy_run_in_bf16() -> None:
    base = _model(device=CUDA).bfloat16()
    moe = copy.deepcopy(base)
    apply_surgery(moe, SurgeryConfig(variant="moe", moe_experts=3, moe_top_k=2))
    lengths = (40, 7, 23)
    width = max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, n in enumerate(lengths):
        ids[row, width - n :] = _ids(n, 10 + row)[0]
        mask[row, width - n :] = 1
    cfg = GenerationConfig(
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        output_logits=True,
        return_dict_in_generate=True,
    )
    inputs = {"input_ids": ids.to(CUDA), "attention_mask": mask.to(CUDA)}
    with torch.no_grad():
        want = base.generate(**inputs, generation_config=cfg)
        got = moe.generate(**inputs, generation_config=cfg)
    assert want.sequences.shape == (len(lengths), width + 1024)
    differ = [
        i
        for i, (a, b) in enumerate(zip(got.logits, want.logits, strict=True))
        if not torch.equal(a, b)
    ]
    assert not differ, f"logits first differ at new token {differ[0]}"
    assert torch.equal(got.sequences, want.sequences)


@pytest.mark.parametrize("compute", ["bfloat16", "float16"])
@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_trains_on_cuda_without_math_attention(
    tmp_path: Path, arm: str, compute: str
) -> None:
    base = tmp_path / "base"
    _model().save_pretrained(base)
    row = {
        "input_ids": list(range(5, 69)) * 2,
        "mode_ids": [0] * 40 + [1] * 40 + [2] * 48,
        "loss_mask": [0] * 8 + [1] * 120,
        "complete": True,
    }
    Dataset.from_list([row, row]).save_to_disk(str(tmp_path / "data"))
    out = tmp_path / "run"
    argv = ["--arm", arm, "--data", str(tmp_path / "data"), "--out", str(out)]
    argv += ["--steps", "3", "--seq-len", "64", "--device", "cuda", "--base", str(base)]
    argv += ["--lora-rank", "4", "--warmup", "1", "--compute-dtype", compute]
    if arm == "separate":
        argv += ["--mode", "code"]
    earlier = len(FALLBACKS)
    main(argv)
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert len(records) == 3 and all(math.isfinite(r["loss"]) for r in records)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["attention_math_fallbacks"] == earlier, FALLBACKS[earlier:]


@pytest.mark.parametrize("variant", [None, "full", "lora", "moe"])
def test_captured_graph_decode_matches_eager(tmp_path: Path, variant: Variant | None) -> None:
    from chained_transformer.generate import TokenTexts, mode_tracking
    from chained_transformer.graph_decode import greedy_generate
    from evals.generation import _StopOrRefill, load_runner
    from tools.tiny_checkpoint import build

    runner = load_runner(str(build(tmp_path / "m", variant)), None, None, torch.float32, "cuda")
    model, tok = runner.models[0], runner.tokenizer
    texts = TokenTexts.from_tokenizer(tok)
    prompts = ["Please reason step by step.", "import sys", "What is 2+2?"]
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids, mask = enc.input_ids.to(CUDA), enc.attention_mask.to(CUDA)

    def run(capture: bool) -> list[list[int]]:
        stop = _StopOrRefill(ids.shape[1], [40] * 3, [10**6], -1, CUDA)
        if runner.routing is None:
            out = greedy_generate(model, ids, mask, 40, 0, stop, capture=capture)
        else:
            with mode_tracking(model, runner.routing, texts) as hook:
                out = greedy_generate(
                    model, ids, mask, 40, 0, stop, runner.routing, hook, capture=capture
                )
        rows: list[list[int]] = out.tolist()
        return rows

    assert run(capture=True) == run(capture=False)
