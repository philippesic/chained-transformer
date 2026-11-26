from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from transformers import Qwen3ForCausalLM

from chained_transformer.surgery import Routing
from serving.export import TOKEN_TEXTS_FILE, export_for_vllm
from serving.tiny import TINY_PROMPTS, tiny_chained_model, tiny_greedy, tiny_token_texts

vllm = pytest.importorskip("vllm")

NEW_TOKENS = 24
LONG_BASE = [0, 1, *range(6, 30), 4, 2, 4, *range(30, 60), 4, 3, *range(10, 20)]
LONG_PROMPTS = [[*LONG_BASE, k] for k in (20, 21, 22, 23)]


@pytest.fixture(scope="module")
def tiny() -> tuple[Qwen3ForCausalLM, Routing]:
    return tiny_chained_model()


@pytest.fixture(scope="module")
def llm(
    tiny: tuple[Qwen3ForCausalLM, Routing], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Any]:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
    from vllm.platforms import current_platform
    from vllm.triton_utils import HAS_TRITON

    from serving.vllm_plugin import register

    if current_platform.is_cpu() and not HAS_TRITON:
        pytest.skip("vLLM's CPU engine runs this plugin only with triton-cpu installed")

    register()
    out: Path = export_for_vllm(
        tiny[0], tmp_path_factory.mktemp("tiny-chained"), token_texts=tiny_token_texts()
    )
    engine = vllm.LLM(
        model=str(out),
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=32,
        block_size=32,
        enable_prefix_caching=True,
        seed=0,
        hf_overrides={
            "mode_tracking": {
                "prompt_mode": 0,
                "token_texts_file": TOKEN_TEXTS_FILE,
                "mlp_impl": "select",
                "record": True,
            }
        },
    )
    yield engine
    del engine


def run(engine: Any, prompts: list[list[int]]) -> tuple[list[str], list[list[int]]]:
    from vllm.inputs import TokensPrompt

    params = vllm.SamplingParams(
        temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True, detokenize=False
    )
    outs = engine.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    return [o.request_id for o in outs], [list(o.outputs[0].token_ids) for o in outs]


def worker_stats(engine: Any) -> dict[str, Any]:
    from serving.vllm_plugin.state import collect_traces

    return engine.collective_rpc(collect_traces)[0]


@pytest.mark.parametrize("impl", ["select", "loop"])
def test_vllm_greedy_matches_hf_with_mixed_batches(
    llm: Any, tiny: tuple[Qwen3ForCausalLM, Routing], impl: str
) -> None:
    from serving.vllm_plugin.state import set_mlp_impl

    assert llm.collective_rpc(set_mlp_impl, args=(impl,))[0] == 2
    before = worker_stats(llm)["mixed_steps"]
    want_tokens, want_modes = tiny_greedy(*tiny, TINY_PROMPTS, NEW_TOKENS)
    req_ids, got = run(llm, TINY_PROMPTS)
    assert got == want_tokens
    stats = worker_stats(llm)
    for req_id, modes in zip(req_ids, want_modes, strict=True):
        assert stats["traces"][req_id] == modes
    assert stats["mixed_steps"] > before
    assert {0, 1, 2} <= {m for row in want_modes for m in row}


def test_vllm_chunked_prefill_and_prefix_cache_keep_modes(
    llm: Any, tiny: tuple[Qwen3ForCausalLM, Routing]
) -> None:
    from serving.vllm_plugin.state import set_mlp_impl

    llm.collective_rpc(set_mlp_impl, args=("select",))
    want_tokens, want_modes = tiny_greedy(*tiny, LONG_PROMPTS, NEW_TOKENS)
    starts = []
    for _ in range(2):
        req_ids, got = run(llm, LONG_PROMPTS)
        assert got == want_tokens
        stats = worker_stats(llm)
        for req_id, modes in zip(req_ids, want_modes, strict=True):
            assert stats["traces"][req_id] == modes
        starts.append([stats["first_starts"][r] for r in req_ids])
    assert max(starts[1]) >= 32
    assert {0, 1, 2} <= set(want_modes[0])
