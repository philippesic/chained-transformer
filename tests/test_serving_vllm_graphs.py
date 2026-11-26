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
GPU_MEMORY = 0.2


@pytest.fixture(scope="module")
def tiny() -> tuple[Qwen3ForCausalLM, Routing]:
    return tiny_chained_model()


@pytest.fixture(scope="module")
def llm(
    tiny: tuple[Qwen3ForCausalLM, Routing], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Any]:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("CUDA graphs need a CUDA GPU")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
    from serving.vllm_plugin import register

    register()
    out: Path = export_for_vllm(
        tiny[0], tmp_path_factory.mktemp("tiny-chained-graphs"), token_texts=tiny_token_texts()
    )
    engine = vllm.LLM(
        model=str(out),
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=False,
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        block_size=32,
        enable_prefix_caching=True,
        seed=0,
        gpu_memory_utilization=GPU_MEMORY,
        hf_overrides={
            "mode_tracking": {
                "prompt_mode": 0,
                "token_texts_file": TOKEN_TEXTS_FILE,
                "mlp_impl": "triton",
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


def test_graphs_are_on_and_weights_are_stacked(llm: Any) -> None:
    from serving.vllm_plugin.state import runner_report

    report = llm.collective_rpc(runner_report)[0]
    assert report["model_state"] == "ModeModelState", report
    assert report["impls"] == ["triton"], report
    assert report["stacking_problems"] == [], report
    assert report["weights_device"].startswith("cuda"), report
    assert not report["enforce_eager"], report
    assert report["compilation_mode"] == "VLLM_COMPILE", report
    assert report["cudagraph_mode"] != "NONE", report


@pytest.mark.parametrize("count", [len(TINY_PROMPTS), 1, 3])
def test_compiled_triton_with_cuda_graphs_matches_hf(
    llm: Any, tiny: tuple[Qwen3ForCausalLM, Routing], count: int
) -> None:
    from serving.vllm_plugin.state import collect_traces

    prompts = TINY_PROMPTS[:count]
    want_tokens, want_modes = tiny_greedy(*tiny, prompts, NEW_TOKENS)
    req_ids, got = run(llm, prompts)
    assert got == want_tokens
    traces = llm.collective_rpc(collect_traces)[0]["traces"]
    for req_id, modes in zip(req_ids, want_modes, strict=True):
        assert traces[req_id] == modes
