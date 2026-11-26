from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from chained_transformer.checkpoint import load_checkpoint, save_trained
from chained_transformer.surgery import SurgeryConfig, apply_surgery
from serving.export import export_state_dict
from serving.tiny import tiny_base, tiny_pilot_checkpoint
from tools import bench_vllm


def prepare_args(out: Path, checkpoint: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        dir=out,
        model="tiny",
        base="Qwen/Qwen3-0.6B",
        checkpoint=checkpoint,
        dtype="float32",
        num_prompts=4,
        perturb=0.0,
    )


@pytest.mark.parametrize("variant", ["full", "lora"])
def test_prepare_exports_a_trained_checkpoint(variant: str, tmp_path: Path) -> None:
    trained = tiny_pilot_checkpoint(variant, tmp_path / "trained")
    bench_vllm.prepare(prepare_args(tmp_path / "bench", trained))
    want = export_state_dict(load_checkpoint(trained, torch.float32).model)
    got = load_file(str(tmp_path / "bench" / "chained" / "model.safetensors"))
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        torch.testing.assert_close(got[name], tensor, rtol=0, atol=0)
    meta = json.loads((tmp_path / "bench" / "meta.json").read_text())
    assert meta["checkpoint"] == str(trained)
    assert all((tmp_path / "bench" / f"mode{m}" / "model.safetensors").exists() for m in range(3))


def test_prepare_refuses_checkpoints_it_cannot_serve(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    save_trained(tiny_base(), None, plain)
    with pytest.raises(SystemExit, match="plain Qwen3"):
        bench_vllm.prepare(prepare_args(tmp_path / "a", plain))
    moe = tiny_base()
    surgery = SurgeryConfig(variant="moe")
    apply_surgery(moe, surgery)
    save_trained(moe, surgery, tmp_path / "moe")
    with pytest.raises(SystemExit, match="moe"):
        bench_vllm.prepare(prepare_args(tmp_path / "b", tmp_path / "moe"))


class FakeEngine:
    def __init__(self, script: list[int], prompt_len: int, block: int) -> None:
        self.script = script
        self.prompt_len = prompt_len
        self.block = block
        self.computed: list[list[int]] = []

    def cached(self, seq: list[int]) -> int:
        best = 0
        for done in self.computed:
            n = 0
            while n < min(len(seq), len(done)) and seq[n] == done[n]:
                n += 1
            best = max(best, n)
        return min(best // self.block * self.block, len(seq) - 1)

    def generate(self, prompts: list[dict[str, Any]], params: Any) -> list[Any]:
        params = params if isinstance(params, list) else [params] * len(prompts)
        outs = []
        for prompt, param in zip(prompts, params, strict=True):
            seq = list(prompt["prompt_token_ids"])
            start = len(seq) - self.prompt_len
            gen = self.script[start : start + param.max_tokens]
            cached = self.cached(seq)
            self.computed.append(seq + gen[:-1])
            outs.append(
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=gen)], num_cached_tokens=cached)
            )
        return outs


def test_separate_counts_prefill_after_each_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("vllm")
    fake.SamplingParams = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    texts = ["x", "</think>", "<|im_start|>", "assistant", "\n"]
    prompt = [2, 3, 4]
    script = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    engines = [FakeEngine(script, len(prompt), block=2) for _ in range(3)]
    row = bench_vllm.separate(engines, [prompt], texts, max_tokens=12, segment=4)
    assert row["outputs"] == [script]
    assert row["new_tokens"] == 12
    assert (row["switches"], row["wasted_tokens"], row["calls"]) == (1, 2, 4)
    assert row["prefill_tokens"] == 6
    assert row["switch_prefill_tokens"] == 3
