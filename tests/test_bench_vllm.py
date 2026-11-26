from __future__ import annotations

import argparse
import json
from pathlib import Path

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
