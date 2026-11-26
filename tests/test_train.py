from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import Qwen3ForCausalLM

from chained_transformer.model import ARMS
from chained_transformer.train import IGNORE, main, make_batch, token_losses

ROW = {
    "input_ids": [5, 6, 7, 8, 9, 10, 11],
    "mode_ids": [0, 0, 1, 1, 2, 2, 2],
    "loss_mask": [0, 0, 1, 1, 1, 1, 1],
}


def test_make_batch_masks_prompt_and_shifts_labels() -> None:
    batch = make_batch(ROW, 6, torch.device("cpu"), None)
    assert batch.input_ids.tolist() == [[5, 6, 7, 8, 9, 10]]
    assert batch.targets.tolist() == [[IGNORE, 7, 8, 9, 10, 11]]
    assert batch.target_labels.tolist() == [[0, 0, 1, 1, 2, 2]]


def test_make_batch_keeps_one_label_for_separate_models() -> None:
    batch = make_batch(ROW, 6, torch.device("cpu"), 1)
    assert batch.targets.tolist() == [[IGNORE, IGNORE, 8, 9, IGNORE, IGNORE]]


def test_make_batch_cuts_long_rows_to_seq_len() -> None:
    batch = make_batch(ROW, 3, torch.device("cpu"), None)
    assert batch.input_ids.tolist() == [[5, 6, 7]]
    assert batch.targets.tolist() == [[IGNORE, 7, 8]]
    assert batch.mode_ids.tolist() == batch.target_labels.tolist() == [[0, 0, 1]]


@pytest.mark.parametrize("chunk", [1, 4, 5, 64])
def test_token_losses_do_not_depend_on_the_loss_chunk(
    tiny_model: Qwen3ForCausalLM, chunk: int
) -> None:
    row = {k: v * 3 for k, v in ROW.items()}
    batch = make_batch(row, 16, torch.device("cpu"), None)
    whole = token_losses(tiny_model, batch, 10_000)
    assert whole.shape == (16,)
    assert torch.allclose(token_losses(tiny_model, batch, chunk), whole, atol=1e-6)


@pytest.mark.parametrize("arm", ["sft", "moe"])
def test_only_mode_is_refused_for_arms_without_mode_weights(tmp_path: Path, arm: str) -> None:
    argv = ["--arm", arm, "--data", str(tmp_path), "--out", str(tmp_path / "run")]
    with pytest.raises(SystemExit):
        main([*argv, "--only-mode", "code", "--device", "cpu"])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_trains_on_cpu(tiny_model: Qwen3ForCausalLM, tmp_path: Path, arm: str) -> None:
    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    rows = [
        {**{k: v * 3 for k, v in ROW.items()}, "complete": True},
        {**{k: v * 3 for k, v in ROW.items()}, "complete": False},
    ]
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    out = tmp_path / "run"
    argv = [
        "--arm",
        arm,
        "--data",
        str(tmp_path / "data"),
        "--out",
        str(out),
        "--steps",
        "4",
        "--seq-len",
        "16",
        "--loss-chunk",
        "5",
        "--device",
        "cpu",
        "--base",
        str(base),
        "--lora-rank",
        "4",
        "--warmup",
        "1",
    ]
    if arm == "separate":
        argv += ["--mode", "code"]
    main(argv)
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(math.isfinite(r["loss"]) for r in records)
    if arm == "separate":
        assert set(records[0]["loss_by_label"]) == {"code"}
    assert json.loads((out / "summary.json").read_text())["arm"] == arm


def test_complete_only_keeps_complete_rows(tiny_model: Qwen3ForCausalLM, tmp_path: Path) -> None:
    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    rows = [{**ROW, "complete": complete} for complete in (True, False, False)]
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    out = tmp_path / "run"
    argv = ["--arm", "sft", "--data", str(tmp_path / "data"), "--out", str(out), "--steps", "1"]
    argv += ["--seq-len", "8", "--device", "cpu", "--base", str(base)]
    for extra, kept in (([], 3), (["--complete-only"], 1)):
        main([*argv, *extra])
        assert json.loads((out / "config.json").read_text())["rows"] == kept
