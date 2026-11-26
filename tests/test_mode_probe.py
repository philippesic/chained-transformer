from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Qwen3ForCausalLM

from chained_transformer.modes import MODE_NAMES
from tools.mode_probe import balanced_accuracy, collect, main, pick_rows, train_probe

IDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ROW: dict[str, object] = {
    "input_ids": IDS,
    "mode_ids": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    "loss_mask": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "complete": True,
}


def test_balanced_accuracy_averages_over_present_classes() -> None:
    gold = torch.tensor([0, 0, 0, 1])
    assert balanced_accuracy(torch.tensor([0, 0, 0, 0]), gold, 3) == 0.5
    assert balanced_accuracy(gold.clone(), gold, 3) == 1.0


def test_probe_weights_apply_to_raw_features() -> None:
    g = torch.Generator().manual_seed(0)
    y = torch.arange(3).repeat(100)
    centers = torch.randn(3, 16, generator=g) * 4
    x = centers[y] + torch.randn(300, 16, generator=g) * 0.1
    x = x * torch.linspace(0.01, 100, 16) + 50
    probe = train_probe(x, y, 3, steps=100, weight_decay=1e-4)
    with torch.no_grad():
        assert balanced_accuracy(probe(x).argmax(1), y, 3) == 1.0


def test_pick_rows_skips_incomplete_and_long_rows() -> None:
    rows = [ROW, {**ROW, "complete": False}, {**ROW, "input_ids": IDS * 2}, ROW]
    assert sorted(pick_rows(rows, 12, 10, seed=0)) == [0, 3]


def test_collect_samples_masked_tokens_per_label(tiny_model: Qwen3ForCausalLM) -> None:
    rows = [ROW, ROW]
    feats, labels, groups = collect(
        tiny_model, rows, [0, 1], [0, 2], per_label=2, device=torch.device("cpu"), seed=0
    )
    assert set(feats) == {0, 2}
    assert feats[0].shape == (12, tiny_model.config.hidden_size)
    assert labels.shape == (12, 2)
    assert groups.tolist() == [0] * 6 + [1] * 6
    assert torch.bincount(labels[:, 0], minlength=len(MODE_NAMES)).tolist() == [4, 4, 4]


def test_main_writes_a_report_per_layer(tiny_model: Qwen3ForCausalLM, tmp_path: Path) -> None:
    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    Dataset.from_list([ROW] * 10).save_to_disk(str(tmp_path / "data"))
    out = tmp_path / "probe.json"
    argv = ["--data", str(tmp_path / "data"), "--base", str(base), "--rows", "10"]
    main([*argv, "--per-label", "3", "--steps", "20", "--layers", "1", "2", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["rows"] == 10
    assert set(report["position_mode"]) >= {"1", "2", "majority_baseline", "test_tokens"}
    assert report["position_mode"]["test_tokens"] == 2 * (2 + 3 + 3)
