from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import Qwen3ForCausalLM

from chained_transformer.train import main as train_main
from tools import loss_vs_base


def test_steps_parse_ranges_and_single_steps() -> None:
    assert loss_vs_base.parse_steps(["3", "0-2", "2"]) == [0, 1, 2, 3]


def test_rows_of_a_step_follow_the_training_cursor_across_passes() -> None:
    order = loss_vs_base.row_order(5, seed=1)
    assert sorted(order) == [0, 1, 2, 3, 4]
    assert loss_vs_base.rows_of_step(0, 2, order) == order[:2]
    assert loss_vs_base.rows_of_step(2, 2, order) == [order[4], order[0]]


def test_base_loss_matches_an_untrained_run_step_by_step(
    tiny_model: Qwen3ForCausalLM, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    rng = random.Random(0)
    rows = []
    for i in range(7):
        n = rng.randint(9, 20)
        rows.append(
            {
                "input_ids": [rng.randrange(5, 60) for _ in range(n)],
                "mode_ids": [rng.randrange(3) for _ in range(n)],
                "loss_mask": [0, 0] + [1] * (n - 2),
                "domain": "math" if i % 2 else "code",
                "complete": True,
            }
        )
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    run = tmp_path / "run"
    argv = [
        "--arm",
        "sft",
        "--data",
        str(tmp_path / "data"),
        "--out",
        str(run),
        "--base",
        str(base),
    ]
    argv += ["--device", "cpu", "--steps", "5", "--seq-len", "16", "--grad-accum", "2"]
    train_main([*argv, "--lr", "0", "--complete-only", "--seed", "3"])
    out = tmp_path / "cmp.jsonl"
    cli = ["--run-dir", str(run), "--steps", "0-4", "--dtype", "bfloat16", "--out", str(out)]
    loss_vs_base.main([*cli, "--calibration-below", "2"])
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4]
    assert max(r["base"] for r in records) - min(r["base"] for r in records) > 0.05
    assert all(abs(r["diff"]) < 2e-3 for r in records)
    assert records[4]["passes_before"] == 1
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["calibration"]["steps"] == [0, 1]
    assert summary["trained"]["all"]["n"] == 3
    loss_vs_base.main([*cli, "--calibration-below", "2"])
    assert len(out.read_text().splitlines()) == 5
