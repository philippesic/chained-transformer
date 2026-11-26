from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from evals.generation import runs_of
from evals.switch_stats import from_dataset, from_eval_dir, main, summarize

THINK, CODE, ANSWER = 0, 1, 2


def test_runs_of_compresses_a_trace() -> None:
    assert runs_of([0, 0, 1, 1, 1, 2, 1]) == [[0, 2], [1, 3], [2, 1], [1, 1]]
    assert runs_of([]) == []


def test_summary_counts_switches_runs_and_transitions() -> None:
    traces = [
        [[THINK, 10], [ANSWER, 5]],
        [[THINK, 8], [CODE, 4], [ANSWER, 2], [CODE, 3], [ANSWER, 1]],
        [[THINK, 20]],
    ]
    s = summarize(traces, [True, False, None])
    assert s["switches"]["max"] == 4 and s["switches"]["median"] == 1
    assert s["switch_histogram"] == {"1": 1, "3-5": 1, "0": 1}
    assert s["transitions"]["code->answer"]["count"] == 2
    assert s["transitions"]["answer->code"]["count"] == 1
    assert s["run_length_by_mode"]["code"]["n"] == 2
    assert s["reaches_mode"]["code"] == round(1 / 3, 4)
    assert s["first_position"]["answer"]["median"] == 11
    assert s["switches_per_1k_tokens"] == round(1000 * 5 / 53, 3)
    assert s["by_outcome"]["mean_switches_passed"] == 1
    assert s["by_outcome"]["pass_rate_by_switches"]["3-5"] == {"n": 1, "pass_rate": 0.0}


def test_eval_dir_joins_scores(tmp_path: Path) -> None:
    bench = tmp_path / "math500"
    bench.mkdir()
    rows = [
        {"task_id": "a", "mode_runs": [[0, 3], [2, 2]]},
        {"task_id": "b", "mode_runs": [[0, 5]]},
        {"task_id": "c"},
    ]
    (bench / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    scores = [{"task_id": "a", "passed": True}, {"task_id": "b", "passed": False}]
    (bench / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    out = from_eval_dir(tmp_path)["math500"]
    assert out["responses"] == 2 and out["responses_without_trace"] == 1
    assert out["by_outcome"]["mean_switches_passed"] == 1


def test_dataset_uses_loss_tokens_with_their_labels(tmp_path: Path) -> None:
    row = {
        "input_ids": list(range(8)),
        "mode_ids": [0, 0, 0, 1, 1, 2, 2, 2],
        "loss_mask": [0, 0, 1, 1, 1, 1, 1, 1],
    }
    Dataset.from_list([row]).save_to_disk(str(tmp_path / "data"))
    s = from_dataset(tmp_path / "data", None)
    assert s["tokens"] == 6
    assert s["token_share_by_mode"] == {
        "think": round(2 / 6, 4),
        "code": round(2 / 6, 4),
        "answer": round(2 / 6, 4),
    }
    main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "stats.json")])
    assert json.loads((tmp_path / "stats.json").read_text())["data"]
