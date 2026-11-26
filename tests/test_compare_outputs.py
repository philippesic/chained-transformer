from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.compare_outputs import compare, first_difference, read_outputs, stuck, summary


def test_first_difference_and_stuck() -> None:
    assert first_difference([1, 2, 3], [1, 2, 4]) == 2
    assert first_difference([1, 2], [1, 2, 3]) == 2
    assert first_difference([], [5]) == 0
    assert stuck([7, 7, 7, 1], 3)
    assert not stuck([7, 7, 1], 3)
    assert not stuck([7, 7], 3)


def test_compare_counts_agreement_and_stuck_outputs() -> None:
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]]
    b = [[1, 2, 3, 4], [5, 6, 0, 0], [9, 9, 9, 9]]
    result = compare(a, b, window=2)
    assert result["same_window"] == 3
    assert result["same_whole"] == 2
    assert result["median_first_difference"] == 4
    assert (result["stuck_a"], result["stuck_b"]) == (1, 1)
    assert summary(result).startswith("3 of 3 prompts agree on the first 2 tokens (2 on every")
    assert compare(a, b, window=3)["same_window"] == 2
    with pytest.raises(ValueError, match="same prompts"):
        compare(a, b[:2], window=2)


def test_cli_reads_bench_rows(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"arm": "base", "outputs": [[1, 2], [3, 4]]}))
    (tmp_path / "b.json").write_text(json.dumps({"arm": "base", "outputs": [[1, 2], [3, 5]]}))
    (tmp_path / "bad.json").write_text(json.dumps({"arm": "base"}))
    assert read_outputs(tmp_path / "a.json") == [[1, 2], [3, 4]]
    with pytest.raises(SystemExit, match="has no outputs"):
        read_outputs(tmp_path / "bad.json")
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.compare_outputs",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "--window",
            "2",
            "--out",
            str(tmp_path / "r.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.startswith("1 of 2 prompts agree on the first 2 tokens")
    assert json.loads((tmp_path / "r.json").read_text())["same_whole"] == 1
