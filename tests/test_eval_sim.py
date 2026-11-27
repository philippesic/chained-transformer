from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import generation
from tools import eval_sim
from tools.eval_sim import Cost, Row


def test_call_cost_counts_every_cached_token_once_per_step() -> None:
    cost = Cost(fixed_ms=10.0, ms_per_1000=2.0, prefill_tokens_per_s=1000.0)
    by_hand = sum(10.0 + 2.0 * 3 * (100 + t) / 1000 for t in range(1, 5)) / 1000 + 3 * 100 / 1000
    assert cost.call_seconds(rows=3, width=100, steps=4) == pytest.approx(by_hand)


def test_every_token_is_generated_once_under_both_schedulers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "MIN_CALL", 4)
    rows = [Row(20, n) for n in (5, 30, 30, 200, 200, 200, 12, 80, 150, 200, 9, 60)]
    cost = Cost(40.0, 3.0, 3000.0)
    results = [eval_sim.simulate(rows, s, 2, 6, 200, cost) for s in eval_sim.SCHEDULERS]
    assert [r["new_tokens"] for r in results] == [sum(r.new for r in rows)] * 2
    fixed, grouped = results
    assert fixed["rows_per_step"] <= 2 < grouped["rows_per_step"]
    assert grouped["hours"] <= fixed["hours"]


def test_main_reads_responses_and_applies_the_cap_and_loop_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "responses.jsonl"
    rows = [{"new_tokens": 100}, {"new_tokens": 900, "loop_start": 150}, {"new_tokens": 50}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    read = eval_sim.read_rows([path], prompt=30, cap=400, stop_loops_after=100)
    assert [r.new for r in read] == [100, 250, 50]
    argv = ["--responses", str(path), "--cap", "400", "--step", "2", "--group-rows", "3"]
    eval_sim.main([*argv, "--fixed-ms", "40", "--ms-per-1000", "3", "--problems", "9"])
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["scheduler"] for r in out] == ["fixed", "grouped"]
    assert all(r["problems"] == 9 for r in out)
