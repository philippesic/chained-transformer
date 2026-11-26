from __future__ import annotations

from typing import Any

from tools.memory_check import judge

ROWS = {"peak": {"measured_gib": 6.0, "predicted_gib": 7.0, "ratio": 0.857}}


def _result(status: str = "ok", reserved: float = 6.5, fallbacks: int = 0) -> dict[str, Any]:
    measured: dict[str, Any] = {"status": status, "oom_phase": "optimizer"}
    measured["reserved_peak_gib"] = reserved
    return {"measured": measured, "train_summary": {"attention_math_fallbacks": fallbacks}}


def test_out_of_memory_fails_with_or_without_a_budget() -> None:
    for budget in (None, 10.5):
        fits, text = judge(_result("oom"), ROWS, budget)
        assert fits is False and text == "OUT OF MEMORY in optimizer"


def test_budget_counts_reserved_memory() -> None:
    assert judge(_result(reserved=6.5), ROWS, 10.5) == (True, "fits")
    assert judge(_result(reserved=10.6), ROWS, 10.5) == (False, "OVER")
    assert judge(_result(), ROWS, None) == (None, "")


def test_math_attention_fallbacks_are_reported() -> None:
    fits, text = judge(_result(fallbacks=3), ROWS, 10.5)
    assert fits is True and text == "fits  attention fell back to math 3x"
