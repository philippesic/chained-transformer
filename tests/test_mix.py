from __future__ import annotations

from chained_transformer.mix import (
    CANDIDATES,
    MixPolicy,
    RowProfile,
    compare_policies,
    profile,
    select,
    summarize,
)

THINK, CODE, ANSWER = 0, 1, 2


def _row(domain: str, think: int, code: int, answer: int, complete: bool = True) -> RowProfile:
    modes = [THINK] * 4 + [THINK] * think + [CODE] * code + [ANSWER] * answer
    mask = [0] * 4 + [1] * (think + code + answer)
    return profile(modes, mask, domain, complete)


def test_profile_counts_labels_switches_and_fences() -> None:
    modes = [0] * 3 + [0] * 5 + [1] * 3 + [0] * 2 + [2] * 4 + [1] * 3 + [2] * 2
    mask = [0] * 3 + [1] * (len(modes) - 3)
    p = profile(modes, mask, "code", True)
    assert sum(p.label_tokens) == len(modes) - 3
    assert p.switches == 5
    assert p.fences_in_think == 1 and p.fences_in_answer == 1
    assert p.answer_start is not None and 0 < p.answer_start < 1
    assert len(p.code_starts) == 2


def test_summary_reports_length_caps_and_shares() -> None:
    rows = [_row("math", 100, 0, 10), _row("code", 50, 30, 20), _row("math", 9000, 0, 5, False)]
    s = summarize(rows, caps=(64, 1024))
    assert s["complete_rows"] == 2
    caps = s["length_caps"]
    assert isinstance(caps, dict)
    assert caps["64"]["rows"] == 0 and caps["1024"]["rows"] == 2


def test_equal_domain_policy_balances_tokens() -> None:
    rows = [_row("math", 90, 0, 10) for _ in range(30)] + [_row("code", 60, 20, 20)] * 10
    picked = [rows[i] for i in select(rows, MixPolicy(10_000), None, seed=0)]
    by_domain: dict[str, int] = {}
    for r in picked:
        by_domain[r.domain] = by_domain.get(r.domain, 0) + r.total_tokens
    assert abs(by_domain["math"] - by_domain["code"]) <= 104


def test_policy_drops_incomplete_and_long_rows() -> None:
    rows = [_row("math", 10, 0, 5), _row("math", 10, 0, 5, False), _row("math", 5000, 0, 5)]
    assert select(rows, MixPolicy(1000, None), None, seed=0) == [0]


def test_compare_policies_covers_all_candidates() -> None:
    rows = [_row("math", 90, 0, 10), _row("code", 60, 20, 20), _row("science", 40, 0, 10)]
    out = compare_policies(rows, CANDIDATES, seed=0)
    assert set(out) == set(CANDIDATES)
