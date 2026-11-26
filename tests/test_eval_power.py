from __future__ import annotations

import pytest

from tools import eval_power
from tools.eval_power import FULL_SIZES, discordance_of, power_at, required_sizes, seed_se


def test_target_se_gives_the_planned_power() -> None:
    target = eval_power.target_seed_se()
    assert target == pytest.approx(1.138, abs=1e-3)
    assert power_at(target) == pytest.approx(eval_power.POWER, abs=1e-9)


@pytest.mark.parametrize(
    ("d", "sizes"),
    [
        (0.05, {"math500": 129, "humanevalplus": 129, "livecodebench": 129}),
        (0.10, {"math500": 361, "humanevalplus": 164, "livecodebench": 361}),
        (0.30, FULL_SIZES),
        (0.001, {"math500": 100, "humanevalplus": 100, "livecodebench": 100}),
    ],
)
def test_required_sizes(d: float, sizes: dict[str, int]) -> None:
    disc = dict.fromkeys(FULL_SIZES, d)
    got = required_sizes(disc, FULL_SIZES)
    assert got == sizes
    if got != FULL_SIZES:
        assert seed_se(disc, got) <= eval_power.target_seed_se() + 1e-9


def test_uneven_discordance_puts_more_problems_where_arms_disagree() -> None:
    disc = {"math500": 0.04, "humanevalplus": 0.04, "livecodebench": 0.12}
    got = required_sizes(disc, FULL_SIZES)
    assert got["livecodebench"] > got["math500"] == got["humanevalplus"]
    assert seed_se(disc, got) <= eval_power.target_seed_se() + 1e-9


def test_discordance_counts_shared_problems_only() -> None:
    a = {"x": True, "y": False, "z": True}
    b = {"x": False, "y": False, "w": True}
    assert discordance_of(a, b) == (0.5, 2)
    assert discordance_of(a, {}) == (0.0, 0)


def test_main_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    eval_power.main(["--discordance", "0.1", "0.2"])
    out = capsys.readouterr().out
    assert "| 0.10 | 361 / 164 / 361 | 57% | 1.01 | 87% |" in out
    assert "| 0.20 | not reachable | 100% | 1.43 | 63% |" in out
