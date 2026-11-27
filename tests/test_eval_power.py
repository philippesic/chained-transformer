from __future__ import annotations

import math

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
        (0.10, {"math500": 383, "humanevalplus": 164, "livecodebench": 341}),
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
    assert "| 0.10 | 383 / 164 / 341 | 88% | 1.11 | 82% |" in out
    assert "| 0.20 | not reachable | 100% | 1.57 | 56% |" in out


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.9])
def test_both_margins_clear_a_line_at_their_mean_as_the_orthant_formula_says(rho: float) -> None:
    expected = 0.25 + math.asin(rho) / (2 * math.pi)
    assert eval_power.both_clear(2.0, 1.14, 2.0, rho) == pytest.approx(expected, abs=1e-6)


def test_kill_rule_odds_at_a_true_two_point_gap() -> None:
    same = eval_power.kill_rule_odds(2.0, 1.14, seeds=2, rho=1.0)
    assert same["seed_passes"] == pytest.approx(0.5)
    assert (same["continue"], same["mixed"], same["kill"]) == pytest.approx((0.25, 0.5, 0.25))
    apart = eval_power.kill_rule_odds(2.0, 1.14, seeds=2, rho=0.0)
    assert apart["seed_passes"] == pytest.approx(0.25, abs=1e-6)
    assert apart["continue"] == pytest.approx(0.0625, abs=1e-6)


@pytest.mark.parametrize("gap", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_a_third_seed_makes_the_every_seed_rule_stricter(gap: float) -> None:
    two = eval_power.kill_rule_odds(gap, 1.14, seeds=2)
    three = eval_power.kill_rule_odds(gap, 1.14, seeds=3)
    for odds in (two, three):
        assert odds["continue"] + odds["mixed"] + odds["kill"] == pytest.approx(1.0)
    assert three["continue"] <= two["continue"]
    assert three["kill"] <= two["kill"]
    assert three["mixed"] >= two["mixed"]


def test_more_problems_or_training_noise_move_the_odds_the_expected_way() -> None:
    planned = eval_power.kill_rule_odds(3.0, 1.14)["continue"]
    assert eval_power.kill_rule_odds(3.0, 0.5)["continue"] > planned
    assert eval_power.kill_rule_odds(3.0, 1.14, training_sd=1.0)["continue"] < planned
    at_line = [eval_power.kill_rule_odds(2.0, se)["continue"] for se in (2.0, 1.14, 0.3)]
    assert at_line == pytest.approx([(1 / 3) ** 2] * 3, abs=1e-3)


def test_zero_standard_error_does_not_divide_by_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert power_at(0.0) == 1.0
    assert eval_power.kill_rule_odds(2.0, 0.0)["continue"] == 1.0
    assert eval_power.kill_rule_odds(1.9, 0.0)["kill"] == 1.0
    eval_power.main(["--discordance", "0"])
    assert "| 0.00 |" in capsys.readouterr().out


def test_main_prints_the_odds_table(capsys: pytest.CaptureFixture[str]) -> None:
    eval_power.main(["--odds", "--gaps", "2", "4"])
    out = capsys.readouterr().out
    assert "| 2 | 33% | 11% / 44% / 44% | 4% / 67% / 30% | 69% | 86% |" in out
    assert "| 4 | 93% | 86% / 13% / 0% | 80% / 20% / 0% | 100% | 100% |" in out
    assert "1.32 points with 2 seeds" in out


def test_seed_mean_rule_with_one_margin_is_the_test_the_sample_was_sized_for() -> None:
    target = eval_power.target_seed_se()
    assert eval_power.seed_mean_continue(2.0, target, rho=1.0) == pytest.approx(0.8, abs=1e-4)
    assert eval_power.seed_mean_continue(0.0, target, rho=1.0) == pytest.approx(0.05, abs=1e-4)
    assert eval_power.seed_mean_continue(2.0, 1.14, rho=1.0) == pytest.approx(
        power_at(1.14), abs=1e-4
    )


@pytest.mark.parametrize("gap", [1.0, 2.0, 3.0])
def test_a_third_seed_helps_the_seed_mean_rule(gap: float) -> None:
    two = eval_power.seed_mean_continue(gap, 1.14, seeds=2)
    assert eval_power.seed_mean_continue(gap, 1.14, seeds=3) > two
    assert eval_power.seed_mean_continue(0.0, 1.14, seeds=3) < 0.05
    assert eval_power.seed_mean_continue(gap, 0.0) == 1.0
    assert eval_power.seed_mean_continue(0.0, 0.0) == 0.0


def test_training_noise_the_seed_mean_line_ignores_raises_false_continues() -> None:
    quiet = [eval_power.seed_mean_continue(g, 1.14, rho=1.0) for g in (0.0, 2.0)]
    noisy = [eval_power.seed_mean_continue(g, 1.14, rho=1.0, training_sd=1.0) for g in (0.0, 2.0)]
    assert quiet[0] == pytest.approx(0.05, abs=1e-4)
    assert quiet[0] < noisy[0]
    assert noisy[1] < quiet[1]
