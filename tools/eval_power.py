from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from statistics import NormalDist

FULL_SIZES = {"math500": 500, "humanevalplus": 164, "livecodebench": 880}
GAP_POINTS = 2.0
ALPHA = 0.05
POWER = 0.8
SEEDS = 2
ASSUMED_DISCORDANCE = 0.10
MIN_SAMPLE = 100


def target_seed_se(
    gap: float = GAP_POINTS, alpha: float = ALPHA, power: float = POWER, seeds: int = SEEDS
) -> float:
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return gap / z * math.sqrt(seeds)


def seed_se(discordance: Mapping[str, float], sizes: Mapping[str, int]) -> float:
    k = len(sizes)
    return 100.0 / k * math.sqrt(sum(discordance[b] / n for b, n in sizes.items()))


def power_at(se: float, gap: float = GAP_POINTS, alpha: float = ALPHA, seeds: int = SEEDS) -> float:
    z = NormalDist().inv_cdf(1 - alpha)
    return NormalDist().cdf(gap / (se / math.sqrt(seeds)) - z)


def required_sizes(
    discordance: Mapping[str, float],
    available: Mapping[str, int],
    target: float | None = None,
    floor: int = MIN_SAMPLE,
) -> dict[str, int]:
    se = target_seed_se() if target is None else target
    budget = (se * len(available) / 100.0) ** 2
    sizes: dict[str, int] = {}
    free = dict(available)
    while free:
        spent = sum(discordance[b] / n for b, n in sizes.items())
        left = budget - spent
        roots = {b: math.sqrt(max(discordance[b], 1e-9)) for b in free}
        if left <= 0:
            sizes.update(free)
            break
        want = {b: r * sum(roots.values()) / left for b, r in roots.items()}
        capped = {b for b, n in want.items() if n >= free[b]}
        if not capped:
            sizes.update({b: max(min(floor, free[b]), math.ceil(n)) for b, n in want.items()})
            break
        for b in capped:
            sizes[b] = free.pop(b)
    return {b: sizes[b] for b in available}


def discordance_of(a: Mapping[str, bool], b: Mapping[str, bool]) -> tuple[float, int]:
    shared = sorted(set(a) & set(b))
    if not shared:
        return 0.0, 0
    return sum(a[t] != b[t] for t in shared) / len(shared), len(shared)


def table(values: list[float], available: Mapping[str, int]) -> list[str]:
    target = target_seed_se()
    full_n = sum(available.values())
    lines = [
        "| discordance | problems needed (math500 / humanevalplus / livecodebench) | share of"
        " full suite | per-seed SE, full suite | power, full suite |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for d in values:
        disc = dict.fromkeys(available, d)
        sizes = required_sizes(disc, available, target, floor=0)
        reached = seed_se(disc, sizes) <= target + 1e-9
        need = " / ".join(str(sizes[b]) for b in available) if reached else "not reachable"
        share = f"{sum(sizes.values()) / full_n:.0%}" if reached else "100%"
        full = seed_se(disc, available)
        lines.append(f"| {d:.2f} | {need} | {share} | {full:.2f} | {power_at(full):.0%} |")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Eval problems needed for the kill rule to see a 2-point gap"
    )
    parser.add_argument("--discordance", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3])
    args = parser.parse_args(argv)
    target = target_seed_se()
    print(
        f"one-sided paired test, alpha {ALPHA}, power {POWER:.0%}, gap {GAP_POINTS} points, "
        f"{SEEDS} seeds: per-seed SE of the margin must be at most {target:.2f} points"
    )
    print("\n".join(table(args.discordance, FULL_SIZES)))


if __name__ == "__main__":
    main()
