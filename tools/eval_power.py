from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from statistics import NormalDist

FULL_SIZES = {"math500": 500, "humanevalplus": 164, "livecodebench": 341}
GAP_POINTS = 2.0
ALPHA = 0.05
POWER = 0.8
SEEDS = 2
ASSUMED_DISCORDANCE = 0.10
MIN_SAMPLE = 100
MARGIN_CORRELATION = 0.5


def target_seed_se(
    gap: float = GAP_POINTS, alpha: float = ALPHA, power: float = POWER, seeds: int = SEEDS
) -> float:
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return gap / z * math.sqrt(seeds)


def seed_se(discordance: Mapping[str, float], sizes: Mapping[str, int]) -> float:
    k = len(sizes)
    return 100.0 / k * math.sqrt(sum(discordance[b] / n for b, n in sizes.items()))


def power_at(se: float, gap: float = GAP_POINTS, alpha: float = ALPHA, seeds: int = SEEDS) -> float:
    if se <= 0:
        return 1.0 if gap > 0 else 0.0
    z = NormalDist().inv_cdf(1 - alpha)
    return NormalDist().cdf(gap / (se / math.sqrt(seeds)) - z)


def both_clear(gap: float, sd: float, line: float, rho: float, steps: int = 4000) -> float:
    if sd <= 0:
        return 1.0 if gap >= line else 0.0
    unit = NormalDist()
    h = (line - gap) / sd
    if rho >= 1:
        return 1 - unit.cdf(h)
    spread = math.sqrt(1 - rho * rho)
    top = max(h, 0.0) + 10.0
    width = (top - h) / steps

    def f(z: float) -> float:
        return unit.pdf(z) * (1 - unit.cdf((h - rho * z) / spread))

    total = f(h) + f(top) + sum((4 if i % 2 else 2) * f(h + i * width) for i in range(1, steps))
    return min(1.0, max(0.0, total * width / 3))


def kill_rule_odds(
    gap: float,
    se: float,
    seeds: int = SEEDS,
    rho: float = MARGIN_CORRELATION,
    training_sd: float = 0.0,
    line: float = GAP_POINTS,
) -> dict[str, float]:
    p = both_clear(gap, math.hypot(se, training_sd), line, rho)
    keep, kill = p**seeds, (1 - p) ** seeds
    return {"seed_passes": p, "continue": keep, "mixed": max(0.0, 1 - keep - kill), "kill": kill}


def seed_mean_line(se: float, seeds: int = SEEDS, alpha: float = ALPHA) -> float:
    return NormalDist().inv_cdf(1 - alpha) * se / math.sqrt(seeds)


def seed_mean_continue(
    gap: float,
    se: float,
    seeds: int = SEEDS,
    rho: float = MARGIN_CORRELATION,
    training_sd: float = 0.0,
    alpha: float = ALPHA,
) -> float:
    if se <= 0 and training_sd <= 0:
        return 1.0 if gap > 0 else 0.0
    spread = math.hypot(se, training_sd) / math.sqrt(seeds)
    return both_clear(gap, spread, seed_mean_line(se, seeds, alpha), rho)


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


def odds_table(
    gaps: list[float], se: float, seed_counts: list[int], rho: float, training_sd: float
) -> list[str]:
    head = " | ".join(f"{k} seeds: continue / mixed / kill" for k in seed_counts)
    means = " | ".join(f"seed mean, {k} seeds: continue" for k in seed_counts)
    lines = [f"| true gap | one seed clears | {head} | {means} |"]
    lines.append("| ---: | ---: |" + " --- |" * len(seed_counts) + " ---: |" * len(seed_counts))
    for g in gaps:
        cells = []
        for k in seed_counts:
            o = kill_rule_odds(g, se, k, rho, training_sd)
            cells.append(f"{o['continue']:.0%} / {o['mixed']:.0%} / {o['kill']:.0%}")
        cells += [f"{seed_mean_continue(g, se, k, rho, training_sd):.0%}" for k in seed_counts]
        p = kill_rule_odds(g, se, 1, rho, training_sd)["seed_passes"]
        lines.append(f"| {g:g} | {p:.0%} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Eval problems needed for the kill rule to see a 2-point gap"
    )
    parser.add_argument("--discordance", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument(
        "--odds", action="store_true", help="odds of each verdict at a per-seed SE, by true gap"
    )
    parser.add_argument("--se", type=float, help="per-seed SE of the margin; default the target")
    parser.add_argument("--gaps", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--rho", type=float, default=MARGIN_CORRELATION)
    parser.add_argument("--training-sd", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.odds:
        se = args.se if args.se is not None else target_seed_se()
        print(
            f"continue needs full - sft and full - moe both >= {GAP_POINTS} points on every seed; "
            f"per-seed SE {se:.2f}, correlation {args.rho} between the two margins, "
            f"seed-to-seed training SD {args.training_sd}"
        )
        lines = ", ".join(f"{seed_mean_line(se, k):.2f} points with {k} seeds" for k in args.seeds)
        print(
            f"seed mean (not what pilot compare applies): continue when the seed mean of both "
            f"margins clears a one-sided {ALPHA} test at this SE, {lines}"
        )
        print("\n".join(odds_table(args.gaps, se, args.seeds, args.rho, args.training_sd)))
        return
    target = target_seed_se()
    print(
        f"one-sided paired test, alpha {ALPHA}, power {POWER:.0%}, gap {GAP_POINTS} points, "
        f"{SEEDS} seeds: per-seed SE of the margin must be at most {target:.2f} points"
    )
    print("\n".join(table(args.discordance, FULL_SIZES)))


if __name__ == "__main__":
    main()
