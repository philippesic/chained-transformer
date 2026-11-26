from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from chained_transformer.modes import MODE_NAMES, Mode, segment_labels

LENGTH_CAPS: tuple[int, ...] = (4096, 8192, 12288, 16384, 20480)
SWITCH_BINS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 10)


@dataclass(frozen=True)
class RowProfile:
    domain: str
    total_tokens: int
    assistant_tokens: int
    complete: bool
    label_tokens: tuple[int, ...]
    switches: int
    fences_in_think: int
    fences_in_answer: int
    answer_start: float | None
    code_starts: tuple[float, ...]

    @property
    def non_think_tokens(self) -> int:
        return sum(self.label_tokens) - self.label_tokens[Mode.THINK]


def profile(
    mode_ids: Sequence[int], loss_mask: Sequence[int], domain: str, complete: bool
) -> RowProfile:
    labels = segment_labels(list(mode_ids))
    counts = [0] * len(MODE_NAMES)
    switches = 0
    fences_in_think = 0
    fences_in_answer = 0
    answer_start: float | None = None
    code_starts: list[float] = []
    assistant = [label for label, keep in zip(labels, loss_mask, strict=True) if keep]
    n = len(assistant)
    for pos, label in enumerate(assistant):
        counts[label] += 1
        if pos == 0:
            continue
        prev = assistant[pos - 1]
        if label == prev:
            continue
        switches += 1
        if label == Mode.ANSWER and answer_start is None and prev == Mode.THINK:
            answer_start = pos / n
        if label == Mode.CODE:
            code_starts.append(pos / n)
            if prev == Mode.THINK:
                fences_in_think += 1
            else:
                fences_in_answer += 1
    return RowProfile(
        domain=domain,
        total_tokens=len(mode_ids),
        assistant_tokens=n,
        complete=complete,
        label_tokens=tuple(counts),
        switches=switches,
        fences_in_think=fences_in_think,
        fences_in_answer=fences_in_answer,
        answer_start=answer_start,
        code_starts=tuple(code_starts),
    )


def _quantiles(
    values: Sequence[float], qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        out[f"p{int(q * 100)}"] = round(float(ordered[idx]), 4)
    return out


def _bin(value: int, bins: Sequence[int]) -> str:
    label = f"{bins[-1]}+"
    for lo, hi in zip(bins, bins[1:], strict=False):
        if lo <= value < hi:
            label = str(lo) if hi - lo == 1 else f"{lo}-{hi - 1}"
            break
    return label


def label_shares(rows: Iterable[RowProfile]) -> dict[str, object]:
    totals = [0] * len(MODE_NAMES)
    count = 0
    for r in rows:
        count += 1
        for i, n in enumerate(r.label_tokens):
            totals[i] += n
    tokens = sum(totals)
    return {
        "rows": count,
        "loss_tokens": tokens,
        "tokens": {MODE_NAMES[i]: totals[i] for i in range(len(totals))},
        "share": {
            MODE_NAMES[i]: round(totals[i] / tokens, 4) if tokens else 0.0
            for i in range(len(totals))
        },
    }


def summarize(rows: Sequence[RowProfile], caps: Sequence[int] = LENGTH_CAPS) -> dict[str, object]:
    complete = [r for r in rows if r.complete]
    by_domain = Counter(r.domain for r in rows)
    complete_by_domain = Counter(r.domain for r in complete)
    switch_hist = Counter(_bin(r.switches, SWITCH_BINS) for r in complete)
    code_starts = [s for r in complete for s in r.code_starts]
    length_table = {}
    for cap in caps:
        kept = [r for r in complete if r.total_tokens <= cap]
        length_table[str(cap)] = {
            **label_shares(kept),
            "rows_kept_of_complete": round(len(kept) / len(complete), 4) if complete else 0.0,
            "by_domain": dict(Counter(r.domain for r in kept)),
        }
    return {
        "rows": len(rows),
        "complete_rows": len(complete),
        "domains": dict(by_domain),
        "complete_by_domain": dict(complete_by_domain),
        "all_rows": label_shares(rows),
        "complete": label_shares(complete),
        "complete_by_domain_shares": {
            d: label_shares(r for r in complete if r.domain == d)
            for d in sorted(complete_by_domain)
        },
        "switches_per_complete_row": {
            "mean": round(sum(r.switches for r in complete) / len(complete), 3) if complete else 0,
            "histogram": dict(sorted(switch_hist.items())),
        },
        "code_blocks": {
            "in_think": sum(r.fences_in_think for r in complete),
            "in_answer": sum(r.fences_in_answer for r in complete),
            "rows_with_code": sum(1 for r in complete if r.label_tokens[Mode.CODE] > 0),
            "start_position_quantiles": _quantiles(code_starts),
        },
        "answer_start_quantiles": _quantiles(
            [r.answer_start for r in complete if r.answer_start is not None]
        ),
        "complete_total_tokens_quantiles": _quantiles([r.total_tokens for r in complete]),
        "complete_answer_tokens_quantiles": _quantiles(
            [r.label_tokens[Mode.ANSWER] for r in complete]
        ),
        "length_caps": length_table,
    }


@dataclass(frozen=True)
class MixPolicy:
    max_tokens: int = 16384
    domain_token_shares: tuple[tuple[str, float], ...] | None = (
        ("code", 1.0),
        ("math", 1.0),
        ("science", 1.0),
    )
    complete_only: bool = True

    def eligible(self, row: RowProfile) -> bool:
        if self.complete_only and not row.complete:
            return False
        if self.domain_token_shares is not None and row.domain not in dict(
            self.domain_token_shares
        ):
            return False
        return row.total_tokens <= self.max_tokens


def select(
    rows: Sequence[RowProfile], policy: MixPolicy, token_budget: int | None, seed: int
) -> list[int]:
    rng = random.Random(seed)
    eligible = [i for i, r in enumerate(rows) if policy.eligible(r)]
    rng.shuffle(eligible)
    budget = (
        token_budget if token_budget is not None else sum(rows[i].total_tokens for i in eligible)
    )
    if policy.domain_token_shares is None:
        chosen, spent = [], 0
        for i in eligible:
            if spent + rows[i].total_tokens > budget:
                break
            chosen.append(i)
            spent += rows[i].total_tokens
        return chosen
    pools: dict[str, list[int]] = {}
    for i in eligible:
        pools.setdefault(rows[i].domain, []).append(i)
    shares = {d: w for d, w in policy.domain_token_shares if w > 0 and d in pools}
    used = dict.fromkeys(shares, 0)
    cursor = dict.fromkeys(shares, 0)
    chosen = []
    spent = 0
    while shares:
        domain = min(shares, key=lambda d: used[d] / shares[d])
        if cursor[domain] >= len(pools[domain]):
            break
        idx = pools[domain][cursor[domain]]
        cursor[domain] += 1
        if spent + rows[idx].total_tokens > budget:
            break
        chosen.append(idx)
        spent += rows[idx].total_tokens
        used[domain] += rows[idx].total_tokens
    return chosen


CANDIDATES: dict[str, MixPolicy] = {
    "natural-16k": MixPolicy(16384, None),
    "equal-domains-16k": MixPolicy(16384),
    "code-heavy-16k": MixPolicy(16384, (("code", 2.0), ("math", 1.0), ("science", 1.0))),
    "natural-8k": MixPolicy(8192, None),
    "equal-domains-8k": MixPolicy(8192),
}


def compare_policies(
    rows: Sequence[RowProfile], policies: dict[str, MixPolicy], seed: int
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, policy in policies.items():
        picked = [rows[i] for i in select(rows, policy, None, seed)]
        shares = None if policy.domain_token_shares is None else dict(policy.domain_token_shares)
        out[name] = {
            **label_shares(picked),
            "by_domain": dict(Counter(r.domain for r in picked)),
            "policy": {"max_tokens": policy.max_tokens, "domain_token_shares": shares},
        }
    return out
