from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.generation import REFILL_BELOW, _Slot, pick_group
from evals.types import Problem

SCHEDULERS = ("fixed", "grouped")


@dataclass(frozen=True)
class Cost:
    fixed_ms: float
    ms_per_1000: float
    prefill_tokens_per_s: float

    def call_seconds(self, rows: int, width: int, steps: int) -> float:
        cached = rows * (width * steps + steps * (steps + 1) / 2)
        decode = steps * self.fixed_ms + cached * self.ms_per_1000 / 1000
        return decode / 1000 + rows * width / self.prefill_tokens_per_s


@dataclass(frozen=True)
class Row:
    prompt: int
    new: int


def read_rows(paths: list[Path], prompt: int, cap: int, stop_loops_after: int | None) -> list[Row]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            new = min(int(record["new_tokens"]), cap)
            if stop_loops_after is not None and record.get("loop_start") is not None:
                new = min(new, int(record["loop_start"]) + stop_loops_after)
            rows.append(Row(prompt, max(1, new)))
    return rows


def simulate(
    rows: list[Row], scheduler: str, step: int, group_rows: int, cap: int, cost: Cost
) -> dict[str, Any]:
    slots = [_Slot(Problem("sim", str(i), ""), [0] * r.prompt, order=i) for i, r in enumerate(rows)]
    final = {id(s): r.new for s, r in zip(slots, rows, strict=True)}
    queue = list(reversed(slots))
    capacity = step * (max(r.prompt for r in rows) + cap)
    limit = group_rows if scheduler == "grouped" else step
    pool: list[_Slot] = []
    seconds = 0.0
    calls = steps_total = row_steps = 0
    while queue or pool:
        if scheduler == "grouped":
            active, full = pick_group(pool, limit, capacity, cap)
            while not full and queue:
                pool.append(queue.pop())
                active, full = pick_group(pool, limit, capacity, cap)
        else:
            while queue and len(pool) < step:
                pool.append(queue.pop())
            active = list(pool)
        width = max(s.length for s in active)
        left = sorted(final[id(s)] - len(s.new) for s in active)
        room = max(1, capacity // len(active) - width)
        waiting = bool(queue) or len(pool) > len(active)
        low = int((len(active) if scheduler == "grouped" else step) * REFILL_BELOW)
        early = left[len(active) - low - 1] if waiting and len(active) > low else left[-1]
        steps = max(1, min(room, early, left[-1]))
        seconds += cost.call_seconds(len(active), width, steps)
        calls += 1
        steps_total += steps
        for slot in active:
            made = min(steps, final[id(slot)] - len(slot.new))
            row_steps += made
            slot.new += [0] * made
            if len(slot.new) >= final[id(slot)]:
                pool.remove(slot)
    return {
        "scheduler": scheduler,
        "problems": len(rows),
        "hours": round(seconds / 3600, 2),
        "calls": calls,
        "steps": steps_total,
        "new_tokens": row_steps,
        "rows_per_step": round(row_steps / steps_total, 2),
        "tokens_per_s": round(row_steps / seconds, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Eval decode time for a set of response lengths under a step-time model"
    )
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--problems", type=int, help="resample the lengths to this many problems")
    parser.add_argument("--prompt-tokens", type=int, default=500)
    parser.add_argument("--cap", type=int, default=8192)
    parser.add_argument("--step", type=int, default=7, help="rows that fit at full width")
    parser.add_argument("--group-rows", type=int, default=16)
    parser.add_argument("--fixed-ms", type=float, required=True)
    parser.add_argument("--ms-per-1000", type=float, required=True)
    parser.add_argument("--prefill-tokens-per-s", type=float, default=3000.0)
    parser.add_argument("--stop-loops-after", type=int)
    parser.add_argument("--schedulers", nargs="+", choices=SCHEDULERS, default=list(SCHEDULERS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = read_rows(args.responses, args.prompt_tokens, args.cap, args.stop_loops_after)
    if args.problems:
        rows = random.Random(args.seed).choices(rows, k=args.problems)
    cost = Cost(args.fixed_ms, args.ms_per_1000, args.prefill_tokens_per_s)
    for scheduler in args.schedulers:
        print(json.dumps(simulate(rows, scheduler, args.step, args.group_rows, args.cap, cost)))


if __name__ == "__main__":
    main()
