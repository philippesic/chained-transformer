from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COLUMNS = (
    "arm",
    "tok/s",
    "prefill tokens",
    "extra prefill vs base",
    "prefill after a switch",
    "switches",
    "wasted tokens",
    "same as base",
    "mixed steps",
)


def arm_name(row: dict[str, Any]) -> str:
    return row["arm"] if not row.get("impl") else f"{row['arm']}/{row['impl']}"


def by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arm_name(row): row for row in report.get("arms", [])}


def cell(value: Any) -> str:
    return "" if value is None else str(value)


def table(report: dict[str, Any]) -> str:
    arms = by_name(report)
    base = arms.get("base", {}).get("prefill_tokens")
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for name, row in arms.items():
        if "error" in row:
            cells = [name, f"failed ({row['error']})"] + [""] * (len(COLUMNS) - 2)
        else:
            prefill = row.get("prefill_tokens")
            extra = None if base is None or prefill is None else prefill - base
            cells = [
                name,
                cell(row.get("tok_per_s")),
                cell(prefill),
                cell(extra),
                cell(row.get("switch_prefill_tokens")),
                cell(row.get("switches")),
                cell(row.get("wasted_tokens")),
                cell(row.get("same_tokens_as_base")),
                cell(row.get("mixed_steps")),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(report: dict[str, Any]) -> list[str]:
    arms = {k: v for k, v in by_name(report).items() if "error" not in v}
    out = []
    base = arms.get("base")
    sep = arms.get("separate")
    chained = [k for k in arms if k.startswith("chained/")]
    if base and sep:
        extra = sep["prefill_tokens"] - base["prefill_tokens"]
        line = f"separate prefilled {extra} more tokens than base"
        if "switch_prefill_tokens" in sep:
            line += (
                f": {sep['switch_prefill_tokens']} on the first call after its "
                f"{sep.get('switches', 0)} switches, the rest from restarting requests every "
                "segment"
            )
        else:
            line += f" over {sep.get('switches', 0)} switches"
        out.append(f"{line}; it threw away {sep.get('wasted_tokens', 0)} generated tokens")
    for name in chained:
        row = arms[name]
        parts = []
        if base:
            parts.append(f"{row['prefill_tokens'] - base['prefill_tokens']} extra prefill tokens")
            parts.append(f"{row['tok_per_s'] / base['tok_per_s']:.2f}x base speed")
        if sep:
            parts.append(f"{row['tok_per_s'] / sep['tok_per_s']:.2f}x separate speed")
        if parts:
            out.append(f"{name}: " + ", ".join(parts))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table of a tools.bench_vllm `all` report: one cache with mode swaps against "
        "separate models that re-prefill"
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    text = table(report) + "\n\n" + "\n".join(f"- {line}" for line in headline(report)) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
