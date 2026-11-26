from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def first_difference(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def stuck(tokens: list[int], window: int) -> bool:
    head = tokens[:window]
    return len(head) == window and len(set(head)) == 1


def compare(a: list[list[int]], b: list[list[int]], window: int) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError(f"{len(a)} outputs against {len(b)}: compare runs of the same prompts")
    firsts = sorted(first_difference(x, y) for x, y in zip(a, b, strict=True))
    return {
        "prompts": len(a),
        "window": window,
        "same_window": sum(f >= window for f in firsts),
        "same_whole": sum(x == y for x, y in zip(a, b, strict=True)),
        "median_first_difference": firsts[len(firsts) // 2] if firsts else None,
        "stuck_a": sum(stuck(x, window) for x in a),
        "stuck_b": sum(stuck(x, window) for x in b),
    }


def summary(result: dict[str, Any]) -> str:
    return (
        f"{result['same_window']} of {result['prompts']} prompts agree on the first "
        f"{result['window']} tokens ({result['same_whole']} on every token); "
        f"{result['stuck_a']} and {result['stuck_b']} outputs repeat one token from the start"
    )


def read_outputs(path: Path) -> list[list[int]]:
    data = json.loads(path.read_text())
    if "outputs" not in data:
        raise SystemExit(
            f"{path} has no outputs: pass a row file that tools.bench_vllm writes, from "
            "`arm --json-out` or the row_*.json files that `all` leaves in --dir"
        )
    return [list(map(int, row)) for row in data["outputs"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the generated tokens of two tools.bench_vllm runs on the same prompts"
    )
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "a": str(args.a),
        "b": str(args.b),
        **compare(read_outputs(args.a), read_outputs(args.b), args.window),
    }
    print(summary(result))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
