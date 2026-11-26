from __future__ import annotations

import subprocess
import sys

STEPS: list[tuple[str, list[str], str]] = [
    ("lint", ["ruff", "check", "."], "fix the reported lines, or run `ruff check --fix .`"),
    ("format", ["ruff", "format", "--check", "."], "run `ruff format .`"),
    ("types", ["mypy"], "fix the annotated types; do not add blanket ignores"),
    ("tests", ["pytest", "-q", "-p", "no:cacheprovider"], "fix the failing behavior"),
]


def main() -> int:
    failed = []
    for name, cmd, repair in STEPS:
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(f"{name} failed: {repair}")
    print("== scope: CPU only; GPU behavior needs `python -m tools.gpu_smoke`")
    for line in failed:
        print(line, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
