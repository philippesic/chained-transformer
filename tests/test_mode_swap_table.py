from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.mode_swap_table import headline, table

REPORT = {
    "dir": "x",
    "max_tokens": 8,
    "arms": [
        {"arm": "base", "impl": None, "tok_per_s": 100.0, "prefill_tokens": 40},
        {
            "arm": "chained",
            "impl": "triton",
            "tok_per_s": 90.0,
            "prefill_tokens": 40,
            "same_tokens_as_base": 4,
            "mixed_steps": 3,
        },
        {"arm": "chained", "impl": "select", "error": "exit 1"},
        {
            "arm": "separate",
            "impl": None,
            "tok_per_s": 30.0,
            "prefill_tokens": 140,
            "switch_prefill_tokens": 70,
            "switches": 5,
            "wasted_tokens": 12,
            "same_tokens_as_base": 4,
        },
    ],
}


def test_table_shows_extra_prefill_and_failures() -> None:
    lines = table(REPORT).splitlines()
    assert lines[0].startswith("| arm | tok/s | prefill tokens | extra prefill vs base |")
    assert lines[2] == "| base | 100.0 | 40 | 0 |  |  |  |  |  |"
    assert lines[3] == "| chained/triton | 90.0 | 40 | 0 |  |  |  | 4 | 3 |"
    assert lines[4].startswith("| chained/select | failed (exit 1) |")
    assert lines[4].count("|") == lines[0].count("|")
    assert lines[5] == "| separate | 30.0 | 140 | 100 | 70 | 5 | 12 | 4 |  |"


def test_headline_compares_chained_with_base_and_separate() -> None:
    lines = headline(REPORT)
    assert lines[0] == (
        "separate prefilled 100 more tokens than base: 70 on the first call after its 5 "
        "switches, the rest from restarting requests every segment; it threw away 12 "
        "generated tokens"
    )
    assert lines[1] == (
        "chained/triton: 0 extra prefill tokens, 0.90x base speed, 3.00x separate speed"
    )
    assert len(lines) == 2


def test_cli_writes_markdown(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(json.dumps(REPORT))
    out = tmp_path / "table.md"
    done = subprocess.run(
        [sys.executable, "-m", "tools.mode_swap_table", str(tmp_path / "report.json")]
        + ["--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == out.read_text()
    assert "3.00x separate speed" in done.stdout


def test_headline_reads_reports_without_switch_prefill() -> None:
    old = json.loads(json.dumps(REPORT))
    del old["arms"][3]["switch_prefill_tokens"]
    assert headline(old)[0] == (
        "separate prefilled 100 more tokens than base over 5 switches; it threw away 12 "
        "generated tokens"
    )
