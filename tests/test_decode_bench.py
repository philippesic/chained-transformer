from __future__ import annotations

import json
from pathlib import Path

from transformers import Qwen3ForCausalLM

from tools import decode_bench


def test_shapes_parse_and_lines_fit() -> None:
    assert decode_bench.parse_shape("7x4096") == (7, 4096)
    fit = decode_bench.fit_line([(1000, 12.0), (3000, 16.0), (5000, 20.0)])
    assert fit == {"fixed_ms": 10.0, "ms_per_1000_cache_tokens": 2.0}
    assert decode_bench.fit_line([(1000, 12.0)]) is None


def test_bench_compares_attention_choices_on_the_same_tokens(
    tiny_model: Qwen3ForCausalLM, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    tiny_model.config.bos_token_id = 0
    tiny_model.save_pretrained(base)
    out = tmp_path / "bench.json"
    argv = ["--base", str(base), "--device", "cpu", "--shapes", "3x20", "2x40"]
    decode_bench.main([*argv, "--steps", "6", "--out", str(out)])
    report = json.loads(out.read_text())
    assert [r["attention"] for r in report["shapes"]] == ["auto", "auto", "lean", "lean"]
    lean = [r for r in report["shapes"] if r["attention"] == "lean"]
    assert all(r["same_argmax_rate"] == 1.0 for r in lean)
    assert all(r["max_abs_logit_diff_vs_first"] < 1e-3 for r in lean)
    assert set(report["fits"]) == {"auto", "lean"}
