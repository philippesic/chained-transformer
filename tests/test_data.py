from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase

from chained_transformer.data import (
    MixBuild,
    RowGroup,
    build_mix,
    encode,
    render,
    sample_row_groups,
)
from chained_transformer.generate import TokenTexts
from chained_transformer.mix import CANDIDATES
from chained_transformer.model import load_tokenizer
from chained_transformer.modes import MODE_NAMES, Mode, ModeTracker, segment_labels


@pytest.fixture(scope="module")
def texts_and_tokenizer() -> tuple[TokenTexts, PreTrainedTokenizerBase]:
    try:
        tok = load_tokenizer()
    except OSError as exc:
        pytest.skip(
            f"cannot load the Qwen3 tokenizer (not cached, Hugging Face unreachable?): {exc}"
        )
    return TokenTexts.from_tokenizer(tok), tok


def test_render_rejects_multi_turn() -> None:
    turns = [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}] * 2
    assert render(turns) is None


def test_encoded_labels_follow_segments(
    texts_and_tokenizer: tuple[TokenTexts, PreTrainedTokenizerBase],
) -> None:
    texts, tok = texts_and_tokenizer
    answer = "<think>\nplan\n```python\nprint(1)\n```\n</think>\n\nSo:\n```\nx\n```\nDone."
    rendered = render([{"from": "human", "value": "Q?"}, {"from": "gpt", "value": answer}])
    assert rendered is not None
    enc = encode(rendered, tok, texts)
    ids, modes, mask = enc["input_ids"], enc["mode_ids"], enc["loss_mask"]
    assert texts[ids[mask.index(1)]] == "<think>"
    assert all(m == 0 for m in mask[: mask.index(1)])
    labelled = {
        Mode(label): "".join(
            texts[t] for t, lab in zip(ids, segment_labels(modes), strict=True) if lab == label
        )
        for label in Mode
    }
    assert "print(1)" in labelled[Mode.CODE] and "x\n" in labelled[Mode.CODE]
    assert "plan" in labelled[Mode.THINK] and "Done." in labelled[Mode.ANSWER]
    assert "So:" in labelled[Mode.ANSWER] and "So:" not in labelled[Mode.THINK]


def test_think_close_is_one_special_token(
    texts_and_tokenizer: tuple[TokenTexts, PreTrainedTokenizerBase],
) -> None:
    texts, tok = texts_and_tokenizer
    ids = tok("a</think>b", add_special_tokens=False)["input_ids"]
    assert "</think>" in [texts[i] for i in ids]
    tracker = ModeTracker()
    tracker.feed("<|im_start|>assistant\n")
    assert [tracker.feed(texts[i]) for i in ids][-1] == Mode.ANSWER


def test_row_group_sampling_is_unique_sorted_and_respects_the_download_cap() -> None:
    groups = [RowGroup(f"shard-{s}", i, 100, 10) for s in range(5) for i in range(10)]
    picked = sample_row_groups(groups, 20, seed=0)
    assert len(picked) == len(set(picked)) == 20
    assert picked == sorted(picked, key=lambda g: (g.path, g.index))
    capped = sample_row_groups(groups, 20, seed=0, max_bytes=55)
    assert len(capped) == 5
    assert set(capped) <= set(sample_row_groups(groups, 20, seed=0))


def _trace(domain: str, *answers: str) -> dict[str, Any]:
    turns = [
        t for a in answers for t in ({"from": "human", "value": "Q?"}, {"from": "gpt", "value": a})
    ]
    return {"conversations": turns, "domain": domain, "source": "test"}


@pytest.mark.usefixtures("texts_and_tokenizer")
def test_build_mix_keeps_complete_rows_of_every_domain(tmp_path: Path) -> None:
    answer = "<think>\nplan {i}\n```python\nprint({i})\n```\n</think>\n\nDone {i}."
    rows = [_trace(d, answer.format(i=i)) for d in ("code", "math", "science") for i in range(2)]
    rows += [_trace("math", "<think>\ncut off"), _trace("math", "a", "b")]
    out = tmp_path / "mix"
    report = build_mix(MixBuild(out=out), source=lambda: iter(rows))
    assert report["sample_summary"]["rows"] == 7
    assert report["sample_summary"]["complete_rows"] == 6
    assert set(report["policies"]) == set(CANDIDATES)
    ds = load_from_disk(str(out))
    assert len(ds) == report["selected"]["rows"] >= 3
    assert all(ds["complete"]) and set(ds["domain"]) == {"code", "math", "science"}
    samples = [json.loads(line) for line in (out / "sample.jsonl").read_text().splitlines()]
    assert len(samples) == 6
    assert {run["label"] for run in samples[0]["runs"]} == set(MODE_NAMES)
