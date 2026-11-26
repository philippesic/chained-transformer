from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerBase

from chained_transformer.data import encode, render
from chained_transformer.generate import TokenTexts
from chained_transformer.model import load_tokenizer
from chained_transformer.modes import Mode, ModeTracker, segment_labels


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
