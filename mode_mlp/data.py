from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfFileSystem
from transformers import PreTrainedTokenizerBase

from mode_mlp.generate import TokenTexts
from mode_mlp.model import BASE_MODEL, load_tokenizer
from mode_mlp.modes import MODE_NAMES, THINK_CLOSE, segment_labels

DATASET = "open-thoughts/OpenThoughts3-1.2M"
DATASET_SHARDS = 120
ROLE_NAMES = {"human": "user", "gpt": "assistant"}
TRUNCATION_TOKENS = 16_000


@dataclass(frozen=True)
class Rendered:
    text: str
    assistant_start: int


def render(conversations: list[dict[str, str]]) -> Rendered | None:
    roles = [ROLE_NAMES.get(turn["from"]) for turn in conversations]
    if roles != ["user", "assistant"]:
        return None
    user, assistant = (turn["value"] for turn in conversations)
    head = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return Rendered(head + assistant + "<|im_end|>\n", len(head))


def encode(
    rendered: Rendered, tokenizer: PreTrainedTokenizerBase, texts: TokenTexts
) -> dict[str, list[int]]:
    enc = tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    loss_mask = [int(start >= rendered.assistant_start) for start, _ in enc["offset_mapping"]]
    return {"input_ids": ids, "mode_ids": texts.modes(ids), "loss_mask": loss_mask}


def shard_files(num_shards: int) -> list[str]:
    step = DATASET_SHARDS // num_shards
    return [
        f"datasets/{DATASET}/data/train-{i:05d}-of-{DATASET_SHARDS:05d}.parquet"
        for i in range(0, step * num_shards, step)
    ]


def stream_rows(rows: int, num_shards: int) -> Iterator[dict[str, Any]]:
    fs = HfFileSystem()
    per_shard = -(-rows // num_shards)
    taken = 0
    for path in shard_files(num_shards):
        want = min(per_shard, rows - taken)
        with fs.open(path, "rb", block_size=8 << 20) as handle:
            batches = pq.ParquetFile(handle).iter_batches(batch_size=64)
            for batch in batches:
                for row in batch.to_pylist()[:want]:
                    yield row
                    want -= 1
                    taken += 1
                if want == 0:
                    break
        if taken >= rows:
            return


FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "mode_ids": Sequence(Value("int8")),
        "loss_mask": Sequence(Value("int8")),
        "domain": Value("string"),
        "source": Value("string"),
        "has_think_close": Value("bool"),
        "assistant_tokens": Value("int32"),
        "complete": Value("bool"),
    }
)


@dataclass
class SliceStats:
    rows: int = 0
    skipped_not_single_turn: int = 0
    missing_think_close: int = 0
    domains: Counter[str] = field(default_factory=Counter)
    complete_by_domain: Counter[str] = field(default_factory=Counter)
    loss_tokens_by_label: Counter[str] = field(default_factory=Counter)
    complete_loss_tokens_by_label: Counter[str] = field(default_factory=Counter)
    lengths: list[int] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.rows += 1
        self.missing_think_close += not record["has_think_close"]
        self.domains[record["domain"]] += 1
        self.lengths.append(len(record["input_ids"]))
        labels = Counter(
            MODE_NAMES[label]
            for label, keep in zip(
                segment_labels(record["mode_ids"]), record["loss_mask"], strict=True
            )
            if keep
        )
        self.loss_tokens_by_label.update(labels)
        if record["complete"]:
            self.complete_by_domain[record["domain"]] += 1
            self.complete_loss_tokens_by_label.update(labels)

    def report(self, rows: int, num_shards: int, tokenizer_name: str) -> dict[str, Any]:
        lengths = sorted(self.lengths)
        return {
            "dataset": DATASET,
            "tokenizer": tokenizer_name,
            "requested_rows": rows,
            "shards": shard_files(num_shards),
            "rows": self.rows,
            "skipped_not_single_turn": self.skipped_not_single_turn,
            "missing_think_close": self.missing_think_close,
            "complete_rule": f"has </think> and fewer than {TRUNCATION_TOKENS} assistant tokens",
            "complete_rows": sum(self.complete_by_domain.values()),
            "domains": dict(self.domains),
            "complete_by_domain": dict(self.complete_by_domain),
            "loss_tokens_by_label": dict(self.loss_tokens_by_label),
            "complete_loss_tokens_by_label": dict(self.complete_loss_tokens_by_label),
            "tokens_total": sum(lengths),
            "tokens_median": lengths[len(lengths) // 2] if lengths else 0,
            "tokens_max": lengths[-1] if lengths else 0,
        }


def labelled_rows(
    rows: int, num_shards: int, tokenizer_name: str, stats: SliceStats
) -> Iterator[dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_name)
    texts = TokenTexts.from_tokenizer(tokenizer)
    for row in stream_rows(rows, num_shards):
        rendered = render(row["conversations"])
        if rendered is None:
            stats.skipped_not_single_turn += 1
            continue
        encoded = encode(rendered, tokenizer, texts)
        has_think_close = THINK_CLOSE in rendered.text
        assistant_tokens = sum(encoded["loss_mask"])
        record = {
            **encoded,
            "domain": row["domain"],
            "source": row["source"],
            "has_think_close": has_think_close,
            "assistant_tokens": assistant_tokens,
            "complete": has_think_close and assistant_tokens < TRUNCATION_TOKENS,
        }
        stats.add(record)
        yield record


def build(rows: int, num_shards: int, out: Path, tokenizer_name: str) -> dict[str, Any]:
    stats = SliceStats()
    with tempfile.TemporaryDirectory(dir=out.parent) as cache:
        ds = Dataset.from_generator(
            labelled_rows,
            gen_kwargs={
                "rows": rows,
                "num_shards": num_shards,
                "tokenizer_name": tokenizer_name,
                "stats": stats,
            },
            features=FEATURES,
            cache_dir=cache,
            keep_in_memory=False,
        )
        ds.save_to_disk(str(out))
    report = stats.report(rows, num_shards, tokenizer_name)
    (out / "stats.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labelled OpenThoughts3 slice")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/data/ot3-slice-4k"))
    parser.add_argument("--tokenizer", default=BASE_MODEL)
    args = parser.parse_args()
    stats = build(args.rows, args.shards, args.out, args.tokenizer)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
