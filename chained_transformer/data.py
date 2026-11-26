from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfFileSystem
from transformers import PreTrainedTokenizerBase

from chained_transformer.generate import TokenTexts
from chained_transformer.mix import CANDIDATES, RowProfile, compare_policies, profile, select
from chained_transformer.mix import label_shares as mix_label_shares
from chained_transformer.mix import summarize as mix_summarize
from chained_transformer.model import BASE_MODEL, load_tokenizer
from chained_transformer.modes import MODE_NAMES, THINK_CLOSE, segment_labels

DATASET = "open-thoughts/OpenThoughts3-1.2M"
DATASET_SHARDS = 120
ROLE_NAMES = {"human": "user", "gpt": "assistant"}
TRUNCATION_TOKENS = 16_000
COLUMNS = ["conversations", "domain", "source"]
SAMPLE_ROWS = 12
SNIPPET_CHARS = 160


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


@dataclass(frozen=True)
class RowGroup:
    path: str
    index: int
    rows: int


def all_row_groups(fs: Any) -> list[RowGroup]:
    groups = []
    for path in shard_files(DATASET_SHARDS):
        with fs.open(path, "rb", block_size=1 << 20) as handle:
            meta = pq.ParquetFile(handle).metadata
            groups += [
                RowGroup(path, i, meta.row_group(i).num_rows) for i in range(meta.num_row_groups)
            ]
    return groups


def sample_row_groups(groups: list[RowGroup], count: int, seed: int) -> list[RowGroup]:
    rng = random.Random(seed)
    picked = rng.choices(groups, weights=[g.rows for g in groups], k=min(count, len(groups)) * 4)
    unique: list[RowGroup] = []
    for g in picked:
        if g not in unique:
            unique.append(g)
        if len(unique) == count:
            break
    return sorted(unique, key=lambda g: (g.path, g.index))


def stream_sampled_rows(
    groups: list[RowGroup], rows_per_group: int, fs: Any | None = None
) -> Iterator[dict[str, Any]]:
    fs = fs or HfFileSystem()
    for group in groups:
        want = rows_per_group
        with fs.open(group.path, "rb", block_size=8 << 20) as handle:
            batches = pq.ParquetFile(handle).iter_batches(
                batch_size=64, row_groups=[group.index], columns=COLUMNS
            )
            for batch in batches:
                for row in batch.to_pylist()[:want]:
                    yield row
                    want -= 1
                if want == 0:
                    break


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


@dataclass
class Collector:
    profiles: list[RowProfile] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)


def labelled_rows(
    rows: int,
    num_shards: int,
    tokenizer_name: str,
    stats: SliceStats,
    source: Callable[[], Iterator[dict[str, Any]]] | None = None,
    collector: Collector | None = None,
) -> Iterator[dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_name)
    texts = TokenTexts.from_tokenizer(tokenizer)
    stream = source() if source is not None else stream_rows(rows, num_shards)
    for row in stream:
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
        if collector is not None:
            collector.profiles.append(
                profile(record["mode_ids"], record["loss_mask"], row["domain"], record["complete"])
            )
            if len(collector.samples) < SAMPLE_ROWS and record["complete"]:
                collector.samples.append(sample_view(record, texts))
        yield record


def sample_view(record: dict[str, Any], texts: TokenTexts) -> dict[str, Any]:
    labels = segment_labels(record["mode_ids"])
    runs: list[dict[str, Any]] = []
    for pos, (token, label, keep) in enumerate(
        zip(record["input_ids"], labels, record["loss_mask"], strict=True)
    ):
        if not keep:
            continue
        if not runs or runs[-1]["label"] != MODE_NAMES[label]:
            runs.append({"label": MODE_NAMES[label], "start": pos, "tokens": 0, "text": ""})
        run = runs[-1]
        run["tokens"] += 1
        if len(run["text"]) < SNIPPET_CHARS:
            run["text"] += texts[token]
    for run in runs:
        run["text"] = run["text"][:SNIPPET_CHARS]
    return {
        "domain": record["domain"],
        "source": record["source"],
        "total_tokens": len(record["input_ids"]),
        "assistant_tokens": record["assistant_tokens"],
        "runs": runs,
    }


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


@dataclass(frozen=True)
class MixBuild:
    out: Path
    groups: int = 400
    rows_per_group: int = 12
    policy: str = "equal-domains-16k"
    token_budget: int | None = None
    seed: int = 0
    tokenizer_name: str = BASE_MODEL


def build_mix(
    cfg: MixBuild, source: Callable[[], Iterator[dict[str, Any]]] | None = None
) -> dict[str, Any]:
    sampled: list[RowGroup] = []
    if source is None:
        fs = HfFileSystem()
        sampled = sample_row_groups(all_row_groups(fs), cfg.groups, cfg.seed)

        def source() -> Iterator[dict[str, Any]]:
            return stream_sampled_rows(sampled, cfg.rows_per_group, fs)

    stats = SliceStats()
    collector = Collector()
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cfg.out.parent) as cache:
        ds = Dataset.from_generator(
            labelled_rows,
            gen_kwargs={
                "rows": 0,
                "num_shards": 1,
                "tokenizer_name": cfg.tokenizer_name,
                "stats": stats,
                "source": source,
                "collector": collector,
            },
            features=FEATURES,
            cache_dir=cache,
            keep_in_memory=False,
        )
        profiles = collector.profiles
        if len(profiles) != len(ds):
            profiles = [
                profile(r["mode_ids"], r["loss_mask"], r["domain"], r["complete"]) for r in ds
            ]
        chosen = select(profiles, CANDIDATES[cfg.policy], cfg.token_budget, cfg.seed)
        ds.select(sorted(chosen)).save_to_disk(str(cfg.out))
    picked = [profiles[i] for i in chosen]
    report = {
        "dataset": DATASET,
        "tokenizer": cfg.tokenizer_name,
        "sampling": {
            "row_groups": len(sampled),
            "rows_per_group": cfg.rows_per_group,
            "seed": cfg.seed,
            "groups": [f"{g.path.rsplit('/', 1)[-1]}#{g.index}" for g in sampled],
        },
        "sample_summary": mix_summarize(profiles),
        "policies": compare_policies(profiles, CANDIDATES, cfg.seed),
        "chosen_policy": cfg.policy,
        "selected": {
            **mix_label_shares(picked),
            "by_domain": dict(Counter(r.domain for r in picked)),
            "total_tokens": sum(r.total_tokens for r in picked),
        },
    }
    (cfg.out / "stats.json").write_text(json.dumps(report, indent=2) + "\n")
    with (cfg.out / "sample.jsonl").open("w") as handle:
        for item in collector.samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a labelled OpenThoughts3 training mix")
    parser.add_argument("--out", type=Path, default=Path("artifacts/data/ot3-mix"))
    parser.add_argument("--tokenizer", default=BASE_MODEL)
    parser.add_argument("--groups", type=int, default=400, help="parquet row groups to sample")
    parser.add_argument("--rows-per-group", type=int, default=12)
    parser.add_argument("--policy", choices=sorted(CANDIDATES), default="equal-domains-16k")
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", action="store_true", help="old layout: first rows of shards")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args(argv)
    if args.head:
        report = build(args.rows, args.shards, args.out, args.tokenizer)
    else:
        cfg = MixBuild(
            out=args.out,
            groups=args.groups,
            rows_per_group=args.rows_per_group,
            policy=args.policy,
            token_budget=args.token_budget,
            seed=args.seed,
            tokenizer_name=args.tokenizer,
        )
        report = build_mix(cfg)
    print(json.dumps(report, indent=2)[:20000])


if __name__ == "__main__":
    main()
